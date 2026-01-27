import gzip
import json
import logging
import random
import tempfile
from urllib.parse import quote

from flask import Flask, url_for, redirect, render_template, request, render_template_string, \
    send_from_directory, send_file
from flask_caching import Cache

from pysrc.app.admin import admin_bp, init_admin
from pysrc.app.api import api_bp
from pysrc.app.messages import *
from pysrc.app.predefined import are_predefined_jobs_ready, PREDEFINED_JOBS, is_semantic_predefined, \
    PREDEFINED_TASKS_READY_KEY, is_paper_predefined, is_terms_predefined
from pysrc.app.reports import load_or_save_result_data, preprocess_string
from pysrc.celery.pubtrends_celery import pubtrends_celery
from pysrc.celery.tasks_main import analyze_search_paper, analyze_search_terms, analyze_pubmed_search, \
    analyze_semantic_search
from pysrc.config import *
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.plot.plot_app import prepare_graph_data, prepare_papers_data, prepare_paper_data, prepare_result_data, \
    prepare_search_string
from pysrc.papers.questions.questions import get_relevant_papers
from pysrc.papers.utils import trim_query, IDS_ANALYSIS_TYPE, PAPER_ANALYSIS_TYPE
from pysrc.services.embeddings_service import is_embeddings_service_available, is_embeddings_service_ready, \
    is_texts_embeddings_available
from pysrc.services.semantic_search_service import is_semantic_search_service_available
from pysrc.version import VERSION

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

pubtrends_app = Flask(__name__)

if not pubtrends_app.config['TESTING'] and not pubtrends_app.config['DEBUG']:
    pubtrends_app.config['CACHE_TYPE'] = 'RedisCache'
pubtrends_app.config['CACHE_REDIS_URL'] = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379')
pubtrends_app.config['CACHE_DEFAULT_TIMEOUT'] = 600  # 10 minutes

cache = Cache(pubtrends_app)

# Register blueprints
pubtrends_app.register_blueprint(admin_bp)
pubtrends_app.register_blueprint(api_bp)

#####################
# Configure logging #
#####################

# Deployment and development
LOG_PATHS = ['/logs', os.path.expanduser('~/.pubtrends/logs')]
for p in LOG_PATHS:
    if os.path.isdir(p):
        logfile = os.path.join(p, 'app.log')
        break
else:
    raise RuntimeError('Failed to configure main log file')

logging.basicConfig(filename=logfile,
                    filemode='a',
                    format='[%(asctime)s,%(msecs)03d: %(levelname)s/%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Check to see if our Flask application is being run directly or through Gunicorn,
# and then set your Flask application log level the same as Gunicorn’s.
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    pubtrends_app.logger.setLevel(gunicorn_logger.level)

logger = pubtrends_app.logger


#############
# Main page #
#############

@pubtrends_app.route('/robots.txt')
@pubtrends_app.route('/sitemap.xml')
@pubtrends_app.route('/feedback.js')
@pubtrends_app.route('/wordcloud.js')
@pubtrends_app.route('/navigate.js')
@pubtrends_app.route('/smile.svg')
@pubtrends_app.route('/meh.svg')
@pubtrends_app.route('/frown.svg')
def static_from_root():
    return send_from_directory(pubtrends_app.static_folder, request.path[1:])


SEMANTIC_SEARCH_AVAILABLE = PUBTRENDS_CONFIG.feature_semantic_search_enabled and is_semantic_search_service_available()


def log_request(r):
    v = f'addr:{r.remote_addr} args:{json.dumps(r.args)}'
    if r.method == 'POST':
        v += f' form:{json.dumps(r.form)}'
    try:
        if r.json is not None:
            v += f' json:{json.dumps(r.json)}'
    except Exception:
        pass
    return v


@pubtrends_app.route('/')
@cache.cached(unless=lambda: not pubtrends_app.config.get(PREDEFINED_TASKS_READY_KEY, False))
def index():
    if is_embeddings_service_available() and not is_embeddings_service_ready():
        return render_template('init.html', version=VERSION, message=SERVICE_LOADING_INITIALIZING)
    if not are_predefined_jobs_ready(pubtrends_app, pubtrends_celery):
        return render_template('init.html', version=VERSION, message=SERVICE_LOADING_PREDEFINED_EXAMPLES)

    search_example_message = ''
    search_example_source = ''
    search_example_terms = []
    search_example_papers = []
    semantic_search_example_terms = []

    if len(PREDEFINED_JOBS) != 0:
        search_example_source, example_terms = random.choice(list(PREDEFINED_JOBS.items()))
        search_example_papers = [(*q.split('='), jobid) for q, jobid in example_terms
                                 if is_paper_predefined(jobid)]
        search_example_message = 'Try one of our examples for ' + search_example_source
        search_example_terms = [(q, jobid) for q, jobid in example_terms
                                if is_terms_predefined(jobid)]
        semantic_search_example_terms = [(q, jobid) for q, jobid in example_terms
                                         if is_semantic_predefined(jobid)]

    return render_template('main.html',
                           version=VERSION,
                           limits=SHOW_MAX_ARTICLES_OPTIONS,
                           default_limit=SHOW_MAX_ARTICLES_DEFAULT,
                           topics_variants=SHOW_TOPICS_OPTIONS,
                           default_topics=SHOW_TOPICS_DEFAULT,
                           expand_variants=range(PAPER_EXPAND_STEPS + 1),
                           default_expand=PAPER_EXPAND_STEPS,
                           max_papers=max_number_of_articles,
                           pm_enabled=PUBTRENDS_CONFIG.pm_enabled,
                           ss_enabled=PUBTRENDS_CONFIG.ss_enabled,
                           search_example_message=search_example_message,
                           search_example_source=search_example_source,
                           search_example_terms=search_example_terms,
                           search_example_papers=search_example_papers,
                           semantic_search_example_terms=semantic_search_example_terms,
                           semantic_search_enabled=SEMANTIC_SEARCH_AVAILABLE)


@pubtrends_app.route('/about.html', methods=['GET'])
def about():
    return render_template('about.html', version=VERSION)


############################
# Search form POST methods #
############################

def value_to_bool(value):
    return str(value).lower() == 'on'


def bool_to_value(value):
    return 'on' if value else 'off'


@pubtrends_app.route('/search_terms', methods=['POST'])
def search_terms():
    logger.info(f'{LOG_SEARCH_TERMS} {log_request(request)}')
    try:
        query = request.form.get('query')  # Original search query
        source = request.form.get('source')  # Pubmed or Semantic Scholar
        sort = request.form.get('sort')  # Sort order
        limit = request.form.get('limit')  # Limit
        pubmed_syntax = request.form.get('pubmed-syntax') == 'on'
        noreviews = value_to_bool(request.form.get('noreviews'))
        min_year = request.form.get('min_year')  # Minimal year of publications
        max_year = request.form.get('max_year')  # Maximal year of publications
        topics = request.form.get('topics')  # Topics sizes

        # PubMed syntax
        if query and source and limit and topics and pubmed_syntax:
            job = analyze_pubmed_search.delay(query=query, sort=sort, limit=int(limit), topics=topics,
                                              test=pubtrends_app.config['TESTING'])
            return redirect(url_for('.process',
                                    jobid=job.id,
                                    query=trim_query(query), source=source, sort='', limit=limit))

        # Regular search syntax
        if query and source and sort and limit and topics:
            job = analyze_search_terms.delay(source, query=query, limit=int(limit), sort=sort,
                                             noreviews=noreviews, min_year=min_year, max_year=max_year,
                                             topics=topics,
                                             test=pubtrends_app.config['TESTING'])
            return redirect(url_for('.process',
                                    query=trim_query(query), source=source, sort=sort, limit=limit,
                                    jobid=job.id))

        logger.error(f'{LOG_SEARCH_TERMS}_{ERROR} {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'{LOG_SEARCH_TERMS}_{EXCEPTION} {log_request(request)} {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@pubtrends_app.route('/search_paper', methods=['POST'])
def search_paper():
    logger.info(f'{LOG_SEARCH_PAPER} {log_request(request)}')
    try:
        source = request.form.get('source')  # Pubmed or Semantic Scholar
        key = request.form.get('key')
        value = request.form.get('value')
        limit = request.form.get('limit')
        noreviews = value_to_bool(request.form.get('noreviews'))
        expand = request.form.get('expand')
        topics = request.form.get('topics')
        if source and key and value and limit and topics:
            job = analyze_search_paper.delay(source, None, key, value, expand, limit, noreviews, topics,
                                             test=pubtrends_app.config['TESTING'])
            return redirect(url_for('.process',
                                    query=trim_query(f'{key}={value}'), source=source, sort='', limit=limit,
                                    jobid=job.id))
        logger.error(f'{LOG_SEARCH_TERMS} {ERROR} {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'{LOG_SEARCH_PAPER} {EXCEPTION} {log_request(request)} {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@pubtrends_app.route('/search_semantic', methods=['POST'])
def search_semantic():
    logger.info(f'{LOG_SEMANTIC_SEARCH} {log_request(request)}')
    try:
        query = request.form.get('query')  # Original search query
        source = request.form.get('source')  # Pubmed or Semantic Scholar
        limit = request.form.get('limit')  # Limit
        noreviews = value_to_bool(request.form.get('noreviews'))
        topics = request.form.get('topics')  # Topics sizes
        if query and source and limit and topics:
            job = analyze_semantic_search.delay(
                source, query=query, limit=int(limit), noreviews=noreviews,
                topics=topics, test=pubtrends_app.config['TESTING']
            )
            return redirect(url_for('.process',
                                    query=trim_query(query), source=source, sort='semantic', limit=limit,
                                    jobid=job.id))
        logger.error(f'{LOG_SEMANTIC_SEARCH} {ERROR} {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'{LOG_SEMANTIC_SEARCH} {EXCEPTION} {log_request(request)} {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


####################################
# Analysis progress and cancelling #
####################################

@pubtrends_app.route('/process')
def process():
    """ Rendering process.html for task being queued or executed at the moment """
    logger.info(f'{LOG_PROCESS} {log_request(request)}')
    jobid = request.values.get('jobid')
    if jobid:
        query = request.args.get('query')
        source = request.args.get('source')
        sort = request.args.get('sort')
        limit = request.args.get('limit') or PUBTRENDS_CONFIG.show_max_articles_default_value
        return render_template(
            'process.html',
            jobid=jobid,
            query=query, source=source, sort=sort, limit=limit,
            version=VERSION
        )
    logger.error(f'{LOG_PROCESS} {ERROR} {log_request(request)}')
    return render_template_string(SOMETHING_WENT_WRONG_SEARCH)


@pubtrends_app.route('/status/<jobid>')
def status(jobid):
    """ Check tasks status being executed by Celery """
    try:
        job = pubtrends_celery.AsyncResult(jobid)
        if job is None:
            return json.dumps(dict(state='FAILURE', message=f'Unknown task id {jobid}'))
        job_state, job_result = job.state, job.result
        if job_state == 'PROGRESS':
            return json.dumps(dict(state=job_state, log=job_result['log'],
                                   progress=int(100.0 * job_result['current'] / job_result['total'])))
        elif job_state == 'SUCCESS':
            logger.info(f'{LOG_STATUS} {SUCCESS} {log_request(request)}')
            data, _ = job.result
            analysis_type = data['analysis_type']
            query = quote(data['search_query'])
            if analysis_type == IDS_ANALYSIS_TYPE:
                href = f'/result?query={query}&jobid={jobid}'
            elif analysis_type == PAPER_ANALYSIS_TYPE:
                href = f'/paper?query={query}&jobid={jobid}'
            else:
                raise Exception(f'Unknown analysis type {analysis_type}')
            return json.dumps(dict(state=job_state, progress=100, redirect=href))
        elif job_state == 'FAILURE':
            is_search_error = isinstance(job_result, SearchError)
            logger.info(f'{LOG_STATUS} {ERROR}. Search error: {is_search_error}. {log_request(request)}')
            return json.dumps(dict(state=job_state, message=str(job_result).replace('\\n', '\n').replace('\\t', '\t'),
                                   search_error=is_search_error))
        elif job_state == 'STARTED' or job_state == 'PENDING':
            return json.dumps(dict(state=job_state, message='Task is starting, please wait...'))
        elif job_state == 'REVOKED':
            return json.dumps(
                dict(state=job_state, message='Task was cancelled, please <a href="/">rerun</a> your search.'))
        else:
            return json.dumps(dict(state='FAILURE', message=f'Illegal task state {job_state}'))
    except Exception as e:
        logger.exception(f'{LOG_STATUS} {EXCEPTION} {log_request(request)} {e}')
        return json.dumps(dict(state='FAILURE', message=f'Error checking task status: {e}'))


@pubtrends_app.route('/cancel/<jobid>', methods=['POST'])
def cancel(jobid):
    logger.info(f'{LOG_CANCEL} {log_request(request)}')
    try:
        pubtrends_celery.control.revoke(jobid, terminate=True)
        return json.dumps(dict(state='CANCELLED', message=f'Successfully cancelled task {jobid}'))
    except Exception as e:
        logger.exception(f'{LOG_CANCEL} {EXCEPTION} {log_request(request)} {e}')
        return json.dumps(dict(state='FAILURE', message=f'Error cancelling task: {e}'))


####################################
# Results handlers #
####################################


@pubtrends_app.route('/result')
@cache.cached(query_string=True)
def result():
    logger.info(f'/result {log_request(request)}')
    try:
        jobid = request.args.get('jobid')
        if not jobid:
            logger.error(f'{LOG_RESULT} {ERROR} {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG_SEARCH)
        data = load_or_save_result_data(pubtrends_celery, jobid)
        if data is not None:
            logger.info(f'{LOG_RESULT} {SUCCESS} {log_request(request)}')
            return render_template('result.html',
                                   query=trim_query(data.search_query),
                                   source=data.source,
                                   limit=data.limit,
                                   sort=data.sort or '',
                                   max_graph_size=MAX_GRAPH_SIZE,
                                   version=VERSION,
                                   is_predefined=True,
                                   **prepare_result_data(PUBTRENDS_CONFIG, data))
        logger.error(f'{LOG_RESULT} {ERROR} {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'{LOG_RESULT} {EXCEPTION} {log_request(request)} {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@pubtrends_app.route('/paper')
@cache.cached(query_string=True)
def paper():
    logger.info(f'/paper {log_request(request)}')
    try:
        jobid = request.values.get('jobid')
        pid = request.args.get('id')
        if jobid:
            data = load_or_save_result_data(pubtrends_celery, jobid)
            if data is not None:
                logger.info(f'{LOG_PAPER} {SUCCESS} {log_request(request)}')
                return render_template('paper.html',
                                       **prepare_paper_data(data, pid),
                                       max_graph_size=PUBTRENDS_CONFIG.max_graph_size,
                                       version=VERSION)
        logger.error(f'{LOG_PAPER} {ERROR} {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'{LOG_PAPER} {EXCEPTION} {log_request(request)} {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@pubtrends_app.route('/graph')
@cache.cached(query_string=True)
def graph():
    logger.info(f'/graph {log_request(request)}')
    try:
        jobid = request.args.get('jobid')
        pid = request.args.get('pid')
        if jobid:
            data = load_or_save_result_data(pubtrends_celery, jobid)
            if data is not None:
                logger.info(f'{LOG_GRAPH} {SUCCESS} {log_request(request)}')
                return render_template(
                    'graph.html',
                    version=VERSION,
                    **prepare_graph_data(data, pid)
                )
        logger.error(f'{LOG_GRAPH} {ERROR} {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'{LOG_GRAPH} {EXCEPTION} {log_request(request)} {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@pubtrends_app.route('/papers')
@cache.cached(query_string=True)
def papers():
    logger.info(f'{LOG_PAPERS} {log_request(request)}')
    try:
        jobid = request.args.get('jobid')
        if jobid:
            data = load_or_save_result_data(pubtrends_celery, jobid)
            if data is not None:
                topic = request.args.get('topic')
                word = request.args.get('word')
                author = request.args.get('author')
                journal = request.args.get('journal')
                papers_list = request.args.get('papers_list')

                logger.info(f'{LOG_PAPERS} {SUCCESS} {log_request(request)}')
                comp, search_string = prepare_search_string(topic, word, author, journal, papers_list)
                export_name = preprocess_string(f'{data.search_query}-{search_string}')
                papers_data = prepare_papers_data(
                    data, comp, word, author, journal, papers_list
                )
                return render_template('papers.html',
                                       version=VERSION,
                                       source=data.source,
                                       query=trim_query(data.search_query),
                                       search_string=search_string,
                                       limit=data.limit or '',
                                       sort=data.sort or '',
                                       noreviews='on' if data.noreviews else '',
                                       min_year=data.min_year or '',
                                       max_year=data.max_year or '',
                                       export_name=export_name,
                                       papers=papers_data)
        logger.error(f'{LOG_PAPERS} {ERROR} {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'{LOG_PAPERS} {EXCEPTION} {log_request(request)} {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@pubtrends_app.route('/export/<jobid>', methods=['GET'])
def export(jobid):
    logger.info(f'{LOG_EXPORT} {log_request(request)}')
    try:
        data = load_or_save_result_data(pubtrends_celery, jobid)
        if data:
            logger.info(f'{LOG_EXPORT} {SUCCESS} {log_request(request)}')
            with tempfile.TemporaryDirectory() as tmpdir:
                name = preprocess_string(f'{data.source}-{data.search_query}')
                path = os.path.join(tmpdir, f'{name}.json.gz')
                with gzip.open(path, 'w') as f:
                    f.write(json.dumps(data.to_json()).encode('utf-8'))
                    return send_file(path, as_attachment=True)
        logger.error(f'{LOG_EXPORT} {ERROR} {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'{LOG_EXPORT} {EXCEPTION} {log_request(request)} {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


##########################
# Feedback functionality #
##########################

@pubtrends_app.route('/feedback', methods=['POST'])
def feedback():
    logger.info(f'{LOG_FEEDBACK} {log_request(request)}')
    jobid = request.form.get('jobid')
    key = request.form.get('key')
    value = request.form.get('value')
    if key and value and jobid:
        logger.info('Feedback ' + json.dumps(dict(key=key, value=value, jobid=jobid)))
        logger.info(f'{LOG_FEEDBACK} {SUCCESS} {log_request(request)}')
    else:
        logger.error(f'{LOG_FEEDBACK} {ERROR} {log_request(request)}')
    return render_template_string('Thanks you for the feedback!'), 200


##########################
# Question functionality #
##########################

@pubtrends_app.route('/question', methods=['POST'])
def question():
    logger.info(f'{LOG_QUESTION} {log_request(request)}')
    try:
        data = request.json
        jobid = data.get('jobid')
        question_text = data.get('question')

        if not question_text or not jobid:
            logger.error(f'{LOG_QUESTION} {ERROR} {log_request(request)}')
            return {'status': 'error', 'message': 'Missing question or jobid'}, 400

        # Check if text embeddings are available
        if not is_texts_embeddings_available():
            logger.error(f'{LOG_QUESTION} {ERROR} {log_request(request)}')
            return {'status': 'error', 'message': 'Text embeddings not available'}, 400

        # Load result data
        data = load_or_save_result_data(pubtrends_celery, jobid)
        if data is None:
            logger.error(f'{LOG_QUESTION} {ERROR} {log_request(request)}')
            return {'status': 'error', 'message': 'No data found for jobid'}, 404

        papers = get_relevant_papers(
            data.search_query,
            question_text,
            data,
            QUESTIONS_RELEVANCE_THRESHOLD,
            QUESTIONS_ANSWERS_TOP_N
        )
        logger.info(f'{LOG_QUESTION} {SUCCESS} {log_request(request)}')
        return {'status': 'success', 'papers': papers}, 200
    except Exception as e:
        logger.exception(f'{LOG_QUESTION} {EXCEPTION} {log_request(request)} {e}')
        return {'status': 'error', 'message': str(e)}, 500


#######################
# Admin functionality #
#######################

init_admin(pubtrends_app, pubtrends_celery, logfile)


#######################
# Additional features #
#######################

# Application
def get_app():
    return pubtrends_app


# With debug=True, the Flask server will auto-reload on changes
if __name__ == '__main__':
    pubtrends_app.run(host='0.0.0.0', debug=True, extra_files=['templates/'], port=5000)
