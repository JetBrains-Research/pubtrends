import gzip
import json
import logging
import os
import random
import tempfile
from threading import Lock
from urllib.parse import quote

from celery.result import AsyncResult
from flask import Flask, url_for, redirect, render_template, request, render_template_string, \
    send_from_directory, send_file
from flask_caching import Cache

from pysrc.app.admin.admin import configure_admin_functions
from pysrc.app.messages import SOMETHING_WENT_WRONG_SEARCH, ERROR_OCCURRED, \
    SERVICE_LOADING_PREDEFINED_EXAMPLES, SERVICE_LOADING_INITIALIZING
from pysrc.app.reports import get_predefined_jobs, \
    load_result_data, _predefined_example_params_by_jobid, preprocess_string, load_paper_data
from pysrc.celery.pubtrends_celery import pubtrends_celery
from pysrc.celery.tasks_main import analyze_search_paper, analyze_search_terms, analyze_pubmed_search, analyze_id_list, \
    analyze_semantic_search
from pysrc.config import PubtrendsConfig
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.plot.plot_app import prepare_graph_data, prepare_papers_data, prepare_paper_data, prepare_result_data
from pysrc.papers.questions.questions import get_relevant_papers
from pysrc.services.embeddings_service import is_embeddings_service_available, is_embeddings_service_ready, \
    is_texts_embeddings_available
from pysrc.papers.utils import trim_query, IDS_ANALYSIS_TYPE, PAPER_ANALYSIS_TYPE, SORT_MOST_CITED
from pysrc.services.semantic_search_service import is_semantic_search_service_available
from pysrc.version import VERSION

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

pubtrends_app = Flask(__name__)

if not pubtrends_app.config['TESTING'] and not pubtrends_app.config['DEBUG']:
    pubtrends_app.config['CACHE_TYPE'] = 'RedisCache'
pubtrends_app.config['CACHE_REDIS_URL'] = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379')
pubtrends_app.config['CACHE_DEFAULT_TIMEOUT'] = 600  # 10 minutes

cache = Cache(pubtrends_app)

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
@pubtrends_app.route('/style.css')
@pubtrends_app.route('/about_graph_explorer.png')
@pubtrends_app.route('/about_hot_papers.png')
@pubtrends_app.route('/about_keywords.png')
@pubtrends_app.route('/about_main.png')
@pubtrends_app.route('/about_network.png')
@pubtrends_app.route('/about_pubtrends_scheme.png')
@pubtrends_app.route('/about_report.png')
@pubtrends_app.route('/about_review.png')
@pubtrends_app.route('/about_top_cited_papers.png')
@pubtrends_app.route('/about_topic.png')
@pubtrends_app.route('/about_topics_by_year.png')
@pubtrends_app.route('/about_topics_hierarchy.png')
@pubtrends_app.route('/smile.svg')
@pubtrends_app.route('/meh.svg')
@pubtrends_app.route('/frown.svg')
def static_from_root():
    return send_from_directory(pubtrends_app.static_folder, request.path[1:])


PREDEFINED_JOBS = get_predefined_jobs(PUBTRENDS_CONFIG)


def log_request(r):
    return f'addr:{r.remote_addr} args:{json.dumps(r.args)}'


@pubtrends_app.route('/')
def index():
    if is_embeddings_service_available() and not is_embeddings_service_ready():
        return render_template('init.html', version=VERSION, message=SERVICE_LOADING_INITIALIZING)
    if not are_predefined_jobs_ready():
        return render_template('init.html', version=VERSION, message=SERVICE_LOADING_PREDEFINED_EXAMPLES)

    search_example_message = ''
    search_example_source = ''
    search_example_terms = []

    if len(PREDEFINED_JOBS) != 0:
        search_example_source, search_example_terms = random.choice(list(PREDEFINED_JOBS.items()))
        search_example_message = 'Try one of our examples for ' + search_example_source

    semantic_search_enabled = \
        PUBTRENDS_CONFIG.feature_semantic_search_enabled and is_semantic_search_service_available()
    return render_template('main.html',
                           version=VERSION,
                           limits=PUBTRENDS_CONFIG.show_max_articles_options,
                           default_limit=PUBTRENDS_CONFIG.show_max_articles_default_value,
                           topics_variants=PUBTRENDS_CONFIG.show_topics_options,
                           default_topics=PUBTRENDS_CONFIG.show_topics_default_value,
                           expand_variants=range(PUBTRENDS_CONFIG.paper_expands_steps + 1),
                           default_expand=PUBTRENDS_CONFIG.paper_expands_steps,
                           max_papers=PUBTRENDS_CONFIG.max_number_of_articles,
                           pm_enabled=PUBTRENDS_CONFIG.pm_enabled,
                           ss_enabled=PUBTRENDS_CONFIG.ss_enabled,
                           search_example_message=search_example_message,
                           search_example_source=search_example_source,
                           search_example_terms=search_example_terms,
                           semantic_search_enabled=semantic_search_enabled)


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
    logger.info(f'/search_terms {log_request(request)}')
    query = request.form.get('query')  # Original search query
    source = request.form.get('source')  # Pubmed or Semantic Scholar
    sort = request.form.get('sort')  # Sort order
    limit = request.form.get('limit')  # Limit
    pubmed_syntax = request.form.get('pubmed-syntax') == 'on'
    noreviews = value_to_bool(request.form.get('noreviews'))
    min_year = request.form.get('min_year')  # Minimal year of publications
    max_year = request.form.get('max_year')  # Maximal year of publications
    topics = request.form.get('topics')  # Topics sizes

    try:
        # PubMed syntax
        if query and source and limit and topics and pubmed_syntax:
            # Regular analysis
            job = analyze_pubmed_search.delay(query=query, sort=sort, limit=limit, topics=topics,
                                              test=pubtrends_app.config['TESTING'])
            return redirect(
                url_for('.process', source='Pubmed', query=trim_query(query), limit=limit, sort='',
                        noreviews='on' if noreviews else '',
                        min_year=min_year or '', max_year=max_year or '',
                        topics=topics,
                        jobid=job.id))

        # Regular search syntax
        if query and source and sort and limit and topics:
            job = analyze_search_terms.delay(source, query=query, limit=int(limit), sort=sort,
                                             noreviews=noreviews, min_year=min_year, max_year=max_year,
                                             topics=topics,
                                             test=pubtrends_app.config['TESTING'])
            return redirect(
                url_for('.process', query=trim_query(query), source=source, limit=limit, sort=sort,
                        noreviews='on' if noreviews else '',
                        min_year=min_year or '', max_year=max_year or '',
                        topics=topics,
                        jobid=job.id))
        logger.error(f'/search_terms error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/search_terms exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@pubtrends_app.route('/search_paper', methods=['POST'])
def search_paper():
    logger.info(f'/search_paper {log_request(request)}')
    data = request.form
    try:
        if 'source' in data and 'key' in data and 'value' in data and 'topics' in data:
            source = data.get('source')  # Pubmed or Semantic Scholar
            key = data.get('key')
            value = data.get('value')
            limit = data.get('limit')
            noreviews = value_to_bool(request.form.get('noreviews'))
            expand = data.get('expand')
            topics = data.get('topics')
            job = analyze_search_paper.delay(source, None, key, value, expand, limit, noreviews, topics,
                                             test=pubtrends_app.config['TESTING'])
            return redirect(url_for('.process', query=trim_query(f'Papers {key}={value}'),
                                    analysis_type=PAPER_ANALYSIS_TYPE,
                                    key=key, value=value,
                                    source=source, expand=expand, limit=limit,
                                    noreviews='on' if noreviews else '',
                                    topics=topics,
                                    jobid=job.id))
        logger.error(f'/search_paper error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/search_paper exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500

@pubtrends_app.route('/search_semantic', methods=['POST'])
def search_semantic():
    logger.info(f'/search_semantic {log_request(request)}')
    query = request.form.get('query')  # Original search query
    source = request.form.get('source')  # Pubmed or Semantic Scholar
    limit = request.form.get('limit')  # Limit
    noreviews = value_to_bool(request.form.get('noreviews'))
    topics = request.form.get('topics')  # Topics sizes

    try:
        if query and source and limit and topics:
            job = analyze_semantic_search.delay(
                source, query=query, limit=int(limit), noreviews=noreviews,
                topics=topics, test=pubtrends_app.config['TESTING']
            )
            return redirect(
                url_for('.process', query=trim_query(query), source=source, limit=limit,
                        noreviews='on' if noreviews else '',
                        sort = '', min_year = '', max_year= '',
                        topics=topics,
                        jobid=job.id))
        logger.error(f'/search_semantic error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/search_semantic exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


####################################
# Analysis progress and cancelling #
####################################

@pubtrends_app.route('/process')
def process():
    """ Rendering process.html for task being queued or executed at the moment """
    if len(request.args) > 0:
        jobid = request.values.get('jobid')

        if not jobid:
            logger.error(f'/process error wrong request {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG_SEARCH)

        query = request.args.get('query') or ''
        analysis_type = request.values.get('analysis_type')
        source = request.values.get('source')
        noreviews = value_to_bool(request.form.get('noreviews'))
        min_year = request.form.get('min_year')  # Minimal year of publications
        max_year = request.form.get('max_year')  # Maximal year of publications
        topics = request.values.get('topics')

        if analysis_type == IDS_ANALYSIS_TYPE:
            logger.info(f'/process ids {log_request(request)}')
            return render_template('process.html',
                                   redirect_page='result',  # redirect in case of success
                                   redirect_args=dict(query=quote(trim_query(query)), source=source, jobid=jobid,
                                                      limit=analysis_type, sort='',
                                                      noreviews='on' if noreviews else '',
                                                      min_year=min_year or '', max_year=max_year or '',
                                                      topics=topics),
                                   query=trim_query(query), source=source, limit=analysis_type, sort='',
                                   jobid=jobid, version=VERSION)

        elif analysis_type == PAPER_ANALYSIS_TYPE:
            logger.info(f'/process paper analysis {log_request(request)}')
            key = request.args.get('key')
            value = request.args.get('value')
            limit = request.args.get('limit') or PUBTRENDS_CONFIG.show_max_articles_default_value
            noreviews = value_to_bool(request.args.get('noreviews'))
            if ';' in value:
                return render_template('process.html',
                                       redirect_page='result',  # redirect in case of success
                                       redirect_args=dict(query=quote(trim_query(f'Papers {key}={value}')),
                                                          source=source, jobid=jobid, sort='', limit=limit,
                                                          noreviews='on' if noreviews else '',
                                                          min_year=min_year or '', max_year=max_year or '',
                                                          topics=topics),
                                       query=trim_query(query), source=source,
                                       jobid=jobid, version=VERSION)
            else:
                return render_template('process.html',
                                       redirect_page='paper',  # redirect in case of success
                                       redirect_args=dict(query=quote(trim_query(f'Papers {key}={value}')),
                                                          source=source, jobid=jobid, sort='',
                                                          key=key, value=value, limit=limit,
                                                          noreviews='on' if noreviews else '',
                                                          min_year=min_year or '', max_year=max_year or '',
                                                          topics=topics),
                                       query=trim_query(query), source=source,
                                       jobid=jobid, version=VERSION)

        elif query:  # This option should be the last default
            logger.info(f'/process regular search {log_request(request)}')
            limit = request.args.get('limit') or PUBTRENDS_CONFIG.show_max_articles_default_value
            sort = request.args.get('sort')
            noreviews = value_to_bool(request.args.get('noreviews'))
            return render_template('process.html',
                                   redirect_page='result',  # redirect in case of success
                                   redirect_args=dict(query=quote(trim_query(query)),
                                                      source=source, limit=limit, sort=sort,
                                                      noreviews='on' if noreviews else '',
                                                      min_year=min_year or '', max_year=max_year or '',
                                                      topics=topics, jobid=jobid),
                                   query=trim_query(query), source=source,
                                   limit=limit, sort=sort,
                                   jobid=jobid, version=VERSION)

    logger.error(f'/process error {log_request(request)}')
    return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400


@pubtrends_app.route('/status')
def status():
    """ Check tasks status being executed by Celery """
    jobid = request.values.get('jobid')
    if jobid:
        job = AsyncResult(jobid, app=pubtrends_celery)
        if job is None:
            return json.dumps(dict(state='FAILURE', message=f'Unknown task id {jobid}'))
        job_state, job_result = job.state, job.result
        if job_state == 'PROGRESS':
            return json.dumps(dict(state=job_state, log=job_result['log'],
                                   progress=int(100.0 * job_result['current'] / job_result['total'])))
        elif job_state == 'SUCCESS':
            return json.dumps(dict(state=job_state, progress=100))
        elif job_state == 'FAILURE':
            is_search_error = isinstance(job_result, SearchError)
            logger.info(f'/status failure. Search error: {is_search_error}. {log_request(request)}')
            return json.dumps(dict(state=job_state, message=str(job_result).replace('\\n', '\n').replace('\\t', '\t'),
                                   search_error=is_search_error))
        elif job_state == 'STARTED':
            return json.dumps(dict(state=job_state, message='Task is starting, please wait...'))
        elif job_state == 'PENDING':
            return json.dumps(dict(state=job_state, message='Task is in queue, please wait...'))
        elif job_state == 'REVOKED':
            return json.dumps(
                dict(state=job_state, message='Task was cancelled, please <a href="/">rerun</a> your search.'))
        else:
            return json.dumps(dict(state='FAILURE', message=f'Illegal task state {job_state}'))
    # no jobid
    return json.dumps(dict(state='FAILURE', message='No task id'))


@pubtrends_app.route('/cancel')
def cancel():
    if len(request.args) > 0:
        jobid = request.values.get('jobid')
        if jobid:
            pubtrends_celery.control.revoke(jobid, terminate=True)
            return json.dumps(dict(state='CANCELLED', message=f'Successfully cancelled task {jobid}'))
        else:
            return json.dumps(dict(state='FAILURE', message=f'Failed to cancel task {jobid}'))
    return json.dumps(dict(state='FAILURE', message='Unknown task id'))


####################################
# Results handlers #
####################################


@pubtrends_app.route('/result')
@cache.cached(query_string=True)
def result():
    jobid = request.args.get('jobid')
    query = request.args.get('query')
    source = request.args.get('source')
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    noreviews = value_to_bool(request.form.get('noreviews'))
    min_year = request.form.get('min_year')  # Minimal year of publications
    max_year = request.form.get('max_year')  # Maximal year of publications
    topics = request.args.get('topics')
    try:
        if jobid:
            data = load_result_data(jobid, source, query, sort, limit, noreviews, min_year, max_year, pubtrends_celery)
            if data is not None:
                logger.info(f'/result success {log_request(request)}')
                return render_template('result.html',
                                       query=trim_query(query),
                                       source=source,
                                       limit=limit,
                                       sort=sort,
                                       max_graph_size=PUBTRENDS_CONFIG.max_graph_size,
                                       version=VERSION,
                                       is_predefined=True,
                                       **prepare_result_data(PUBTRENDS_CONFIG, data))
            logger.info(f'/result No job or out-of-date job, restart it {log_request(request)}')
            analyze_search_terms.apply_async(
                args=[source, query, sort, int(limit), noreviews, min_year, max_year, topics, pubtrends_app.config['TESTING']],
                task_id=jobid
            )
            return redirect(
                url_for('.process', query=trim_query(query), source=source, limit=limit, sort=sort,
                        noreviews='on' if noreviews else '',
                        min_year=min_year or '', max_year=max_year or '',
                        topics=topics,
                        jobid=jobid))
        else:
            logger.error(f'/result error wrong request {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/result exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@pubtrends_app.route('/graph')
@cache.cached(query_string=True)
def graph():
    jobid = request.values.get('jobid')
    query = request.args.get('query')
    source = request.args.get('source')
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    noreviews = value_to_bool(request.form.get('noreviews'))
    min_year = request.form.get('min_year')  # Minimal year of publications
    max_year = request.form.get('max_year')  # Maximal year of publications
    pid = request.args.get('id')
    try:
        if jobid:
            data = load_result_data(jobid, source, query, sort, limit, noreviews, min_year, max_year, pubtrends_celery)
            if data is not None:
                logger.info(f'/graph success {log_request(request)}')
                return render_template(
                    'graph.html',
                    version=VERSION,
                    **prepare_graph_data(PUBTRENDS_CONFIG, data, pid)
                )
            logger.error(f'/graph error job id {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
        else:
            logger.error(f'/graph error wrong request {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/graph exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@pubtrends_app.route('/paper')
@cache.cached(query_string=True)
def paper():
    jobid = request.values.get('jobid')
    source = request.args.get('source')
    pid = request.args.get('id')
    key = request.args.get('key')
    value = request.args.get('value')
    limit = request.args.get('limit')
    noreviews = value_to_bool(request.form.get('noreviews'))
    expand = request.args.get('expand')
    topics = request.args.get('topics')
    try:
        if jobid:
            data = load_paper_data(jobid, source, f'{key}={value}', pubtrends_celery)
            if data is not None:
                logger.info(f'/paper success {log_request(request)}')
                return render_template('paper.html',
                                       **prepare_paper_data(PUBTRENDS_CONFIG, data, source, pid),
                                       max_graph_size=PUBTRENDS_CONFIG.max_graph_size,
                                       version=VERSION)
            else:
                logger.info(f'/paper No job or out-of-date job, restart it {log_request(request)}')
                analyze_search_paper.apply_async(
                    args=[source, pid, key, value, expand, limit, noreviews, topics, pubtrends_app.config['TESTING']],
                    task_id=jobid
                )
                return redirect(url_for('.process', query=trim_query(f'Paper {key}={value}'),
                                        analysis_type=PAPER_ANALYSIS_TYPE,
                                        source=source, key=key, value=value,
                                        noreviews='on' if noreviews else '',
                                        topics=topics, jobid=jobid))
        logger.error(f'/paper error wrong request {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/paper exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@pubtrends_app.route('/papers')
@cache.cached(query_string=True)
def show_ids():
    jobid = request.values.get('jobid')
    query = request.args.get('query')
    source = request.args.get('source')  # Pubmed or Semantic Scholar
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    noreviews = value_to_bool(request.form.get('noreviews'))
    min_year = request.form.get('min_year')  # Minimal year of publications
    max_year = request.form.get('max_year')  # Maximal year of publications
    search_string = ''
    topic = request.args.get('topic')
    try:
        if topic is not None:
            search_string += f'topic: {topic}'
            comp = int(topic) - 1  # Component was exposed so it was 1-based
        else:
            comp = None

        word = request.args.get('word')
        if word is not None:
            search_string += f'word: {word}'

        author = request.args.get('author')
        if author is not None:
            search_string += f'author: {author}'

        journal = request.args.get('journal')
        if journal is not None:
            search_string += f'journal: {journal}'

        papers_list = request.args.get('papers_list')
        if papers_list == 'top':
            search_string += 'Top Papers'
        if papers_list == 'year':
            search_string += 'Papers of the Year'
        if papers_list == 'hot':
            search_string += 'Hot Papers'

        if jobid:
            data = load_result_data(jobid, source, query, sort, limit, noreviews, min_year, max_year, pubtrends_celery)
            if data is not None:
                logger.info(f'/papers success {log_request(request)}')
                export_name = preprocess_string(f'{query}-{search_string}')
                papers_data = prepare_papers_data(
                    PUBTRENDS_CONFIG, data, comp, word, author, journal, papers_list
                )
                return render_template('papers.html',
                                       version=VERSION,
                                       source=source,
                                       query=trim_query(query),
                                       search_string=search_string,
                                       limit=limit,
                                       sort=sort,
                                       noreviews='on' if noreviews else '',
                                       min_year=min_year or '', max_year=max_year or '',
                                       export_name=export_name,
                                       papers=papers_data)
        logger.error(f'/papers error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/papers exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@pubtrends_app.route('/export_data', methods=['GET'])
def export_data():
    logger.info(f'/export_data {log_request(request)}')
    try:
        jobid = request.values.get('jobid')
        query = request.args.get('query')
        source = request.args.get('source')
        limit = request.args.get('limit')
        sort = request.args.get('sort')
        if jobid:
            if request.args.get('paper') == 'on':
                data = load_paper_data(jobid, source, query, pubtrends_celery)
            else:
                noreviews = value_to_bool(request.form.get('noreviews'))
                min_year = request.form.get('min_year')  # Minimal year of publications
                max_year = request.form.get('max_year')  # Maximal year of publications
                data = load_result_data(jobid, source, query, sort, limit, noreviews, min_year, max_year, pubtrends_celery)
            if data:
                with tempfile.TemporaryDirectory() as tmpdir:
                    name = preprocess_string(f'{source}-{query}')
                    path = os.path.join(tmpdir, f'{name}.json.gz')
                    with gzip.open(path, 'w') as f:
                        f.write(json.dumps(data.to_json()).encode('utf-8'))
                    return send_file(path, as_attachment=True)
        logger.error(f'/export_results error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/export_results exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


#########################
# Loading functionality #
#########################

PREDEFINED_JOBS_LOCK = Lock()


def are_predefined_jobs_ready():
    """ Checks if all the precomputed examples are available and gensim fasttext model is loaded """
    if len(PREDEFINED_JOBS) == 0:
        return True
    try:
        PREDEFINED_JOBS_LOCK.acquire()
        if pubtrends_app.config.get('PREDEFINED_TASKS_READY', False):
            return True
        ready = True
        inspect = pubtrends_celery.control.inspect()
        active = inspect.active()
        if active is None:
            return False
        active_jobs = [j['id'] for j in list(active.items())[0][1]]
        reserved = inspect.reserved()
        if reserved is None:
            return False
        scheduled_jobs = [j['id'] for j in list(reserved.items())[0][1]]

        for source, predefine_info in PREDEFINED_JOBS.items():
            for query, jobid in predefine_info:
                logger.info(f'Check predefined search for source={source} query={query} jobid={jobid}')
                # Check celery queue
                if jobid in active_jobs or jobid in scheduled_jobs:
                    ready = False
                    continue
                query, sort, limit = _predefined_example_params_by_jobid(source, jobid, PREDEFINED_JOBS)
                data = load_result_data(jobid, source, query, sort, limit, False, None, None, pubtrends_celery)
                if data is None:
                    logger.info(f'No job or out-of-date job for source={source} query={query}, launch it')
                    analyze_search_terms.apply_async(
                        args=[source, query, sort, int(limit), False, None, None,
                              PUBTRENDS_CONFIG.show_topics_default_value, pubtrends_app.config['TESTING']],
                        task_id=jobid
                    )
                    ready = False
        if ready:
            pubtrends_app.config['PREDEFINED_TASKS_READY'] = True
        return ready
    finally:
        PREDEFINED_JOBS_LOCK.release()


##########################
# Feedback functionality #
##########################

@pubtrends_app.route('/feedback', methods=['POST'])
def feedback():
    logger.info(f'/feedback {log_request(request)}')
    data = request.form
    if 'key' in data:
        key = data.get('key')
        value = data.get('value')
        jobid = data.get('jobid')
        logger.info('Feedback ' + json.dumps(dict(key=key, value=value, jobid=jobid)))
    else:
        logger.error(f'/feedback error')
    return render_template_string('Thanks you for the feedback!'), 200


#######################
# AIHKTN25-28 API     #
#######################

@pubtrends_app.route('/search_terms_api', methods=['POST'])
def search_terms_api():
    logger.info(f'/search_terms_api {log_request(request)}')
    query = request.form.get('query')  # Original search query
    try:
        # Regular search syntax
        if query:
            job = analyze_search_terms.delay('Pubmed', query=query, limit=1000, sort=SORT_MOST_CITED,
                                             noreviews=True, min_year=None, max_year=None,
                                             topics=10,
                                             test=False)
            return {'success': True, 'jobid': job.id}
        logger.error(f'/search_terms_api error {log_request(request)}')
        return {'success': False, 'jobid': None}
    except Exception as e:
        logger.exception(f'/search_terms_api exception {e}')
        return {'success': False, 'jobid': None}, 500

@pubtrends_app.route('/analyse_ids_api', methods=['POST'])
def analyse_ids_api():
    logger.info(f'/analyse_ids_api {log_request(request)}')
    query = request.form.get('query')  # Original search query
    ids = request.form.get('ids').split(',')
    job_id = request.form.get('job_id')
    try:
        if query and job_id and ids:
            analyze_id_list.apply_async(args=['Pubmed', query, ids, 10, False], task_id=job_id)
            return {'success': True, 'jobid': job_id}
        logger.error(f'/analyse_ids_api error {log_request(request)}')
        return {'success': False, 'jobid': None}
    except Exception as e:
        logger.exception(f'/analyse_ids_api exception {e}')
        return {'success': False, 'jobid': None}, 500


@pubtrends_app.route('/check_status_api/<jobid>', methods=['GET'])
def check_status_api(jobid):
    logger.info(f'/check_status_api {log_request(request)}')
    try:
        job = pubtrends_celery.AsyncResult(jobid)
        if job.state == 'PENDING':
            return {'status': 'pending'}, 200
        elif job.state == 'SUCCESS':
            return {'status': 'success'}, 200
        elif job.state == 'FAILURE':
            return {'status': 'failed'}, 200
        return {'status': 'unknown'}, 200
    except Exception as e:
        logger.exception(f'/check_status_api exception {e}')
        return {'status': 'error'}, 500

@pubtrends_app.route('/get_result_api', methods=['GET'])
def get_result_api():
    logger.info(f'/get_result_api {log_request(request)}')
    jobid = request.args.get('jobid')
    query = request.args.get('query')
    try:
        if jobid and query:
            data = load_result_data(jobid, 'Pubmed', query, SORT_MOST_CITED, 1000, True, None, None, pubtrends_celery)
            return data.to_json(), 200
        return {'status': 'error'}, 500
    except Exception as e:
        logger.exception(f'/get_result_api exception {e}')
        return {'status': 'error'}, 500


@pubtrends_app.route('/question', methods=['POST'])
def question():
    logger.info(f'/question {log_request(request)}')
    try:
        data = request.json
        question_text = data['question']
        jobid = data['jobid']

        if not question_text or not jobid:
            logger.error(f'/question error: missing question or jobid {log_request(request)}')
            return {'status': 'error', 'message': 'Missing question or jobid'}, 400

        # Check if text embeddings are available
        if not is_texts_embeddings_available():
            logger.error(f'/question error: text embeddings not available {log_request(request)}')
            return {'status': 'error', 'message': 'Text embeddings not available'}, 400

        # Load result data
        data = load_result_data(jobid, None, None, None, None, None, None, None, pubtrends_celery)
        if data is None:
            logger.error(f'/question error: no data for jobid {jobid} {log_request(request)}')
            return {'status': 'error', 'message': 'No data found for jobid'}, 404

        papers = get_relevant_papers(
            question_text, data, PUBTRENDS_CONFIG.questions_threshold, PUBTRENDS_CONFIG.questions_top_n
        )
        return {'status': 'success', 'papers': papers}, 200
    except Exception as e:
        logger.exception(f'/question exception {e}')
        return {'status': 'error', 'message': str(e)}, 500


#######################
# Admin functionality #
#######################

configure_admin_functions(pubtrends_app, pubtrends_celery, logfile)

#######################
# Additional features #
#######################

# Application
def get_app():
    return pubtrends_app


# With debug=True, the Flask server will auto-reload on changes
if __name__ == '__main__':
    pubtrends_app.run(host='0.0.0.0', debug=True, extra_files=['templates/'], port=5000)
