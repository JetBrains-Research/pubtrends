import gzip
import json
import logging
import os
import random
import requests
import tempfile
import time
from celery.result import AsyncResult
from flask import Flask, url_for, redirect, render_template, request, render_template_string, \
    send_from_directory, send_file
from flask_caching import Cache
from threading import Lock
from urllib.parse import quote

from pysrc.app.admin.admin import configure_admin_functions
from pysrc.app.messages import SOMETHING_WENT_WRONG_SEARCH, ERROR_OCCURRED, \
    SERVICE_LOADING_PREDEFINED_EXAMPLES, SERVICE_LOADING_INITIALIZING
from pysrc.app.reports import get_predefined_jobs, \
    load_result_data, _predefined_example_params_by_jobid, preprocess_string, load_paper_data
from pysrc.celery.pubtrends_celery import pubtrends_celery
from pysrc.celery.tasks_main import analyze_search_paper, analyze_search_terms, analyze_pubmed_search
from pysrc.config import PubtrendsConfig
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.plot.plot_app import prepare_graph_data, prepare_papers_data, prepare_paper_data, prepare_result_data
from pysrc.papers.utils import trim, IDS_ANALYSIS_TYPE, PAPER_ANALYSIS_TYPE, MAX_QUERY_LENGTH
from pysrc.version import VERSION

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

if PUBTRENDS_CONFIG.feature_review_enabled:
    from pysrc.review.app.app import REVIEW_ANALYSIS_TYPE, register_app_review
else:
    REVIEW_ANALYSIS_TYPE = 'not_available'

app = Flask(__name__)

if not app.config['TESTING']:
    app.config['CACHE_TYPE'] = 'RedisCache'
app.config['CELERY_REDIS_URL'] = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379')
app.config['CACHE_DEFAULT_TIMEOUT'] = 600  # 10 minutes

cache = Cache(app)

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
    app.logger.setLevel(gunicorn_logger.level)

logger = app.logger


#############
# Main page #
#############

@app.route('/robots.txt')
@app.route('/sitemap.xml')
@app.route('/feedback.js')
@app.route('/style.css')
@app.route('/about_graph_explorer.png')
@app.route('/about_hot_papers.png')
@app.route('/about_keywords.png')
@app.route('/about_main.png')
@app.route('/about_network.png')
@app.route('/about_pubtrends_scheme.png')
@app.route('/about_report.png')
@app.route('/about_review.png')
@app.route('/about_top_cited_papers.png')
@app.route('/about_topic.png')
@app.route('/about_topics_by_year.png')
@app.route('/about_topics_hierarchy.png')
@app.route('/smile.svg')
@app.route('/meh.svg')
@app.route('/frown.svg')
def static_from_root():
    return send_from_directory(app.static_folder, request.path[1:])


PREDEFINED_JOBS = get_predefined_jobs(PUBTRENDS_CONFIG)


def log_request(r):
    return f'addr:{r.remote_addr} args:{json.dumps(r.args)}'


@app.route('/')
def index():
    if not is_fasttext_endpoint_ready():
        return render_template('init.html', version=VERSION, message=SERVICE_LOADING_INITIALIZING)
    if not are_predefined_jobs_ready():
        return render_template('init.html', version=VERSION, message=SERVICE_LOADING_PREDEFINED_EXAMPLES)

    search_example_message = ''
    search_example_source = ''
    search_example_terms = []

    if len(PREDEFINED_JOBS) != 0:
        search_example_source, search_example_terms = random.choice(list(PREDEFINED_JOBS.items()))
        search_example_message = 'Try one of our examples for ' + search_example_source

    if PUBTRENDS_CONFIG.min_search_words > 1:
        min_words_message = f'Minimum {PUBTRENDS_CONFIG.min_search_words} words per query. '
    else:
        min_words_message = ''

    return render_template('main.html',
                           version=VERSION,
                           limits=PUBTRENDS_CONFIG.show_max_articles_options,
                           default_limit=PUBTRENDS_CONFIG.show_max_articles_default_value,
                           topics_variants=PUBTRENDS_CONFIG.show_topics_options,
                           default_topics=PUBTRENDS_CONFIG.show_topics_default_value,
                           min_words_message=min_words_message,
                           max_papers=PUBTRENDS_CONFIG.max_number_of_articles,
                           pm_enabled=PUBTRENDS_CONFIG.pm_enabled,
                           save_to_files_enabled=PUBTRENDS_CONFIG.save_to_files_enabled,
                           ss_enabled=PUBTRENDS_CONFIG.ss_enabled,
                           search_example_message=search_example_message,
                           search_example_source=search_example_source,
                           search_example_terms=search_example_terms)


@app.route('/about.html', methods=['GET'])
def about():
    return render_template('about.html', version=VERSION)


############################
# Search form POST methods #
############################

@app.route('/search_terms', methods=['POST'])
def search_terms():
    logger.info(f'/search_terms {log_request(request)}')
    query = request.form.get('query')  # Original search query
    source = request.form.get('source')  # Pubmed or Semantic Scholar
    sort = request.form.get('sort')  # Sort order
    limit = request.form.get('limit')  # Limit
    pubmed_syntax = request.form.get('pubmed-syntax') == 'on'
    noreviews = request.form.get('noreviews') == 'on'  # Include reviews in the initial search phase
    topics = request.form.get('topics')  # Topics sizes

    try:
        # PubMed syntax
        if query and source and limit and topics and pubmed_syntax:
            # Regular analysis
            job = analyze_pubmed_search.delay(query=query, sort=sort, limit=limit, topics=topics,
                                              test=app.config['TESTING'])
            return redirect(
                url_for('.process', source='Pubmed', query=trim(query, MAX_QUERY_LENGTH), limit=limit, sort='',
                        topics=topics,
                        jobid=job.id))

        # Regular search syntax
        if query and source and sort and limit and topics:
            job = analyze_search_terms.delay(source, query=query, limit=int(limit), sort=sort,
                                             noreviews=noreviews, topics=topics,
                                             test=app.config['TESTING'])
            return redirect(
                url_for('.process', query=trim(query, MAX_QUERY_LENGTH), source=source, limit=limit, sort=sort,
                        noreviews=noreviews, topics=topics,
                        jobid=job.id))
        logger.error(f'/search_terms error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/search_terms exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@app.route('/search_paper', methods=['POST'])
def search_paper():
    logger.info(f'/search_paper {log_request(request)}')
    data = request.form
    try:
        if 'source' in data and 'key' in data and 'value' in data and 'topics' in data:
            source = data.get('source')  # Pubmed or Semantic Scholar
            key = data.get('key')
            value = data.get('value')
            limit = data.get('limit')
            topics = data.get('topics')
            job = analyze_search_paper.delay(source, None, key, value, limit, topics, test=app.config['TESTING'])
            return redirect(url_for('.process', query=trim(f'Paper {key}={value}', MAX_QUERY_LENGTH),
                                    analysis_type=PAPER_ANALYSIS_TYPE,
                                    key=key, value=value, source=source, limit=limit, topics=topics, jobid=job.id))
        logger.error(f'/search_paper error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/search_paper exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


####################################
# Analysis progress and cancelling #
####################################

@app.route('/process')
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
        topics = request.values.get('topics')

        if analysis_type == IDS_ANALYSIS_TYPE:
            logger.info(f'/process ids {log_request(request)}')
            return render_template('process.html',
                                   redirect_page='result',  # redirect in case of success
                                   redirect_args=dict(query=quote(query), source=source, jobid=jobid,
                                                      limit=analysis_type, sort='', topics=topics),
                                   query=trim(query, MAX_QUERY_LENGTH), source=source, limit=analysis_type, sort='',
                                   jobid=jobid, version=VERSION)

        elif analysis_type == PAPER_ANALYSIS_TYPE:
            logger.info(f'/process paper analysis {log_request(request)}')
            key = request.args.get('key')
            value = request.args.get('value')
            limit = request.args.get('limit') or PUBTRENDS_CONFIG.show_max_articles_default_value
            return render_template('process.html',
                                   redirect_page='paper',  # redirect in case of success
                                   redirect_args=dict(source=source, jobid=jobid, sort='',
                                                      key=key, value=value, limit=limit, topics=topics),
                                   query=trim(query, MAX_QUERY_LENGTH), source=source,
                                   jobid=jobid, version=VERSION)

        elif analysis_type == REVIEW_ANALYSIS_TYPE:
            logger.info(f'/process review {log_request(request)}')
            limit = request.args.get('limit') or PUBTRENDS_CONFIG.show_max_articles_default_value
            sort = request.args.get('sort')
            return render_template('process.html',
                                   redirect_page='review',  # redirect in case of success
                                   redirect_args=dict(query=quote(query), source=source, limit=limit, sort=sort,
                                                      jobid=jobid),
                                   query=trim(query, MAX_QUERY_LENGTH), source=source,
                                   limit=limit, sort=sort,
                                   jobid=jobid, version=VERSION)

        elif query:  # This option should be the last default
            logger.info(f'/process regular search {log_request(request)}')
            limit = request.args.get('limit') or PUBTRENDS_CONFIG.show_max_articles_default_value
            sort = request.args.get('sort')
            return render_template('process.html',
                                   redirect_page='result',  # redirect in case of success
                                   redirect_args=dict(query=quote(query), source=source, limit=limit, sort=sort,
                                                      topics=topics, jobid=jobid),
                                   query=trim(query, MAX_QUERY_LENGTH), source=source,
                                   limit=limit, sort=sort,
                                   jobid=jobid, version=VERSION)

    logger.error(f'/process error {log_request(request)}')
    return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400


@app.route('/status')
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


@app.route('/cancel')
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


@app.route('/result')
@cache.cached(query_string=True)
@cache.memoize(timeout=600)  # 10min
def result():
    jobid = request.args.get('jobid')
    query = request.args.get('query')
    source = request.args.get('source')
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    noreviews = request.args.get('noreviews') == 'on'  # Include reviews in the initial search phase
    topics = request.args.get('topics')
    try:
        if jobid:
            data = load_result_data(jobid, source, query, sort, limit, pubtrends_celery)
            if data is not None:
                logger.info(f'/result success {log_request(request)}')
                return render_template('result.html',
                                       query=trim(query, MAX_QUERY_LENGTH),
                                       source=source,
                                       limit=limit,
                                       sort=sort,
                                       max_graph_size=PUBTRENDS_CONFIG.max_graph_size,
                                       version=VERSION,
                                       is_predefined=True,
                                       **prepare_result_data(PUBTRENDS_CONFIG, data))
            logger.info(f'/result No job or out-of-date job, restart it {log_request(request)}')
            analyze_search_terms.apply_async(
                args=[source, query, sort, int(limit), noreviews, topics, app.config['TESTING']],
                task_id=jobid
            )
            return redirect(
                url_for('.process', query=trim(query, MAX_QUERY_LENGTH), source=source, limit=limit, sort=sort,
                        noreviews=noreviews, topics=topics,
                        jobid=jobid))
        else:
            logger.error(f'/result error wrong request {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/result exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@app.route('/graph')
@cache.cached(query_string=True)
@cache.memoize(timeout=600)  # 10min
def graph():
    jobid = request.values.get('jobid')
    query = request.args.get('query')
    source = request.args.get('source')
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    pid = request.args.get('id')
    try:
        if jobid:
            data = load_result_data(jobid, source, query, sort, limit, pubtrends_celery)
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


@app.route('/paper')
@cache.cached(query_string=True)
@cache.memoize(timeout=600)  # 10min
def paper():
    jobid = request.values.get('jobid')
    source = request.args.get('source')
    pid = request.args.get('id')
    key = request.args.get('key')
    value = request.args.get('value')
    limit = request.args.get('limit')
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
                    args=[source, pid, key, value, limit, topics, app.config['TESTING']], task_id=jobid
                )
                return redirect(url_for('.process', query=trim(f'Paper {key}={value}', MAX_QUERY_LENGTH),
                                        analysis_type=PAPER_ANALYSIS_TYPE,
                                        source=source, key=key, value=value, topics=topics, jobid=jobid))
        logger.error(f'/paper error wrong request {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/paper exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@app.route('/papers')
@cache.cached(query_string=True)
@cache.memoize(timeout=600)  # 10min
def show_ids():
    jobid = request.values.get('jobid')
    query = request.args.get('query')
    source = request.args.get('source')  # Pubmed or Semantic Scholar
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    search_string = ''
    comp = request.args.get('comp')
    try:
        if comp is not None:
            search_string += f'topic: {comp}'
            comp = int(comp) - 1  # Component was exposed so it was 1-based

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
            data = load_result_data(jobid, source, query, sort, limit, pubtrends_celery)
            if data is not None:
                logger.info(f'/papers success {log_request(request)}')
                export_name = preprocess_string(f'{query}-{search_string}')
                papers_data = prepare_papers_data(
                    PUBTRENDS_CONFIG, data, comp, word, author, journal, papers_list
                )
                return render_template('papers.html',
                                       version=VERSION,
                                       source=source,
                                       query=trim(query, MAX_QUERY_LENGTH),
                                       search_string=search_string,
                                       limit=limit,
                                       sort=sort,
                                       export_name=export_name,
                                       papers=papers_data)
        logger.error(f'/papers error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/papers exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@app.route('/export_data', methods=['GET'])
@cache.cached(query_string=True)
@cache.memoize(timeout=600)  # 10min
def export_results():
    logger.info(f'/export_data {log_request(request)}')
    try:
        jobid = request.values.get('jobid')
        query = request.args.get('query')
        source = request.args.get('source')
        limit = request.args.get('limit')
        sort = request.args.get('sort')
        if jobid:
            data = load_result_data(jobid, source, query, sort, limit, pubtrends_celery)
            with tempfile.TemporaryDirectory() as tmpdir:
                name = preprocess_string(f'{source}-{query}-{sort}-{limit}')
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
        if app.config.get('PREDEFINED_TASKS_READY', False):
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
                data = load_result_data(jobid, source, query, sort, limit, pubtrends_celery)
                if data is None:
                    logger.info(f'No job or out-of-date job for source={source} query={query}, launch it')
                    analyze_search_terms.apply_async(
                        args=[source, query, sort, int(limit), False, PUBTRENDS_CONFIG.show_topics_default_value,
                              app.config['TESTING']],
                        task_id=jobid
                    )
                    ready = False
        if ready:
            app.config['PREDEFINED_TASKS_READY'] = True
        return ready
    finally:
        PREDEFINED_JOBS_LOCK.release()


# Launch with Docker address or locally
FASTTEXT_URL = os.getenv('FASTTEXT_URL', 'http://localhost:5001')


def is_fasttext_endpoint_ready():
    logger.debug(f'Check fasttext endpoint is ready')
    try:
        r = requests.request(url=FASTTEXT_URL, method='GET')
        if r.status_code != 200:
            return False
        r = requests.request(url=f'{FASTTEXT_URL}/initialized', method='GET', headers={'Accept': 'application/json'})
        if r.status_code != 200 or r.json() is not True:
            return False
        return True
    except Exception as e:
        logger.debug(f'Fasttext endpoint is not ready: {e}')
        return False


##########################
# Feedback functionality #
##########################

@app.route('/feedback', methods=['POST'])
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
# Admin functionality #
#######################

configure_admin_functions(app, pubtrends_celery, logfile)

#######################
# Additional features #
#######################

if PUBTRENDS_CONFIG.feature_review_enabled:
    register_app_review(app)


# Application
def get_app():
    return app


# With debug=True, Flask server will auto-reload on changes
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, extra_files=['templates/'], port=5000)
