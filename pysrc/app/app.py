import gzip
import json
import logging
import os
import random
import tempfile
from threading import Lock
from urllib.parse import quote

import requests
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
from pysrc.celery.tasks_main import analyze_search_paper, analyze_search_terms, analyze_pubmed_search
from pysrc.config import PubtrendsConfig
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.plot.plot_app import prepare_graph_data, prepare_papers_data, prepare_paper_data, prepare_result_data
from pysrc.papers.utils import trim_query, IDS_ANALYSIS_TYPE, PAPER_ANALYSIS_TYPE
from pysrc.version import VERSION

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

flask_app = Flask(__name__)

if not flask_app.config['TESTING'] and not flask_app.config['DEBUG']:
    flask_app.config['CACHE_TYPE'] = 'RedisCache'
flask_app.config['CACHE_REDIS_URL'] = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379')
flask_app.config['CACHE_DEFAULT_TIMEOUT'] = 600  # 10 minutes

cache = Cache(flask_app)

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
    flask_app.logger.setLevel(gunicorn_logger.level)

logger = flask_app.logger


#############
# Main page #
#############

@flask_app.route('/robots.txt')
@flask_app.route('/sitemap.xml')
@flask_app.route('/feedback.js')
@flask_app.route('/style.css')
@flask_app.route('/about_graph_explorer.png')
@flask_app.route('/about_hot_papers.png')
@flask_app.route('/about_keywords.png')
@flask_app.route('/about_main.png')
@flask_app.route('/about_network.png')
@flask_app.route('/about_pubtrends_scheme.png')
@flask_app.route('/about_report.png')
@flask_app.route('/about_review.png')
@flask_app.route('/about_top_cited_papers.png')
@flask_app.route('/about_topic.png')
@flask_app.route('/about_topics_by_year.png')
@flask_app.route('/about_topics_hierarchy.png')
@flask_app.route('/smile.svg')
@flask_app.route('/meh.svg')
@flask_app.route('/frown.svg')
def static_from_root():
    return send_from_directory(flask_app.static_folder, request.path[1:])


PREDEFINED_JOBS = get_predefined_jobs(PUBTRENDS_CONFIG)


def log_request(r):
    return f'addr:{r.remote_addr} args:{json.dumps(r.args)}'


@flask_app.route('/')
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
                           search_example_terms=search_example_terms)


@flask_app.route('/about.html', methods=['GET'])
def about():
    return render_template('about.html', version=VERSION)


############################
# Search form POST methods #
############################

def value_to_bool(value):
    return str(value).lower() == 'on'

def bool_to_value(value):
    return 'on' if value else 'off'

@flask_app.route('/search_terms', methods=['POST'])
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
                                              test=flask_app.config['TESTING'])
            return redirect(
                url_for('.process', source='Pubmed', query=trim_query(query), limit=limit, sort='',
                        topics=topics,
                        jobid=job.id))

        # Regular search syntax
        if query and source and sort and limit and topics:
            job = analyze_search_terms.delay(source, query=query, limit=int(limit), sort=sort,
                                             noreviews=noreviews, min_year=min_year, max_year=max_year,
                                             topics=topics,
                                             test=flask_app.config['TESTING'])
            return redirect(
                url_for('.process', query=trim_query(query), source=source, limit=limit, sort=sort,
                        noreviews=noreviews, min_year=min_year, max_year=max_year, topics=topics,
                        jobid=job.id))
        logger.error(f'/search_terms error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/search_terms exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@flask_app.route('/search_paper', methods=['POST'])
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
            min_year = request.form.get('min_year')  # Minimal year of publications
            max_year = request.form.get('max_year')  # Maximal year of publications
            expand = data.get('expand')
            topics = data.get('topics')
            job = analyze_search_paper.delay(source, None, key, value, expand, limit, noreviews, topics,
                                             test=flask_app.config['TESTING'])
            return redirect(url_for('.process', query=trim_query(f'Papers {key}={value}'),
                                    analysis_type=PAPER_ANALYSIS_TYPE,
                                    key=key, value=value,
                                    source=source, expand=expand, limit=limit,
                                    noreviews=noreviews, min_year=min_year, max_year=max_year,
                                    topics=topics,
                                    jobid=job.id))
        logger.error(f'/search_paper error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/search_paper exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


####################################
# Analysis progress and cancelling #
####################################

@flask_app.route('/process')
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


@flask_app.route('/status')
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


@flask_app.route('/cancel')
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


@flask_app.route('/result')
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
                args=[source, query, sort, int(limit), noreviews, min_year, max_year, topics, flask_app.config['TESTING']],
                task_id=jobid
            )
            return redirect(
                url_for('.process', query=trim_query(query), source=source, limit=limit, sort=sort,
                        noreviews=noreviews, min_year=min_year, max_year=max_year,
                        topics=topics,
                        jobid=jobid))
        else:
            logger.error(f'/result error wrong request {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/result exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@flask_app.route('/graph')
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


@flask_app.route('/paper')
@cache.cached(query_string=True)
def paper():
    jobid = request.values.get('jobid')
    source = request.args.get('source')
    pid = request.args.get('id')
    key = request.args.get('key')
    value = request.args.get('value')
    limit = request.args.get('limit')
    noreviews = value_to_bool(request.form.get('noreviews'))
    min_year = request.form.get('min_year')  # Minimal year of publications
    max_year = request.form.get('max_year')  # Maximal year of publications
    expand = request.args.get('expand')
    topics = request.args.get('topics')
    try:
        if jobid:
            data = load_paper_data(jobid, source, f'{key}={value}', flask_app)
            if data is not None:
                logger.info(f'/paper success {log_request(request)}')
                return render_template('paper.html',
                                       **prepare_paper_data(PUBTRENDS_CONFIG, data, source, pid),
                                       max_graph_size=PUBTRENDS_CONFIG.max_graph_size,
                                       version=VERSION)
            else:
                logger.info(f'/paper No job or out-of-date job, restart it {log_request(request)}')
                analyze_search_paper.apply_async(
                    args=[
                        source, pid, key, value, expand, limit, noreviews, min_year, max_year, topics,
                        flask_app.config['TESTING']
                    ], task_id=jobid
                )
                return redirect(url_for('.process', query=trim_query(f'Paper {key}={value}'),
                                        analysis_type=PAPER_ANALYSIS_TYPE,
                                        source=source, key=key, value=value, topics=topics, jobid=jobid))
        logger.error(f'/paper error wrong request {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/paper exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@flask_app.route('/papers')
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
                                       export_name=export_name,
                                       papers=papers_data)
        logger.error(f'/papers error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/papers exception {e}')
        return render_template_string(f'<strong>{ERROR_OCCURRED}</strong><br>{e}'), 500


@flask_app.route('/export_data', methods=['GET'])
@cache.cached(query_string=True)
def export_results():
    logger.info(f'/export_data {log_request(request)}')
    try:
        jobid = request.values.get('jobid')
        query = request.args.get('query')
        source = request.args.get('source')
        limit = request.args.get('limit')
        sort = request.args.get('sort')
        noreviews = value_to_bool(request.form.get('noreviews'))
        min_year = request.form.get('min_year')  # Minimal year of publications
        max_year = request.form.get('max_year')  # Maximal year of publications
        if jobid:
            data = load_result_data(jobid, source, query, sort, limit, noreviews, min_year, max_year, pubtrends_celery)
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
        if flask_app.config.get('PREDEFINED_TASKS_READY', False):
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
                              PUBTRENDS_CONFIG.show_topics_default_value, flask_app.config['TESTING']],
                        task_id=jobid
                    )
                    ready = False
        if ready:
            flask_app.config['PREDEFINED_TASKS_READY'] = True
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

@flask_app.route('/feedback', methods=['POST'])
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

configure_admin_functions(flask_app, pubtrends_celery, logfile)

#######################
# Additional features #
#######################

# Application
def get_app():
    return flask_app


# With debug=True, Flask server will auto-reload on changes
if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', debug=True, extra_files=['templates/'], port=5000)
