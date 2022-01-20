import gzip
import json
import logging
import os
import random
import re
import tempfile
import time
from urllib.parse import quote
from threading import Lock


import requests
from celery.result import AsyncResult
from flask import Flask, url_for, redirect, render_template, request, render_template_string, \
    send_from_directory, send_file

from pysrc.app.admin.admin import configure_admin_functions
from pysrc.app.messages import SOMETHING_WENT_WRONG_SEARCH, ERROR_OCCURRED, SOMETHING_WENT_WRONG_PAPER, \
    SOMETHING_WENT_WRONG_TOPIC, SERVICE_LOADING_NLP_MODELS, SERVICE_LOADING_PREDEFINED_EXAMPLES
from pysrc.app.predefined import get_predefined_jobs, load_predefined_viz_log, \
    load_predefined_or_result_data, _example_by_jobid
from pysrc.celery.pubtrends_celery import pubtrends_celery
from pysrc.celery.tasks_cache import get_or_cancel_task
from pysrc.celery.tasks_main import analyze_search_paper, analyze_search_terms, analyze_id_list, \
    analyze_pubmed_search_files, analyze_pubmed_search, analyze_search_terms_files
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.analyzer_files import FILES_WITH_DESCRIPTIONS, ANALYSIS_FILES_TYPE
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.plot.plot_preprocessor import PlotPreprocessor
from pysrc.papers.plot.plotter import TOPIC_KEYWORDS
from pysrc.papers.plot.plotter_paper import prepare_paper_data
from pysrc.papers.utils import trim, IDS_ANALYSIS_TYPE, PAPER_ANALYSIS_TYPE, human_readable_size, topics_palette
from pysrc.version import VERSION

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

if PUBTRENDS_CONFIG.feature_review_enabled:
    from pysrc.review.app.app import REVIEW_ANALYSIS_TYPE, register_app_review
else:
    REVIEW_ANALYSIS_TYPE = 'not_available'

app = Flask(__name__)

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

MAX_QUERY_LENGTH = 60


def log_request(r):
    return f'addr:{r.remote_addr} args:{json.dumps(r.args)}'


@app.route('/')
def index():
    if not is_fasttext_endpoint_ready():
        return render_template('init.html', version=VERSION, message=SERVICE_LOADING_NLP_MODELS)
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
    expand = request.form.get('expand')  # Fraction of papers to cover by references
    topics = request.form.get('topics')  # Topics sizes
    files = request.form.get('files') == 'on'

    try:
        # PubMed syntax
        if query and source and limit and topics and pubmed_syntax:
            if files:
                # Save results to files
                job = analyze_pubmed_search_files.delay(query=query, sort=sort, limit=limit, topics=topics,
                                                        test=app.config['TESTING'])
                return redirect(url_for('.process', query=query, analysis_type=ANALYSIS_FILES_TYPE,
                                        sort='', limit=limit, source='Pubmed', topics=topics, jobid=job.id))
            else:
                # Regular analysis
                job = analyze_pubmed_search.delay(query=query, sort=sort, limit=limit, topics=topics,
                                                  test=app.config['TESTING'])
                return redirect(url_for('.process', source='Pubmed', query=query, limit=limit, sort='', topics=topics,
                                        jobid=job.id))

        # Regular search syntax
        if query and source and sort and limit and expand and topics:
            if files:
                # Save results to files
                job = analyze_search_terms_files.delay(source, query=query, limit=int(limit), sort=sort,
                                                       noreviews=noreviews, expand=int(expand) / 100, topics=topics,
                                                       test=app.config['TESTING'])
                return redirect(url_for('.process', query=query, analysis_type=ANALYSIS_FILES_TYPE,
                                        sort=sort, limit=limit, source=source, topics=topics, jobid=job.id))
            else:
                # Regular analysis
                job = analyze_search_terms.delay(source, query=query, limit=int(limit), sort=sort,
                                                 noreviews=noreviews, expand=int(expand) / 100, topics=topics,
                                                 test=app.config['TESTING'])
                return redirect(url_for('.process', query=query, source=source, limit=limit, sort=sort,
                                        noreviews=noreviews, expand=expand, topics=topics,
                                        jobid=job.id))
        logger.error(f'/search_terms error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_TOPIC), 400
    except Exception as e:
        logger.exception(f'/search_terms exception {e}')
        return render_template_string(ERROR_OCCURRED), 500


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
            return redirect(url_for('.process', query=f'Paper {key}={value}', analysis_type=PAPER_ANALYSIS_TYPE,
                                    key=key, value=value, source=source, limit=limit, topics=topics, jobid=job.id))
        logger.error(f'/search_paper error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_PAPER), 400
    except Exception as e:
        logger.exception(f'/search_paper exception {e}')
        return render_template_string(ERROR_OCCURRED), 500


@app.route('/process_ids', methods=['POST'])
def process_ids():
    logger.info(f'/process_ids {log_request(request)}')
    source = request.form.get('source')  # Pubmed or Semantic Scholar
    query = request.form.get('query')  # Original search query
    topics = request.form.get('topics')
    try:
        if source and query and topics and 'id_list' in request.form:
            id_list = request.form.get('id_list').split(',')
            analysis_type = request.form.get('analysis_type')
            job = analyze_id_list.delay(
                source, ids=id_list, query=query, analysis_type=analysis_type, limit=None, topics=topics,
                test=app.config['TESTING']
            )
            return redirect(url_for('.process', query=query, analysis_type=analysis_type, source=source, topics=topics,
                                    jobid=job.id))
        logger.error(f'/process_ids error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/process_ids exception {e}')
        return render_template_string(ERROR_OCCURRED), 500


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
        topics = request.values.get('topics') or 'medium'

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
            limit = request.args.get('limit')
            return render_template('process.html',
                                   redirect_page='paper',  # redirect in case of success
                                   redirect_args=dict(source=source, jobid=jobid, sort='',
                                                      key=key, value=value, limit=limit, topics=topics),
                                   query=trim(query, MAX_QUERY_LENGTH), source=source,
                                   jobid=jobid, version=VERSION)

        elif analysis_type == REVIEW_ANALYSIS_TYPE:
            logger.info(f'/process review {log_request(request)}')
            limit = request.args.get('limit')
            sort = request.args.get('sort')
            return render_template('process.html',
                                   redirect_page='review',  # redirect in case of success
                                   redirect_args=dict(query=quote(query), source=source, limit=limit, sort=sort,
                                                      jobid=jobid),
                                   query=trim(query, MAX_QUERY_LENGTH), source=source,
                                   limit=limit, sort=sort,
                                   jobid=jobid, version=VERSION)

        elif analysis_type == ANALYSIS_FILES_TYPE:
            logger.info(f'/process files {log_request(request)}')
            limit = request.args.get('limit')
            return render_template('process.html',
                                   redirect_page='result_files',  # redirect in case of success
                                   redirect_args=dict(source=source, query=quote(query), limit=limit, sort='',
                                                      topics=topics, jobid=jobid),
                                   query=trim(query, MAX_QUERY_LENGTH), source=source,
                                   limit=limit, sort='',
                                   jobid=jobid, version=VERSION)

        elif query:  # This option should be the last default
            logger.info(f'/process regular search {log_request(request)}')
            limit = request.args.get('limit')
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
        job = get_or_cancel_task(jobid)
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
def result():
    jobid = request.args.get('jobid')
    query = request.args.get('query')
    source = request.args.get('source')
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    noreviews = request.args.get('noreviews') == 'on'  # Include reviews in the initial search phase
    expand = request.args.get('expand') or 0  # Fraction of papers to cover by references
    topics = request.args.get('topics')
    try:
        if jobid and query and source and limit is not None and sort is not None:
            viz_log = load_predefined_viz_log(source, jobid, PREDEFINED_JOBS, pubtrends_celery)
            if viz_log is not None:
                logger.info(f'/result success {log_request(request)}')
                viz, log = viz_log
                return render_template('result.html',
                                       query=trim(query, MAX_QUERY_LENGTH),
                                       source=source,
                                       limit=limit,
                                       sort=sort,
                                       max_graph_size=PUBTRENDS_CONFIG.max_graph_size,
                                       version=VERSION,
                                       log=log,
                                       **viz)
            logger.info(f'/result No job or out-of-date job, restart it {log_request(request)}')
            analyze_search_terms.apply_async(
                args=[source, query, sort, int(limit), noreviews, int(expand) / 100, topics, app.config['TESTING']],
                task_id=jobid
            )
            return redirect(url_for('.process', query=query, source=source, limit=limit, sort=sort,
                                    noreviews=noreviews, expand=expand, topics=topics,
                                    jobid=jobid))
        else:
            logger.error(f'/result error wrong request {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/result exception {e}')
        return render_template_string(ERROR_OCCURRED), 500


@app.route('/graph')
def graph():
    jobid = request.values.get('jobid')
    query = request.args.get('query')
    source = request.args.get('source')
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    if jobid:
        data = load_predefined_or_result_data(source, jobid, PREDEFINED_JOBS, pubtrends_celery)
        if data is not None:
            loader, url_prefix = Loaders.get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
            analyzer = PapersAnalyzer(loader, PUBTRENDS_CONFIG)
            analyzer.init(data)
            topics_tags = {comp: ','.join(
                [w[0] for w in analyzer.kwd_df[analyzer.kwd_df['comp'] == comp]['kwd'].values[0][:TOPIC_KEYWORDS]]
            ) for comp in sorted(set(analyzer.df['comp']))}
            logger.debug('Computing sparse graph')
            graph_cs = PlotPreprocessor.dump_similarity_graph_cytoscape(
                analyzer.df, analyzer.sparse_papers_graph
            )
            logger.info(f'/graph success similarity {log_request(request)}')
            return render_template(
                'graph.html',
                version=VERSION,
                source=source,
                query=trim(query, MAX_QUERY_LENGTH),
                limit=limit,
                sort=sort,
                topics_palette_json=json.dumps(topics_palette(analyzer.df)),
                topics_description_json=json.dumps(topics_tags),
                graph_cytoscape_json=json.dumps(graph_cs)
            )
        logger.error(f'/graph error job id {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    else:
        logger.error(f'/graph error wrong request {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400


@app.route('/paper')
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
            data = load_predefined_or_result_data(source, jobid, PREDEFINED_JOBS, pubtrends_celery)
            if data is not None:
                logger.info(f'/paper success {log_request(request)}')
                return render_template('paper.html',
                                       **prepare_paper_data(data, source, pid),
                                       max_graph_size=PUBTRENDS_CONFIG.max_graph_size,
                                       version=VERSION)
            else:
                logger.info(f'/paper No job or out-of-date job, restart it {log_request(request)}')
                analyze_search_paper.apply_async(
                    args=[source, pid, key, value, limit, topics, app.config['TESTING']], task_id=jobid
                )
                return redirect(url_for('.process', query=f'Paper {key}={value}', analysis_type=PAPER_ANALYSIS_TYPE,
                                        source=source, key=key, value=value, topics=topics, jobid=jobid))
        else:
            logger.error(f'/paper error wrong request {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG_PAPER), 400
    except Exception as e:
        logger.exception(f'/paper exception {e}')
        return render_template_string(ERROR_OCCURRED), 500


@app.route('/papers')
def show_ids():
    jobid = request.values.get('jobid')
    query = request.args.get('query')
    source = request.args.get('source')  # Pubmed or Semantic Scholar
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    search_string = ''
    comp = request.args.get('comp')
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
        data = load_predefined_or_result_data(source, jobid, PREDEFINED_JOBS, pubtrends_celery)
        if data is not None:
            logger.info(f'/papers success {log_request(request)}')
            export_name = re.sub('_{2,}', '_', re.sub('["\':,. ]', '_', f'{query}_{search_string}'.lower())).strip('_')
            loader, url_prefix = Loaders.get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
            analyzer = PapersAnalyzer(loader, PUBTRENDS_CONFIG)
            analyzer.init(data)
            papers_data = PlotPreprocessor.prepare_papers_data(
                analyzer.df, analyzer.top_cited_papers, analyzer.max_gain_papers, analyzer.max_rel_gain_papers,
                url_prefix,
                comp, word, author, journal, papers_list
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


@app.route('/result_files', methods=['GET'])
def result_files():
    logger.info(f'/result_files {log_request(request)}')
    source = request.args.get('source')
    jobid = request.args.get('jobid')
    query = request.args.get('query')
    limit = request.args.get('limit')
    try:
        if jobid and query and limit:
            job = AsyncResult(jobid, app=pubtrends_celery)
            if job and job.state == 'SUCCESS':
                query_folder = job.result
                if 'file' in request.args:
                    file = request.args.get('file')
                    return send_file(os.path.join(query_folder, file), as_attachment=True, attachment_filename=file)
                else:
                    available_files = list(sorted(os.listdir(query_folder)))
                    file_infos = []
                    qq = quote(query)
                    # IMPORTANT: only files from description are showed
                    for f, d in FILES_WITH_DESCRIPTIONS.items():
                        if f in available_files:
                            full_path = os.path.join(query_folder, f)
                            if os.path.exists(full_path):
                                url = f'/result_files?file={f}&jobid={jobid}&query={qq}&source={source}&limit={limit}'
                                file_infos.append((f, url, d,
                                                   time.ctime(os.path.getmtime(full_path)),
                                                   human_readable_size(os.path.getsize(full_path))))

                    return render_template('result_files.html',
                                           query=trim(query, MAX_QUERY_LENGTH),
                                           full_query=query,
                                           source=source,
                                           limit=limit,
                                           file_infos=file_infos,
                                           version=VERSION)
        logger.error(f'/result_files error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/result_files exception {e}')
        return render_template_string(ERROR_OCCURRED), 500


@app.route('/export_data', methods=['GET'])
def export_results():
    logger.info(f'/export_data {log_request(request)}')
    try:
        jobid = request.values.get('jobid')
        query = request.args.get('query')
        source = request.args.get('source')
        limit = request.args.get('limit')
        sort = request.args.get('sort')
        if jobid and query and source and limit and source:
            data = load_predefined_or_result_data(source, jobid, PREDEFINED_JOBS, pubtrends_celery)
            with tempfile.TemporaryDirectory() as tmpdir:
                name = re.sub('_{2,}', '_',
                              re.sub('["\':,. ]', '_', f'{source}_{query}_{sort}_{limit}'.lower())).strip('_')
                path = os.path.join(tmpdir, f'{name}.json.gz')
                with gzip.open(path, 'w') as f:
                    f.write(json.dumps(data).encode('utf-8'))
                return send_file(path, as_attachment=True)
        logger.error(f'/export_results error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.exception(f'/export_results exception {e}')
        return render_template_string(ERROR_OCCURRED), 500


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
        active_jobs = [j['id'] for j in list(inspect.active().items())[0][1]]
        scheduled_jobs = [j['id'] for j in list(inspect.reserved().items())[0][1]]

        for source, predefine_info in PREDEFINED_JOBS.items():
            for query, query_hash in predefine_info:
                logger.info(f'Check predefined search for source={source} query={query}')
                jobid = f'predefined_{query_hash}'
                # Check celery queue
                if jobid in active_jobs or jobid in scheduled_jobs:
                    ready = False
                    continue
                if load_predefined_or_result_data(source, jobid, PREDEFINED_JOBS, pubtrends_celery) is None:
                    query, sort, limit = _example_by_jobid(source, jobid, PREDEFINED_JOBS)
                    logger.info(f'No job or out-of-date job for source={source} query={query}, launch it')
                    expand = 20 if source == 'Pubmed' else 0
                    analyze_search_terms.apply_async(
                        args=[source, query, sort, int(limit), False, expand / 100, 'medium', app.config['TESTING']],
                        task_id=jobid
                    )
                    ready = False
        if ready:
            app.config['PREDEFINED_TASKS_READY'] = True
        return ready
    finally:
        PREDEFINED_JOBS_LOCK.release()


# Launch with Docker address or locally
FASTTEXT_URL = os.getenv('FASTTEXT_URL', 'http://localhost:8081')


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
    except:
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
    elif 'type' in data:
        feedback_type = data.get('type')
        message = data.get('message')
        email = data.get('email')
        jobid = data.get('jobid')
        logger.info('Feedback ' + json.dumps(dict(type=feedback_type, message=message, email=email, jobid=jobid)))
    else:
        logger.error(f'/feedback error')
        return render_template_string(ERROR_OCCURRED), 500
    return render_template_string('Thanks you for the feedback!'), 200


#######################
# Admin functionality #
#######################

configure_admin_functions(app, pubtrends_celery, logfile)

#######################
# Additional features #
#######################

if PUBTRENDS_CONFIG.feature_review_enabled:
    register_app_review(app, PREDEFINED_JOBS)


# Application
def get_app():
    return app


# With debug=True, Flask server will auto-reload on changes
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, extra_files=['templates/'])
