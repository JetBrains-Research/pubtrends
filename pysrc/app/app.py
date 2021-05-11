import gzip
import hashlib
import json
import logging
import os
import random
import re
import tempfile
from celery.result import AsyncResult
from flask import Flask, url_for, redirect, render_template, request, render_template_string, \
    send_from_directory, send_file
from urllib.parse import quote

from pysrc.app.admin.admin import configure_admin_functions
from pysrc.app.predefined import save_predefined, load_predefined_viz_log, load_predefined_or_result_data
from pysrc.app.utils import log_request, MAX_QUERY_LENGTH, SOMETHING_WENT_WRONG_SEARCH, ERROR_OCCURRED, \
    SOMETHING_WENT_WRONG_PAPER, SOMETHING_WENT_WRONG_TOPIC
from pysrc.celery.pubtrends_celery import pubtrends_celery
from pysrc.celery.tasks_cache import get_or_cancel_task
from pysrc.celery.tasks_main import find_paper_async, analyze_search_terms, analyze_id_list
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.plot.plotter_paper import prepare_paper_data
from pysrc.papers.plot.plot_preprocessor import PlotPreprocessor
from pysrc.papers.plot.plotter import Plotter
from pysrc.papers.utils import zoom_name, trim, PAPER_ANALYSIS, ZOOM_IN_TITLE, PAPER_ANALYSIS_TITLE, ZOOM_OUT_TITLE
from pysrc.version import VERSION

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

if PUBTRENDS_CONFIG.feature_review_enabled:
    from pysrc.review.app.review import REVIEW_ANALYSIS_TITLE, register_app_review
else:
    REVIEW_ANALYSIS_TITLE = 'not_available'

app = Flask(__name__)

#####################
# Configure logging #
#####################

# Deployment and development
LOG_PATHS = ['/logs', os.path.expanduser('~/.pubtrends/logs')]
for p in LOG_PATHS:
    if os.path.isdir(p):
        logfile = os.path.join(p, 'pubtrends.log')
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


@app.route('/robots.txt')
@app.route('/sitemap.xml')
@app.route('/feedback.js')
@app.route('/style.css')
@app.route('/about_humanaging_graph.png')
@app.route('/about_humanaging_report.png')
@app.route('/about_humanaging_topic_other.png')
@app.route('/about_pubtrends_scheme.png')
@app.route('/smile.svg')
@app.route('/meh.svg')
@app.route('/frown.svg')
def static_from_root():
    return send_from_directory(app.static_folder, request.path[1:])


@app.route('/status')
def status():
    jobid = request.values.get('jobid')
    if jobid:
        job = get_or_cancel_task(jobid)
        if job is None:
            return json.dumps({
                'state': 'FAILURE',
                'message': f'Unknown task id {jobid}'
            })
        job_state, job_result = job.state, job.result
        if job_state == 'PROGRESS':
            return json.dumps({
                'state': job_state,
                'log': job_result['log'],
                'progress': int(100.0 * job_result['current'] / job_result['total'])
            })
        elif job_state == 'SUCCESS':
            return json.dumps({
                'state': job_state,
                'progress': 100
            })
        elif job_state == 'FAILURE':
            is_search_error = isinstance(job_result, SearchError)
            logger.info(f'/status failure. Search error: {is_search_error}. {log_request(request)}')
            return json.dumps({
                'state': job_state,
                'message': str(job_result).replace('\\n', '\n').replace('\\t', '\t'),
                'search_error': is_search_error
            })
        elif job_state == 'STARTED':
            return json.dumps({
                'state': job_state,
                'message': 'Task is starting, please wait...'
            })
        elif job_state == 'PENDING':
            return json.dumps({
                'state': job_state,
                'message': 'Task is in queue, please wait...'
            })
        else:
            return json.dumps({
                'state': 'FAILURE',
                'message': f'Illegal task state {job_state}'
            })
    # no jobid
    return json.dumps({
        'state': 'FAILURE',
        'message': 'No task id'
    })


@app.route('/result')
def result():
    jobid = request.args.get('jobid')
    query = request.args.get('query')
    source = request.args.get('source')
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    noreviews = request.args.get('noreviews') == 'on'  # Include reviews in the initial search phase
    expand = request.args.get('expand') or 0  # Fraction of papers to cover by references
    try:
        if jobid and query and source and limit is not None and sort is not None:
            job = AsyncResult(jobid, app=pubtrends_celery)
            if job and job.state == 'SUCCESS':
                viz, data, log = job.result
                save_predefined(viz, data, log, jobid)
                logger.info(f'/result success {log_request(request)}')
                return render_template('result.html',
                                       query=trim(query, MAX_QUERY_LENGTH),
                                       source=source,
                                       limit=limit,
                                       sort=sort,
                                       max_graph_size=PUBTRENDS_CONFIG.max_graph_size,
                                       version=VERSION,
                                       log=log,
                                       **viz)
            viz_log = load_predefined_viz_log(jobid)
            if viz_log is not None:
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
            analyze_search_terms.apply_async(args=[source, query, sort, int(limit), noreviews, int(expand) / 100],
                                             task_id=jobid, test=app.config['TESTING'])
            return redirect(url_for('.process', query=query, source=source, limit=limit, sort=sort,
                                    noreviews=noreviews, expand=expand,
                                    jobid=jobid))
        else:
            logger.error(f'/result error wrong request {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.error(f'/result error', e)
        return render_template_string(ERROR_OCCURRED), 500


@app.route('/process')
def process():
    if len(request.args) > 0:
        jobid = request.values.get('jobid')

        if not jobid:
            logger.error(f'/process error wrong request {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG_SEARCH)

        query = request.args.get('query') or ''
        analysis_type = request.values.get('analysis_type')
        source = request.values.get('source')
        key = request.args.get('key')
        value = request.args.get('value')
        id = request.args.get('id')

        if key and value:
            logger.info(f'/process key:value search {log_request(request)}')
            query = f'Paper {key}: {value}'
            return render_template('process.html',
                                   redirect_args={'query': quote(query), 'source': source, 'jobid': jobid,
                                                  'limit': request.args.get('limit'), 'sort': ''},
                                   query=trim(query, MAX_QUERY_LENGTH), source=source,
                                   redirect_page="process_paper",  # redirect in case of success
                                   jobid=jobid, version=VERSION)

        elif analysis_type in [ZOOM_IN_TITLE, ZOOM_OUT_TITLE]:
            logger.info(f'/process zoom processing {log_request(request)}')
            return render_template('process.html',
                                   redirect_args={'query': quote(query), 'source': source, 'jobid': jobid,
                                                  'limit': analysis_type, 'sort': ''},
                                   query=trim(query, MAX_QUERY_LENGTH), source=source, limit=analysis_type, sort='',
                                   redirect_page="result",  # redirect in case of success
                                   jobid=jobid, version=VERSION)

        elif analysis_type == PAPER_ANALYSIS_TITLE:
            logger.info(f'/process paper analysis {log_request(request)}')
            return render_template('process.html',
                                   redirect_args={'source': source, 'jobid': jobid, 'id': id,
                                                  'limit': '', 'sort': ''},
                                   query=trim(query, MAX_QUERY_LENGTH), source=source,
                                   redirect_page="paper",  # redirect in case of success
                                   jobid=jobid, version=VERSION)

        elif analysis_type == REVIEW_ANALYSIS_TITLE:
            logger.info(f'/process review {log_request(request)}')
            limit = request.args.get('limit')
            sort = request.args.get('sort')
            return render_template('process.html',
                                   redirect_args={
                                       'query': quote(query), 'source': source,
                                       'limit': limit, 'sort': sort,
                                       'jobid': jobid},
                                   query=trim(query, MAX_QUERY_LENGTH), source=source,
                                   limit=limit, sort=sort,
                                   redirect_page="review",  # redirect in case of success
                                   jobid=jobid, version=VERSION)

        elif query:  # This option should be the last default
            logger.info(f'/process regular search {log_request(request)}')
            limit = request.args.get('limit')
            sort = request.args.get('sort')
            return render_template('process.html',
                                   redirect_args={
                                       'query': quote(query), 'source': source,
                                       'limit': limit, 'sort': sort,
                                       'jobid': jobid
                                   },
                                   query=trim(query, MAX_QUERY_LENGTH), source=source,
                                   limit=limit, sort=sort,
                                   redirect_page="result",  # redirect in case of success
                                   jobid=jobid, version=VERSION)

    logger.error(f'/process error {log_request(request)}')
    return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400


@app.route('/process_paper')
def process_paper():
    jobid = request.values.get('jobid')
    source = request.values.get('source')
    query = request.values.get('query')
    if jobid:
        job = get_or_cancel_task(jobid)
        if job and job.state == 'SUCCESS':
            id_list = job.result
            logger.info(f'/process_paper single paper analysis {log_request(request)}')
            job = analyze_id_list.delay(
                source, ids=id_list, zoom=PAPER_ANALYSIS, query=query, limit=request.values.get('limit'),
                test=app.config['TESTING']
            )
            return redirect(url_for('.process', query=query, analysis_type=PAPER_ANALYSIS_TITLE,
                                    id=id_list[0], source=source, jobid=job.id))
        logger.error(f'/process_paper error job is not success {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_PAPER), 400
    else:
        logger.error(f'/process_paper error wrong request no jobid {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_PAPER), 400


@app.route('/paper')
def paper():
    jobid = request.values.get('jobid')
    source = request.args.get('source')
    pid = request.args.get('id')
    query = request.args.get('query')
    limit = request.args.get('limit')
    try:
        if jobid and pid:
            data = load_predefined_or_result_data(jobid, pubtrends_celery)
            if data is not None:
                logger.info(f'/paper success {log_request(request)}')
                return render_template('paper.html',
                                       **prepare_paper_data(data, source, pid),
                                       max_graph_size=PUBTRENDS_CONFIG.max_graph_size,
                                       version=VERSION)
            else:
                logger.info(f'/paper No job or out-of-date job, restart it {log_request(request)}')
                job = analyze_id_list.apply_async(
                    args=[source, [pid], PAPER_ANALYSIS, query, limit, app.config['TESTING']],
                    task_id=jobid
                )
                return redirect(url_for('.process', query=query, analysis_type=PAPER_ANALYSIS_TITLE,
                                        id=pid, source=source, jobid=job.id))
        else:
            logger.error(f'/paper error wrong request {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG_PAPER), 400
    except Exception as e:
        logger.error(f'/paper error', e)
        return render_template_string(ERROR_OCCURRED), 500


@app.route('/graph')
def graph():
    jobid = request.values.get('jobid')
    query = request.args.get('query')
    source = request.args.get('source')
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    graph_type = request.args.get('type')
    if jobid:
        data = load_predefined_or_result_data(jobid, pubtrends_celery)
        if data is not None:
            loader, url_prefix = Loaders.get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
            analyzer = PapersAnalyzer(loader, PUBTRENDS_CONFIG)
            analyzer.init(data)
            topics_tags = {comp: ', '.join(
                [w[0] for w in analyzer.kwd_df[analyzer.kwd_df['comp'] == comp]['kwd'].values[0][:10]]
            ) for comp in sorted(set(analyzer.df['comp']))}
            if graph_type == "citations":
                graph_cs = PlotPreprocessor.dump_citations_graph_cytoscape(analyzer.df, analyzer.citations_graph)
                logger.info(f'/graph success citations {log_request(request)}')
                return render_template(
                    'graph.html',
                    version=VERSION,
                    source=source,
                    query=trim(query, MAX_QUERY_LENGTH),
                    limit=limit,
                    sort=sort,
                    citation_graph="true",
                    topics_palette_json=json.dumps(Plotter.topics_palette(analyzer.df)),
                    topics_description_json=json.dumps(topics_tags),
                    graph_cytoscape_json=json.dumps(graph_cs)
                )
            else:
                graph_cs = PlotPreprocessor.dump_structure_graph_cytoscape(analyzer.df, analyzer.structure_graph)
                logger.info(f'/graph success structure {log_request(request)}')
                return render_template(
                    'graph.html',
                    version=VERSION,
                    source=source,
                    query=trim(query, MAX_QUERY_LENGTH),
                    limit=limit,
                    sort=sort,
                    citation_graph="false",
                    topics_palette_json=json.dumps(Plotter.topics_palette(analyzer.df)),
                    topics_description_json=json.dumps(topics_tags),
                    graph_cytoscape_json=json.dumps(graph_cs)
                )
        logger.error(f'/graph error job id {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    else:
        logger.error(f'/graph error wrong request {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400


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
        data = load_predefined_or_result_data(jobid, pubtrends_celery)
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


@app.route('/cancel')
def cancel():
    if len(request.args) > 0:
        jobid = request.values.get('jobid')
        if jobid:
            pubtrends_celery.control.revoke(jobid, terminate=True)
            return json.dumps({
                'state': 'CANCELLED',
                'message': f'Successfully cancelled task {jobid}'
            })
        else:
            return json.dumps({
                'state': 'FAILURE',
                'message': f'Failed to cancel task {jobid}'
            })
    return json.dumps({
        'state': 'FAILURE',
        'message': 'Unknown task id'
    })


# Index page
@app.route('/')
def index():
    search_example_source = ''
    search_example_terms = ''
    sources = []
    if PUBTRENDS_CONFIG.pm_enabled:
        sources.append('pm')
    if PUBTRENDS_CONFIG.ss_enabled:
        sources.append('ss')
    if len(sources):
        if random.choice(sources) == 'pm':
            search_example_source = 'Pubmed'
            search_example_terms = PUBTRENDS_CONFIG.pm_search_example_terms
        if random.choice(sources) == 'ss':
            search_example_source = 'Semantic Scholar'
            search_example_terms = PUBTRENDS_CONFIG.ss_search_example_terms
    search_example_terms = [(t, hashlib.md5(t.encode('utf-8')).hexdigest()) for t in search_example_terms]
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
                           ss_enabled=PUBTRENDS_CONFIG.ss_enabled,
                           search_example_source=search_example_source,
                           search_example_terms=search_example_terms)


@app.route('/search_terms', methods=['POST'])
def search_terms():
    logger.info(f'/search_terms {log_request(request)}')
    query = request.form.get('query')  # Original search query
    source = request.form.get('source')  # Pubmed or Semantic Scholar
    sort = request.form.get('sort')  # Sort order
    limit = request.form.get('limit')  # Limit
    noreviews = request.form.get('noreviews') == 'on'  # Include reviews in the initial search phase
    expand = request.form.get('expand')  # Fraction of papers to cover by references
    try:
        if query and source and sort and limit and expand:
            job = analyze_search_terms.delay(source, query=query, limit=int(limit), sort=sort,
                                             noreviews=noreviews, expand=int(expand) / 100,
                                             test=app.config['TESTING'])
            return redirect(url_for('.process', query=query, source=source, limit=limit, sort=sort,
                                    noreviews=noreviews, expand=expand,
                                    jobid=job.id))
        logger.error(f'/search_terms error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_TOPIC), 400
    except Exception as e:
        logger.error(f'/search_terms error', e)
        return render_template_string(ERROR_OCCURRED), 500


@app.route('/search_paper', methods=['POST'])
def search_paper():
    logger.info(f'/search_paper {log_request(request)}')
    data = request.form
    try:
        if 'source' in data and 'key' in data and 'value' in data:
            source = data.get('source')  # Pubmed or Semantic Scholar
            key = data.get('key')
            value = data.get('value')
            limit = data.get('limit')
            job = find_paper_async.delay(source, key, value, test=app.config['TESTING'])
            return redirect(url_for('.process', source=source, key=key, value=value, jobid=job.id, limit=limit))
        logger.error(f'/search_paper error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_PAPER), 400
    except Exception as e:
        logger.error(f'/search_paper error', e)
        return render_template_string(ERROR_OCCURRED), 500


@app.route('/search_list', methods=['POST'])
def search_list():
    logger.info(f'/search_list {log_request(request)}')
    data = request.form
    try:
        if 'source' in data and 'list' in data:
            source = data.get('source')  # Pubmed or Semantic Scholar
            ids_text = data.get('list')
            id_list = [re.sub('[\'"]*', '', pid) for pid in re.split('[ \t\n,;\\.]+', ids_text) if pid]
            id_list = id_list[:PUBTRENDS_CONFIG.max_number_of_articles]
            job = analyze_id_list.delay(source, ids=id_list, query='List of papers', zoom=None,
                                        test=app.config['TESTING'])
            return redirect(url_for('.process', query='List', source=source,
                                    limit=f'{len(id_list)} papers', sort='',
                                    jobid=job.id))
        logger.error(f'/search_list error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_PAPER), 400
    except Exception as e:
        logger.error(f'/search_list error', e)
        return render_template_string(ERROR_OCCURRED), 500


@app.route('/search_pubmed_paper_by_title', methods=['GET'])
def search_pubmed_paper_by_title():
    logger.info(f'/search_paper_by_title {log_request(request)}')
    title = request.values.get('title')
    try:
        if title:
            logger.info(f'/search_paper_by_title {log_request(request)}')
            # Sync call
            loader = Loaders.get_loader('Pubmed', PUBTRENDS_CONFIG)
            ids = loader.find('title', title)
            if not ids:
                return json.dumps([])
            papers = loader.load_publications(ids)
            return json.dumps([
                papers.to_dict(orient='records'),
                [dict(references=loader.load_references(pid, 1000)) for pid in ids]
            ])
        logger.error(f'/search_paper_by_title error missing title {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_PAPER), 400
    except Exception as e:
        logger.error(f'/search_paper_by_title error', e)
        return render_template_string(ERROR_OCCURRED), 500


@app.route('/process_ids', methods=['POST'])
def process_ids():
    logger.info(f'/process_ids {log_request(request)}')
    source = request.form.get('source')  # Pubmed or Semantic Scholar
    query = request.form.get('query')  # Original search query
    try:
        if source and query and 'id_list' in request.form:
            id_list = request.form.get('id_list').split(',')
            zoom = request.form.get('zoom')
            analysis_type = zoom_name(zoom)
            job = analyze_id_list.delay(
                source, ids=id_list, zoom=int(zoom), query=query, limit=None,
                test=app.config['TESTING']
            )
            return redirect(url_for('.process', query=query, analysis_type=analysis_type, source=source, jobid=job.id))
        logger.error(f'/process_ids error {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
    except Exception as e:
        logger.error(f'/process_ids error', e)
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
            data = load_predefined_or_result_data(jobid, pubtrends_celery)
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
        logger.error(f'/export_results error', e)
        return render_template_string(ERROR_OCCURRED), 500


@app.route('/about.html', methods=['GET'])
def about():
    return render_template('about.html', version=VERSION)


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
        ftype = data.get('type')
        message = data.get('message')
        email = data.get('email')
        jobid = data.get('jobid')
        logger.info('Feedback ' + json.dumps(dict(type=ftype, message=message, email=email, jobid=jobid)))
    else:
        logger.error(f'/feedback error')
        return render_template_string(ERROR_OCCURRED), 500
    return render_template_string('Thanks you for the feedback!'), 200


#######################
# Admin functionality #
#######################

configure_admin_functions(app, logfile)

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
    app.run(host='0.0.0.0', debug=True, extra_files=['templates/'])
