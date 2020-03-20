import html
import json
import logging
import random
from urllib.parse import quote

from flask import (
    Flask, request, redirect, url_for,
    render_template, render_template_string
)

from models.celery.tasks import celery, find_paper_async, analyze_search_terms, analyze_id_list
from models.celery.tasks_cache import get_or_cancel_task, complete_task
from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.keypaper.paper import prepare_paper_data, prepare_papers_data, get_loader_and_url_prefix
from models.keypaper.utils import zoom_name, PAPER_ANALYSIS, ZOOM_IN_TITLE, PAPER_ANALYSIS_TITLE, trim
from models.keypaper.visualization_data import PlotPreprocessor

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

MAX_QUERY_LENGTH = 60

app = Flask(__name__)

# Check to see if our Flask application is being run directly or through Gunicorn,
# and then set your Flask application logger’s handlers to the same as Gunicorn’s.
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


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
        if job.state == 'PROGRESS':
            return json.dumps({
                'state': job.state,
                'log': job.result['log'],
                'progress': int(100.0 * job.result['current'] / job.result['total'])
            })
        elif job.state == 'SUCCESS':
            # Mark job as complete to avoid time expiration
            complete_task(jobid)
            return json.dumps({
                'state': job.state,
                'progress': 100
            })
        elif job.state == 'FAILURE':
            return json.dumps({
                'state': job.state,
                'message': html.unescape(str(job.result).replace('\\n', '\n').replace('\\t', '\t')[2:-2])
            })
        elif job.state == 'PENDING':
            return json.dumps({
                'state': job.state,
                'message': 'Task is in queue, please wait...'
            })

    # no jobid
    return json.dumps({
        'state': 'FAILURE',
        'message': f'Unknown task id {jobid}'
    })


@app.route('/result')
def result():
    jobid = request.values.get('jobid')
    query = request.args.get('query')
    source = request.args.get('source')
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    if jobid:
        job = complete_task(jobid)
        if job and job.state == 'SUCCESS':
            data, _, log = job.result
            return render_template('result.html',
                                   query=query,
                                   source=source,
                                   limit=limit,
                                   sort=sort,
                                   version=PUBTRENDS_CONFIG.version,
                                   log=log,
                                   **data)

    return render_template_string("Something went wrong...")


@app.route('/process')
def process():
    if len(request.args) > 0:
        jobid = request.values.get('jobid')

        if not jobid:
            return render_template_string("Something went wrong...")

        query = request.args.get('query')
        analysis_type = request.values.get('analysis_type')
        source = request.values.get('source')
        key = request.args.get('key')
        value = request.args.get('value')
        id = request.args.get('id')

        if key and value:
            logging.debug('/process key:value search')
            query = f'Paper {key}: {value}'
            return render_template('process.html',
                                   redirect_args={'query': quote(query), 'source': source, 'jobid': jobid},
                                   query=trim(query, MAX_QUERY_LENGTH), source=source,
                                   redirect_page="process_paper",  # redirect in case of success
                                   jobid=jobid, version=PUBTRENDS_CONFIG.version)

        elif analysis_type in [ZOOM_IN_TITLE, ZOOM_IN_TITLE]:
            logging.debug('/process zoom processing')
            query = f"{analysis_type} analysis of {query}"
            return render_template('process.html',
                                   redirect_args={'query': quote(query), 'source': source, 'jobid': jobid},
                                   query=trim(query, MAX_QUERY_LENGTH), source=source,
                                   redirect_page="result",  # redirect in case of success
                                   jobid=jobid, version=PUBTRENDS_CONFIG.version)

        elif analysis_type == PAPER_ANALYSIS_TITLE:
            logging.debug('/process paper analysis')
            return render_template('process.html',
                                   redirect_args={'source': source, 'jobid': jobid, 'id': id},
                                   query=trim(query, MAX_QUERY_LENGTH), source=source,
                                   redirect_page="paper",  # redirect in case of success
                                   jobid=jobid, version=PUBTRENDS_CONFIG.version)
        elif query:
            logging.debug('/process regular search')
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
                                   jobid=jobid, version=PUBTRENDS_CONFIG.version)

    return render_template_string("Something went wrong...")


@app.route('/process_paper')
def process_paper():
    jobid = request.values.get('jobid')
    source = request.values.get('source')
    query = request.values.get('query')
    if jobid:
        job = get_or_cancel_task(jobid)
        if job and job.state == 'SUCCESS':
            id_list = job.result
            logging.debug('/process_paper single paper analysis')
            if len(id_list) == 1:
                job = analyze_id_list.delay(source, id_list=id_list, zoom=PAPER_ANALYSIS, query=query)
                return redirect(url_for('.process', query=query, analysis_type=PAPER_ANALYSIS_TITLE,
                                        id=id_list[0], source=source, jobid=job.id))
            elif len(id_list) == 0:
                return render_template_string('Found no papers matching specified key - value pair')
            else:
                return render_template_string('Found multiple papers matching your search')


@app.route('/paper')
def paper():
    jobid = request.values.get('jobid')
    source = request.args.get('source')
    pid = request.args.get('id')
    if jobid:
        job = complete_task(jobid)
        if job and job.state == 'SUCCESS':
            _, data, _ = job.result
            return render_template('paper.html', **prepare_paper_data(data, source, pid),
                                   version=PUBTRENDS_CONFIG.version)

    return render_template_string("Something went wrong...")


@app.route('/graph')
def graph():
    jobid = request.values.get('jobid')
    query = request.args.get('query')
    source = request.args.get('source')
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    graph_type = request.args.get('type')
    if jobid:
        job = complete_task(jobid)
        if job and job.state == 'SUCCESS':
            _, data, _ = job.result
            loader, url_prefix = get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
            analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
            analyzer.init(data)
            min_year, max_year = int(analyzer.df['year'].min()), int(analyzer.df['year'].max())
            if graph_type == "citations":
                graph_cs = PlotPreprocessor.dump_citations_graph_cytoscape(analyzer.df, analyzer.citations_graph)
                return render_template('graph.html',
                                       version=PUBTRENDS_CONFIG.version,
                                       source=source,
                                       query=query,
                                       limit=limit,
                                       sort=sort,
                                       citation_graph="true",
                                       min_year=min_year,
                                       max_year=max_year,
                                       graph_cytoscape_json=json.dumps(graph_cs))
            else:
                graph_cs = PlotPreprocessor.dump_structure_graph_cytoscape(analyzer.df, analyzer.paper_relations_graph)
                return render_template('graph.html',
                                       version=PUBTRENDS_CONFIG.version,
                                       source=source,
                                       query=query,
                                       limit=limit,
                                       sort=sort,
                                       citation_graph="false",
                                       min_year=min_year,
                                       max_year=max_year,
                                       graph_cytoscape_json=json.dumps(graph_cs))

    return render_template_string("Something went wrong...")


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
        search_string += f', topic: {comp}'
        comp = int(comp) - 1  # Component was exposed so it was 1-based

    word = request.args.get('word')
    if word is not None:
        search_string += f', word: {word}'

    author = request.args.get('author')
    if author is not None:
        search_string += f', author: {author}'

    journal = request.args.get('journal')
    if journal is not None:
        search_string += f', journal: {journal}'

    papers_list = request.args.get('papers_list')
    if papers_list == 'top':
        search_string += f', top papers'
    if papers_list == 'year':
        search_string += f', papers of the year'
    if papers_list == 'hot':
        search_string += f', hot papers'

    if jobid:
        job = complete_task(jobid)
        if job and job.state == 'SUCCESS':
            _, data, _ = job.result
            return render_template('papers.html',
                                   version=PUBTRENDS_CONFIG.version,
                                   source=source,
                                   query=query,
                                   search_string=search_string,
                                   limit=limit,
                                   sort=sort,
                                   papers=prepare_papers_data(data, source, comp, word, author, journal, papers_list))

    raise Exception(f"Request does not contain necessary params: {request}")


@app.route('/cancel')
def cancel():
    if len(request.args) > 0:
        jobid = request.values.get('jobid')
        if jobid:
            celery.control.revoke(jobid, terminate=True)
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
        'message': f'Unknown task id'
    })


# Index page
@app.route('/')
def index():
    logging.debug('/ landing page')

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
            search_example_terms = random.choice(PUBTRENDS_CONFIG.pm_search_example_terms)
        if random.choice(sources) == 'ss':
            search_example_source = 'Semantic Scholar'
            search_example_terms = random.choice(PUBTRENDS_CONFIG.ss_search_example_terms)

    return render_template('main.html',
                           version=PUBTRENDS_CONFIG.version,
                           limits=PUBTRENDS_CONFIG.show_max_articles_options,
                           default_limit=PUBTRENDS_CONFIG.show_max_articles_default_value,
                           development=PUBTRENDS_CONFIG.development,
                           pm_enabled=PUBTRENDS_CONFIG.pm_enabled,
                           ss_enabled=PUBTRENDS_CONFIG.ss_enabled,
                           search_example_source=search_example_source,
                           search_example_terms=search_example_terms)


@app.route('/search_terms', methods=['POST'])
def search_terms():
    logging.debug('/search_terms')
    query = request.form.get('query')  # Original search query
    source = request.form.get('source')  # Pubmed or Semantic Scholar
    sort = request.form.get('sort')  # Sort order
    limit = request.form.get('limit')  # Limit

    if query and source and sort:
        logging.debug(f'/ regular search')
        job = analyze_search_terms.delay(source, query=query, limit=limit, sort=sort)
        return redirect(url_for('.process', query=query, source=source, limit=limit, sort=sort, jobid=job.id))

    raise Exception(f"Request does not contain necessary params: {request}")


@app.route('/search_paper', methods=['POST'])
def search_paper():
    logging.debug('/search_paper')
    source = request.form.get('source')  # Pubmed or Semantic Scholar

    if source and 'key' in request.form and 'value' in request.form:
        logging.debug(f'/ paper search')
        key = request.form.get('key')
        value = request.form.get('value')
        job = find_paper_async.delay(source, key, value)
        return redirect(url_for('.process', source=source, key=key, value=value, jobid=job.id))

    raise Exception(f"Request does not contain necessary params: {request}")


@app.route('/process_ids', methods=['POST'])
def process_ids():
    logging.debug('/process_ids')
    source = request.form.get('source')  # Pubmed or Semantic Scholar
    query = request.form.get('query')  # Original search query

    if source and query and 'id_list' in request.form:
        id_list = request.form.get('id_list').split(',')
        zoom = request.form.get('zoom')
        analysis_type = zoom_name(zoom)
        job = analyze_id_list.delay(source, id_list=id_list, zoom=int(zoom), query=query)
        return redirect(url_for('.process', query=query, analysis_type=analysis_type, source=source, jobid=job.id))

    raise Exception(f"Request does not contain necessary params: {request}")


def get_app():
    return app

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True, extra_files=['templates/'])
