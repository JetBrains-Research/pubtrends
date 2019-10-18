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
from models.keypaper.config import PubtrendsConfig
from models.keypaper.paper import prepare_paper_data, prepare_papers_data
from models.keypaper.utils import zoom_name, PAPER_ANALYSIS, ZOOM_IN_TITLE, PAPER_ANALYSIS_TITLE

# logging.basicConfig(level=logging.NOTSET)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

app = Flask(__name__)


@app.route('/status')
def status():
    jobid = request.values.get('jobid')
    if jobid:
        job = get_or_cancel_task(jobid)
        if job is None:
            return json.dumps({
                'state': 'FAILURE',
                'message': f'Unknown search id {jobid}'
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
        'message': f'Unknown search id {jobid}'
    })


@app.route('/result')
def result():
    jobid = request.values.get('jobid')
    query = request.args.get('query')
    if jobid:
        job = complete_task(jobid)
        if job and job.state == 'SUCCESS':
            data, _ = job.result
            return render_template('result.html', search_string=query,
                                   version=PUBTRENDS_CONFIG.version,
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
            return render_template('process.html',
                                   args={'source': source, 'query': quote(f'Paper {key}: {value}'), 'jobid': jobid},
                                   search_string=f'Paper {key}: {value}',
                                   subpage="process_paper",  # redirect in case of success
                                   jobid=jobid, version=PUBTRENDS_CONFIG.version)

        elif analysis_type in [ZOOM_IN_TITLE, ZOOM_IN_TITLE]:
            logging.debug('/process zoom processing')
            query = f"{analysis_type} analysis of {query} at {source}"
            return render_template('process.html',
                                   args={'query': quote(query), 'jobid': jobid},
                                   search_string=query, subpage="result",  # redirect in case of success
                                   jobid=jobid, version=PUBTRENDS_CONFIG.version)

        elif analysis_type == PAPER_ANALYSIS_TITLE:
            logging.debug('/process paper analysis')
            return render_template('process.html',
                                   args={'source': source, 'jobid': jobid, 'id': id},
                                   search_string=query,
                                   subpage="paper",  # redirect in case of success
                                   jobid=jobid, version=PUBTRENDS_CONFIG.version)
        elif query:
            logging.debug('/process regular search')
            query = f'{query} at {source}'
            return render_template('process.html',
                                   args={'query': quote(query), 'jobid': jobid},
                                   search_string=query,
                                   subpage="result",  # redirect in case of success
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
            _, data = job.result
            return render_template('paper.html', **prepare_paper_data(data, source, pid),
                                   version=PUBTRENDS_CONFIG.version)

    return render_template_string("Something went wrong...")


@app.route('/papers')
def show_ids():
    jobid = request.values.get('jobid')
    source = request.args.get('source')
    comp = request.args.get('comp')
    if comp is not None:
        comp = int(comp) - 1  # Component was exposed so it was 1-based
    if jobid:
        job = complete_task(jobid)
        if job and job.state == 'SUCCESS':
            _, data = job.result
            return render_template('papers.html',
                                   version=PUBTRENDS_CONFIG.version,
                                   source=source,
                                   papers=prepare_papers_data(data, source, comp))

    raise Exception(f"Request does not contain necessary params: {request}")


@app.route('/cancel')
def cancel():
    if len(request.args) > 0:
        jobid = request.values.get('jobid')
        if jobid:
            celery.control.revoke(jobid, terminate=True)
            return json.dumps({
                'state': 'CANCELLED',
                'message': f'Successfully cancelled search {jobid}'
            })
        else:
            return json.dumps({
                'state': 'FAILURE',
                'message': f'Failed to cancel search {jobid}'
            })
    return json.dumps({
        'state': 'FAILURE',
        'message': f'Unknown search id'
    })


# Index page
@app.route('/')
def index():
    logging.debug('/ landing page')
    return render_template('main.html',
                           version=PUBTRENDS_CONFIG.version,
                           limits=PUBTRENDS_CONFIG.show_max_articles_options,
                           default_limit=PUBTRENDS_CONFIG.show_max_articles_default_value,
                           development=PUBTRENDS_CONFIG.development,
                           search_example_terms=random.choice(PUBTRENDS_CONFIG.search_example_terms))


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
        return redirect(url_for('.process', query=query, source=source, jobid=job.id))

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

# # With debug=True, Flask server will auto-reload on changes
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True, extra_files=['templates/'])
