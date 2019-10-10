import html
import json
import random
import logging

from urllib.parse import quote

from celery.result import AsyncResult
from flask import (
    Flask, request, redirect, url_for,
    render_template, render_template_string
)

from models.celery.tasks import celery, find_paper_async, analyze_topic_async
from models.keypaper.config import PubtrendsConfig
from models.keypaper.paper import prepare_paper_data
from models.keypaper.utils import zoom_name, DOUBLE_ZOOM_OUT

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

app = Flask(__name__)


@app.route('/progress')
def progress():
    jobid = request.values.get('jobid')
    if jobid:
        job = AsyncResult(jobid, app=celery)
        if job.state == 'PROGRESS':
            return json.dumps({
                'state': job.state,
                'log': job.result['log'],
                'progress': int(100.0 * job.result['current'] / job.result['total'])
            })
        elif job.state == 'SUCCESS':
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
            return json.dumps({'state': job.state,
                               'message': 'Task is in queue, please wait...',
                               })

    # no jobid
    return json.dumps({
        'state': 'FAILURE',
        'message': f'Unknown search id {jobid}'
    })


@app.route('/result')
def result():
    jobid = request.values.get('jobid')
    terms = request.args.get('terms')
    if jobid:
        job = AsyncResult(jobid, app=celery)
        data, _ = job.result
        if job.state == 'SUCCESS':
            return render_template('result.html', search_string=terms,
                                   version=PUBTRENDS_CONFIG.version,
                                   **data)

    return render_template_string("Something went wrong...")


@app.route('/process')
def process():
    if len(request.args) > 0:
        jobid = request.values.get('jobid')
        terms = request.args.get('terms')
        analysis_type = request.values.get('analysis_type')
        source = request.values.get('source')
        key = request.args.get('key')
        value = request.args.get('value')

        if jobid:
            if terms:
                terms += f' at {source}'
                return render_template('process.html',
                                       args={'terms': quote(terms), 'jobid': jobid},
                                       search_string=terms, subpage="result",
                                       jobid=jobid, version=PUBTRENDS_CONFIG.version)
            elif key and value:
                return render_template('process.html',
                                       args={'source': source, 'jobid': jobid},
                                       search_string=f'{key}: {value}', subpage="search",
                                       jobid=jobid, version=PUBTRENDS_CONFIG.version)
            else:
                if analysis_type in ['detailed', 'expanded']:
                    terms = f"{analysis_type} analysis of {terms} at {source}"

                    return render_template('process.html',
                                           args={'key': "terms", 'value': quote(terms), 'jobid': jobid},
                                           search_string=terms, subpage="result",
                                           jobid=jobid, version=PUBTRENDS_CONFIG.version)
                else:
                    return render_template('process.html',
                                           args={'source': source, 'id': analysis_type, 'jobid': jobid},
                                           search_string=f"Paper analysis at {source}", subpage="paper",
                                           jobid=jobid, version=PUBTRENDS_CONFIG.version)

    return render_template_string("Something went wrong...")


@app.route('/search')
def search():
    jobid = request.values.get('jobid')
    source = request.values.get('source')
    if jobid:
        find_job = AsyncResult(jobid, app=celery)

        if find_job.state == 'SUCCESS':
            ids = find_job.result
            logging.debug('/search single paper analysis')
            if len(ids) == 1:
                job = analyze_topic_async.delay(source, id_list=ids, zoom=DOUBLE_ZOOM_OUT)
                return redirect(url_for('.process', terms=None, analysis_type=ids[0],
                                        source=source, jobid=job.id))
            elif len(ids) == 0:
                return render_template_string('Found no papers matching specified key - value pair')
            else:
                return render_template_string('Found multiple papers matching your search')


@app.route('/paper')
def paper():
    jobid = request.values.get('jobid')
    source = request.args.get('source')
    pid = request.args.get('id')
    if jobid:
        job = AsyncResult(jobid, app=celery)
        _, data = job.result

        if job.state == 'SUCCESS':
            return render_template('paper.html', **prepare_paper_data(data, source, pid),
                                   version=PUBTRENDS_CONFIG.version)

    return render_template_string("Something went wrong...")


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
@app.route('/', methods=['GET', 'POST'])
def index():
    logging.debug('/ serving landing page')
    if request.method == 'GET':
        return render_template('main.html',
                               version=PUBTRENDS_CONFIG.version,
                               amounts=PUBTRENDS_CONFIG.show_max_articles_options,
                               default_amount=PUBTRENDS_CONFIG.show_max_articles_default_value,
                               development=PUBTRENDS_CONFIG.development,
                               search_example_terms=random.choice(PUBTRENDS_CONFIG.search_example_terms))

    logging.debug('/ search activity')
    # Common keys
    source = request.form.get('source')  # Pubmed or Semantic Scholar
    sort = request.form.get('sort')      # Sort order
    amount = request.form.get('amount')  # Amount
    terms = request.form.get('terms') if 'terms' in request.form else None

    if 'id_list' in request.form:
        logging.debug(f'/ zoom')
        id_list = request.form.get('id_list').split(',')
        zoom = request.form.get('zoom')
        analysis_type = zoom_name(zoom)
        job = analyze_topic_async.delay(source, terms=terms, id_list=id_list, zoom=int(zoom), sort=sort, amount=amount)
        return redirect(url_for('.process', terms=terms, analysis_type=analysis_type, source=source, jobid=job.id))

    elif 'key' in request.form and 'value' in request.form:
        logging.debug(f'/ paper search')
        key = request.form.get('key')
        value = request.form.get('value')
        job = find_paper_async.delay(source, key, value)
        return redirect(url_for('.process', source=source, key=key, value=value, jobid=job.id))

    elif terms:
        logging.debug(f'/ regular search')
        job = analyze_topic_async.delay(source, terms=terms, sort=sort, amount=amount)
        return redirect(url_for('.process', terms=terms, source=source, jobid=job.id))

    raise Exception("Request contains no parameters")


def get_app():
    return app

# # With debug=True, Flask server will auto-reload on changes
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True, extra_files=['templates/'])
