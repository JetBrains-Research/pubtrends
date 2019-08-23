import html
import json
from urllib.parse import quote

import flask
from celery.result import AsyncResult
from flask import (
    Flask, request, redirect,
    render_template, render_template_string
)

from models.celery.tasks import celery, analyze_async
from models.keypaper.config import PubtrendsConfig

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
                'message': html.unescape(str(job.result).replace('\\n', '\n')[2:-2])
            })
        elif job.state == 'PENDING':
            return json.dumps({'state': job.state,
                               'message': 'Task is in queue, please wait',
                               })

    # no jobid
    return '{}'


@app.route('/result')
def result():
    jobid = request.values.get('jobid')
    terms = request.args.get('terms')
    if jobid:
        job = AsyncResult(jobid, app=celery)
        if job.state == 'SUCCESS':
            return render_template('result.html', search_string=terms,
                                   version=PUBTRENDS_CONFIG.version,
                                   **job.result)

    return render_template_string("Something went wrong...")


@app.route('/process')
def process():
    if len(request.args) > 0:
        jobid = request.values.get('jobid')
        terms = request.args.get('terms')
        analysis_type = request.values.get("analysis_type")

        if jobid:
            if not terms:
                terms = f"{analysis_type} analysis of the previous query"

            return render_template('process.html', search_string=terms,
                                   url_search_string=quote(terms), JOBID=jobid,
                                   version=PUBTRENDS_CONFIG.version)

    return render_template_string("Something went wrong...")


# Index page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        terms, id_list, zoom = '', '', ''
        source = request.form.get('source')
        if 'terms' in request.form:
            terms = request.form.get('terms')
            analysis_type = ''
        elif 'id_list' in request.form:
            id_list = request.form.get('id_list').split(',')
            zoom = request.form.get('zoom')
            analysis_type = 'expanded' if zoom == 'out' else 'detailed'
        else:
            raise Exception("Request should contain either terms or list of ids")

        sort = request.form.get('sort')
        amount = request.form.get('amount')
        if len(terms) > 0 or id_list:
            # Submit Celery task
            job = analyze_async.delay(source, terms=terms, id_list=id_list, zoom=zoom, sort=sort, amount=amount)
            return redirect(flask.url_for('.process', terms=terms, analysis_type=analysis_type, jobid=job.id))

    return render_template('main.html', version=PUBTRENDS_CONFIG.version,
                           amounts=PUBTRENDS_CONFIG.show_max_articles_options,
                           default_amount=PUBTRENDS_CONFIG.show_max_articles_default_value)


def get_app():
    return app

# # With debug=True, Flask server will auto-reload on changes
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True, extra_files=['templates/'])
