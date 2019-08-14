import html
import json
from urllib.parse import quote

from celery.result import AsyncResult
from flask import (
    Flask, request, redirect, url_for,
    render_template, render_template_string
)

from models.celery.tasks import celery, analyze_paper_async, analyze_topic_async, prepare_paper_data
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
        key = request.args.get('key')
        value = request.args.get('value')
        if jobid:
            if terms:
                terms = terms.split('+')
                return render_template('process.html', search_string=terms,
                                       subpage="result", key="terms", value=quote(terms),
                                       JOBID=jobid, version=PUBTRENDS_CONFIG.version)
            if key and value:
                return render_template('process.html', search_string=f'{key}: {value}',
                                       subpage="paper", query=quote(f'{key}+{value}'),
                                       JOBID=jobid, version=PUBTRENDS_CONFIG.version)

    return render_template_string("Something went wrong...")


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


# Index page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        terms = request.form.get('terms')
        source = request.form.get('source')
        sort = request.form.get('sort')
        amount = request.form.get('amount')
        key = request.form.get('key')
        value = request.form.get('value')

        if terms and len(terms) > 0:
            # Submit Celery task for topic analysis
            job = analyze_topic_async.delay(terms, source, sort, amount)
            return redirect(url_for('.process', terms=terms, jobid=job.id))
        elif value and len(value) > 0:
            # Submit Celery task for paper analysis
            job = analyze_paper_async.delay(source, key, value)
            return redirect(url_for('.process', key=key, value=value, jobid=job.id))

    return render_template('main.html', version=PUBTRENDS_CONFIG.version,
                           amounts=PUBTRENDS_CONFIG.show_max_articles_options,
                           default_amount=PUBTRENDS_CONFIG.show_max_articles_default_value)


def get_app():
    return app

# # With debug=True, Flask server will auto-reload on changes
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True, extra_files=['templates/'])
