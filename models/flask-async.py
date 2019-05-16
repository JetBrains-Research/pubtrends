"""
1. Install and start Redis.
2. start server
    python flask-async.py
3. start worker
    celery -A flask-async.celery worker -c 1 --loglevel=DEBUG
4. browse to localhost:5000/

Adopted from https://gist.github.com/whacked/c1feef2bf7a3a014178c
"""
from bokeh.embed import components
from keypaper.analysis import KeyPaperAnalyzer
from keypaper.visualization import Plotter

import json
import flask
import os

from celery import Celery, current_task
from celery.result import AsyncResult

from flask import Flask, \
    request, redirect, flash, \
    url_for, session, g, \
    render_template, render_template_string

# Configure according REDIS server
REDIS_SERVER_URL = 'localhost'
CELERY_BROKER_HOST = f'redis://{REDIS_SERVER_URL}:6379'
celery = Celery(os.path.splitext(__file__)[0],
                backend=CELERY_BROKER_HOST + '/1',
                broker=CELERY_BROKER_HOST + '/1')


# Tasks will be served by Celery,
# specify task name explicitly to avoid problems with modules
@celery.task(name='analyze_async')
def analyze_async(terms):
    analyzer = KeyPaperAnalyzer()
    plotter = Plotter(analyzer)
    # current_task is from @celery.task
    analyzer.launch(*terms, task=current_task)
    data = []
    for p in plotter.subtopic_timeline_graphs():
        data.append(components(p))
    return data


app = Flask(__name__)


@app.route('/progress')
def progress():
    jobid = request.values.get('jobid')
    if jobid:
        job = AsyncResult(jobid, app=celery)
        if job.state == 'PROGRESS':
            return json.dumps({
                'state': job.state,
                'progress': int(100.0 * job.result['current'] / job.result['total'])
            })
        elif job.state == 'SUCCESS':
            return json.dumps({
                'state': job.state,
                'progress': 100
            })
    # PENDING or no jobid
    return '{}'


@app.route('/result')
def result():
    # TODO don't expose JOB_ID
    jobid = request.values.get('jobid')
    terms = request.args.get('terms').split('+')
    if jobid:
        job = AsyncResult(jobid, app=celery)
        if job.state == 'SUCCESS':
            return render_template('result.html', search_string=' '.join(terms), data=job.result)

    return render_template_string("Something went wrong...")


@app.route('/process')
def process():
    if len(request.args) > 0:
        terms = request.args.get('terms').split('+')
        # Submit Celery task
        job = analyze_async.delay(terms)
        return render_template('process.html', search_string=' '.join(terms), JOBID=job.id)

    return render_template_string("Something went wrong...")


# Index page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        terms = request.form.get('terms').split(' ')
        redirect_url = '+'.join(terms)
        if len(terms) > 0:
            return redirect(flask.url_for('.process', terms=redirect_url))

    return render_template('main.html')


# With debug=True, Flask server will auto-reload on changes
if __name__ == '__main__':
    app.run(debug=True, extra_files=['templates/'])
