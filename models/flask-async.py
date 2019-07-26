"""
1. Install and start Redis.
2. start server
    python flask-async.py
3. start worker
    celery -A flask-async.celery worker -c 1 --loglevel=DEBUG
4. browse to localhost:5000/

Adopted from https://gist.github.com/whacked/c1feef2bf7a3a014178c
"""
import json
import os

import flask
from bokeh.embed import components
from celery import Celery, current_task
from celery.result import AsyncResult
from flask import (
    Flask, request, redirect,
    render_template, render_template_string
)
from keypaper.analysis import KeyPaperAnalyzer
from keypaper.pm_loader import PubmedLoader
from keypaper.ss_loader import SemanticScholarLoader
from keypaper.visualization import Plotter

# Configure according REDIS server
REDIS_SERVER_URL = 'localhost'
CELERY_BROKER_HOST = f'redis://{REDIS_SERVER_URL}:6379'
celery = Celery(os.path.splitext(__file__)[0],
                backend=CELERY_BROKER_HOST + '/1',
                broker=CELERY_BROKER_HOST + '/1')


# Tasks will be served by Celery,
# specify task name explicitly to avoid problems with modules
@celery.task(name='analyze_async')
def analyze_async(source, terms):
    if source == 'Pubmed':
        loader = PubmedLoader(test=False)
    elif source == 'Semantic Scholar':
        loader = SemanticScholarLoader(test=False)
    else:
        raise Exception(f"Unknown source {source}")
    analyzer = KeyPaperAnalyzer(loader)
    # current_task is from @celery.task
    log = analyzer.launch(*terms, task=current_task)

    # Initialize plotter after completion of analysis
    plotter = Plotter(analyzer)

    # Subtopic evolution is ignored for now.
    # Order is important here!
    return {
        'log': log,
        'chord_cocitations': [components(plotter.chord_diagram_components())],
        'component_size_summary': [components(plotter.component_size_summary())],
        'subtopic_timeline_graphs': [components(p) for p in plotter.subtopic_timeline_graphs()],
        'top_cited_papers': [components(plotter.top_cited_papers())],
        'max_gain_papers': [components(plotter.max_gain_papers())],
        'max_relative_gain_papers': [components(plotter.max_relative_gain_papers())],
        'subtopic_evolution': [components(plotter.subtopic_evolution())]
        # TODO: this doesn't work
        # 'citations_dynamics': [components(plotter.article_citation_dynamics())],
    }


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
                'message': str(job.result)
            })

    # PENDING or no jobid
    return '{}'


@app.route('/result')
def result():
    jobid = request.values.get('jobid')
    terms = request.args.get('terms').split('+')
    if jobid:
        job = AsyncResult(jobid, app=celery)
        if job.state == 'SUCCESS':
            return render_template('result.html', search_string=' '.join(terms), **job.result)

    return render_template_string("Something went wrong...")


@app.route('/process')
def process():
    if len(request.args) > 0:
        jobid = request.values.get('jobid')
        terms = request.args.get('terms').split('+')
        if jobid:
            return render_template('process.html', search_string=' '.join(terms), JOBID=jobid)

    return render_template_string("Something went wrong...")


# Index page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        terms = request.form.get('terms').split(' ')
        source = request.form.get('source')

        redirect_url = '+'.join(terms)
        if len(terms) > 0:
            # Submit Celery task
            job = analyze_async.delay(source, terms)
            return redirect(flask.url_for('.process', terms=redirect_url, jobid=job.id))

    return render_template('main.html')


# With debug=True, Flask server will auto-reload on changes
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, extra_files=['templates/'])
