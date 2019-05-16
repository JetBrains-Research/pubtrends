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
import random
import time
import uuid

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


# Tasks will be served by Celery
@celery.task
def slow_proc():
    NTOTAL = 10
    for i in range(NTOTAL):
        time.sleep(random.random())
        current_task.update_state(state='PROGRESS',
                                  meta={'current': i, 'total': NTOTAL})
    return '<h1>42</h1>'


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
                'progress': 100,
                'result': job.result
            })
    return '{}'


@app.route('/enqueue')
def enqueue():
    job = slow_proc.delay()
    return render_template_string('''
<head>
    <script src="//ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css">
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js"></script>
    <script>
    function poll() {
        $.ajax("{{url_for('.progress', jobid=JOBID)}}", {
            dataType: "json", success: function(resp) {
                console.log(resp);
                if (resp.progress !== undefined) {
                    $('.progress-bar').css('width', resp.progress + '%').attr('aria-valuenow', resp.progress);
                    $('.progress-bar-label').text(resp.progress + '%');
                    if (resp.progress < 100) {
                        setTimeout(poll, 1000);
                    } else {
                        $('#progress').hide();
                        $('#success').html(resp.result);
                    }
                } else {
                    setTimeout(poll, 1000);
                }
            }
        });
    }
    $(function() {
        poll();
    });
    </script>
</head>
<body>
    <div id="progress" class="progress">
        <div class="progress-bar progress-bar-striped active" role="progressbar" 
            aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%">
            <span class="progress-bar-label">0%</span>
        </div>
    </div>
    <div id="success"/>
</body>

''', JOBID=job.id)


@app.route('/')
def index():
    return render_template_string('''\
<a href="{{ url_for('.enqueue') }}">launch job</a>
''')


if __name__ == '__main__':
    app.run()
