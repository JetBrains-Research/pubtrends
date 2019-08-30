import html
import json
from urllib.parse import quote

from celery.result import AsyncResult
from flask import (
    Flask, request, redirect, url_for,
    render_template, render_template_string
)

from models.celery.tasks import celery, find_paper_async, analyze_topic_async, prepare_paper_data
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
        source = request.args.get('source')
        terms = request.args.get('terms')
        key = request.args.get('key')
        value = request.args.get('value')
        analysis_type = request.values.get("analysis_type")
        print(jobid, source, terms, key, value, analysis_type)

        if jobid:
            if terms:
                return render_template('process.html', search_string=terms,
                                       subpage="result", args={'terms': quote(terms), 'jobid': jobid},
                                       JOBID=jobid, version=PUBTRENDS_CONFIG.version)
            elif key and value:
                return render_template('process.html', search_string=f'{key}: {value}',
                                       subpage="search", args={'source': source, 'jobid': jobid},
                                       JOBID=jobid, version=PUBTRENDS_CONFIG.version)
            else:
                if analysis_type in ['detailed', 'expanded']:
                    terms = f"{analysis_type} analysis of the previous query"

                    return render_template('process.html', search_string=terms,
                                           subpage="result", JOBID=jobid, version=PUBTRENDS_CONFIG.version,
                                           args={'key': "terms", 'value': quote(terms), 'jobid': jobid})
                else:
                    return render_template('process.html', search_string=f"Paper analysis at {source}",
                                           subpage="paper", JOBID=jobid, version=PUBTRENDS_CONFIG.version,
                                           args={'source': source, 'id': analysis_type, 'jobid': jobid})

    return render_template_string("Something went wrong...")


@app.route('/search')
def search():
    jobid = request.values.get('jobid')
    source = request.values.get('source')
    if jobid:
        find_job = AsyncResult(jobid, app=celery)

        if find_job.state == 'SUCCESS':
            ids = find_job.result
            if len(ids) == 1:
                job = analyze_topic_async.delay(source, id_list=ids, zoom=2)
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


# Index page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        terms, id_list, zoom, key, value, analysis_type = '', '', 0, '', '', ''
        source = request.form.get('source')

        if 'terms' in request.form:
            terms = request.form.get('terms')
        elif 'id_list' in request.form:
            id_list = request.form.get('id_list').split(',')
            zoom = request.form.get('zoom')
            analysis_type = 'expanded' if int(zoom) > 0 else 'detailed'
        elif 'key' in request.form and 'value' in request.form:
            key = request.form.get('key')
            value = request.form.get('value')
        else:
            raise Exception("Request contains no parameters")

        sort = request.form.get('sort')
        amount = request.form.get('amount')

        if len(terms) > 0 or id_list:
            # Submit Celery task for topic analysis
            job = analyze_topic_async.delay(source, terms=terms, id_list=id_list,
                                            zoom=int(zoom), sort=sort, amount=amount)
            return redirect(url_for('.process', terms=terms, analysis_type=analysis_type, jobid=job.id))
        elif len(value) > 0:
            # Submit Celery task for paper analysis
            job = find_paper_async.delay(source, key, value)
            return redirect(url_for('.process', source=source, key=key, value=value, jobid=job.id))

    return render_template('main.html', version=PUBTRENDS_CONFIG.version,
                           amounts=PUBTRENDS_CONFIG.show_max_articles_options,
                           default_amount=PUBTRENDS_CONFIG.show_max_articles_default_value)


def get_app():
    return app

# # With debug=True, Flask server will auto-reload on changes
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True, extra_files=['templates/'])
