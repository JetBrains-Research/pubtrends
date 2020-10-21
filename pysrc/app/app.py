import html
import json
import logging
import os
import random
import re
from threading import Lock
from urllib.parse import quote

import flask_admin
from celery.result import AsyncResult
from flask import Flask, url_for, redirect, render_template, request, abort, render_template_string
from flask_admin import helpers as admin_helpers, expose, BaseView
from flask_security import Security, SQLAlchemyUserDatastore, \
    UserMixin, RoleMixin, current_user
from flask_security.utils import hash_password
from flask_sqlalchemy import SQLAlchemy

from pysrc.celery.tasks import celery, find_paper_async, analyze_search_terms, analyze_id_list
from pysrc.celery.tasks_cache import get_or_cancel_task
from pysrc.papers.analyzer import KeyPaperAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.paper import prepare_paper_data, prepare_papers_data
from pysrc.papers.plot.plot_preprocessor import PlotPreprocessor
from pysrc.papers.plot.plotter import Plotter
from pysrc.papers.stats import prepare_stats_data
from pysrc.papers.utils import zoom_name, PAPER_ANALYSIS, ZOOM_IN_TITLE, PAPER_ANALYSIS_TITLE, trim, ZOOM_OUT_TITLE
from pysrc.version import VERSION

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

MAX_QUERY_LENGTH = 60

SOMETHING_WENT_WRONG = 'Something went wrong, please <a href="/">rerun</a> your search.'

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


def log_request(r):
    return f'addr:{r.remote_addr} args:{json.dumps(r.args)}'


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
            return json.dumps({
                'state': job_state,
                'message': html.escape(str(job_result).replace('\\n', '\n').replace('\\t', '\t')),
                'search_error': isinstance(job_result, SearchError)
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
    query = html.unescape(request.args.get('query'))
    source = request.args.get('source')
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    if jobid and query and source and limit is not None and sort is not None:
        job = AsyncResult(jobid, app=celery)
        if job and job.state == 'SUCCESS':
            data, _, log = job.result
            logger.info(f'/result success {log_request(request)}')
            return render_template('result.html',
                                   query=trim(query, MAX_QUERY_LENGTH),
                                   source=source,
                                   limit=limit,
                                   sort=sort,
                                   version=VERSION,
                                   log=log,
                                   **data)
        logger.info(f'/result No job or out-of-date job, restart it {log_request(request)}')
        return search_terms_(request.args)
    else:
        logger.error(f'/result error wrong request {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG), 400


@app.route('/process')
def process():
    if len(request.args) > 0:
        jobid = request.values.get('jobid')

        if not jobid:
            logger.error(f'/process error wrong request {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG)

        query = html.unescape(request.args.get('query') or '')
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
                                                  'limit': '', 'sort': ''},
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
        elif query:
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
    return render_template_string(SOMETHING_WENT_WRONG), 400


@app.route('/process_paper')
def process_paper():
    jobid = request.values.get('jobid')
    source = request.values.get('source')
    query = html.unescape(request.values.get('query'))
    if jobid:
        job = get_or_cancel_task(jobid)
        if job and job.state == 'SUCCESS':
            id_list = job.result
            logger.info(f'/process_paper single paper analysis {log_request(request)}')
            job = analyze_id_list.delay(source, ids=id_list, zoom=PAPER_ANALYSIS, query=query)
            return redirect(url_for('.process', query=query, analysis_type=PAPER_ANALYSIS_TITLE,
                                    id=id_list[0], source=source, jobid=job.id))


@app.route('/paper')
def paper():
    jobid = request.values.get('jobid')
    source = request.args.get('source')
    pid = request.args.get('id')
    if jobid:
        job = AsyncResult(jobid, app=celery)
        if job and job.state == 'SUCCESS':
            _, data, _ = job.result
            logger.info(f'/paper success {log_request(request)}')
            return render_template('paper.html', **prepare_paper_data(data, source, pid),
                                   version=VERSION)
        logger.error(f'/paper error jobid {log_request(request)}')
        return render_template_string("Out-of-date search, please search again..."), 400
    else:
        logger.error(f'/paper error wrong request {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG), 400


@app.route('/graph')
def graph():
    jobid = request.values.get('jobid')
    query = html.unescape(request.args.get('query'))
    source = request.args.get('source')
    limit = request.args.get('limit')
    sort = request.args.get('sort')
    graph_type = request.args.get('type')
    if jobid:
        job = AsyncResult(jobid, app=celery)
        if job and job.state == 'SUCCESS':
            _, data, _ = job.result
            loader, url_prefix = Loaders.get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
            analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
            analyzer.init(data)
            topics_tags = {comp: ', '.join(
                [w[0] for w in analyzer.df_kwd[analyzer.df_kwd['comp'] == comp]['kwd'].values[0][:10]]
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
                    topic_other=analyzer.comp_other or -1,
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
                    topic_other=analyzer.comp_other or -1,
                    topics_palette_json=json.dumps(Plotter.topics_palette(analyzer.df)),
                    topics_description_json=json.dumps(topics_tags),
                    graph_cytoscape_json=json.dumps(graph_cs)
                )
        logger.error(f'/graph error job id {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG), 400
    else:
        logger.error(f'/graph error wrong request {log_request(request)}')
        return render_template_string(SOMETHING_WENT_WRONG), 400


@app.route('/papers')
def show_ids():
    jobid = request.values.get('jobid')
    query = html.unescape(request.args.get('query'))
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
        job = AsyncResult(jobid, app=celery)
        if job and job.state == 'SUCCESS':
            _, data, _ = job.result
            logger.info(f'/papers success {log_request(request)}')
            export_name = re.sub('_{2,}', '_', re.sub('["\':,. ]', '_', f'{query}_{search_string}'.lower().strip('_')))
            return render_template('papers.html',
                                   version=VERSION,
                                   source=source,
                                   query=trim(query, MAX_QUERY_LENGTH),
                                   search_string=search_string,
                                   limit=limit,
                                   sort=sort,
                                   export_name=export_name,
                                   papers=prepare_papers_data(data, source, comp, word, author, journal, papers_list))
    logger.error(f'/papers error {log_request(request)}')
    return render_template_string(SOMETHING_WENT_WRONG), 400


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
        'message': 'Unknown task id'
    })


# Index page
@app.route('/')
def index():
    logger.info(f'/ {log_request(request)}')

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
    if PUBTRENDS_CONFIG.min_search_words > 1:
        min_words_message = f' Minimum {PUBTRENDS_CONFIG.min_search_words} words.'
    else:
        min_words_message = ''

    return render_template('main.html',
                           version=VERSION,
                           limits=PUBTRENDS_CONFIG.show_max_articles_options,
                           default_limit=PUBTRENDS_CONFIG.show_max_articles_default_value,
                           min_words_message=min_words_message,
                           pm_enabled=PUBTRENDS_CONFIG.pm_enabled,
                           ss_enabled=PUBTRENDS_CONFIG.ss_enabled,
                           search_example_source=search_example_source,
                           search_example_terms=search_example_terms)


@app.route('/search_terms', methods=['POST'])
def search_terms():
    return search_terms_(request.form)


def search_terms_(data):
    query = html.unescape(data.get('query'))  # Original search query
    source = data.get('source')  # Pubmed or Semantic Scholar
    sort = data.get('sort')  # Sort order
    limit = data.get('limit')  # Limit
    jobid = data.get('jobid')  # Optional job id
    noreviews = data.get('noreviews')  # Include reviews in the initial search phase
    expand = data.get('expand')  # Fraction of papers to cover by references
    try:
        if query and source and sort and limit and noreviews is not None and expand:
            if not jobid:
                logger.info(f'/search_terms {log_request(request)}')
                job = analyze_search_terms.delay(source, query=query, limit=int(limit), sort=sort,
                                                 noreviews=noreviews == 'on', expand=int(expand) / 100)
                jobid = job.id
            else:
                logger.info(f'/search_terms with fixed jobid {log_request(request)}')
                analyze_search_terms.apply_async(args=[source, query, sort, int(limit), noreviews, int(expand) / 100],
                                                 task_id=jobid)

            return redirect(url_for('.process', query=query, source=source, limit=limit, sort=sort,
                                    noreviews=noreviews, expand=expand,
                                    jobid=jobid))
    except Exception as e:
        logger.error(f'/search_terms error', e)
        return render_template_string("Error occurred. We're working on it. Please check back soon."), 500
    logger.error(f'/search_terms error {log_request(request)}')
    return render_template_string(SOMETHING_WENT_WRONG), 400


@app.route('/search_paper', methods=['POST'])
def search_paper():
    logger.info('/search_paper')
    source = request.form.get('source')  # Pubmed or Semantic Scholar

    try:
        if source and 'key' in request.form and 'value' in request.form:
            logger.info(f'/search_paper {log_request(request)}')
            key = request.form.get('key')
            value = request.form.get('value')
            job = find_paper_async.delay(source, key, value)
            return redirect(url_for('.process', source=source, key=key, value=value, jobid=job.id))
    except Exception as e:
        logger.error(f'/search_paper error', e)
        return render_template_string("Error occurred. We're working on it. Please check back soon."), 500
    logger.error(f'/search_paper error {log_request(request)}')
    return render_template_string(SOMETHING_WENT_WRONG), 400


@app.route('/process_ids', methods=['POST'])
def process_ids():
    source = request.form.get('source')  # Pubmed or Semantic Scholar
    query = html.unescape(request.form.get('query'))  # Original search query

    try:
        if source and query and 'id_list' in request.form:
            id_list = request.form.get('id_list').split(',')
            zoom = request.form.get('zoom')
            analysis_type = zoom_name(zoom)
            job = analyze_id_list.delay(source, ids=id_list, zoom=int(zoom), query=query)
            logger.info(f'/process_ids {log_request(request)}')
            return redirect(url_for('.process', query=query, analysis_type=analysis_type, source=source, jobid=job.id))
    except Exception as e:
        logger.error(f'/process_ids error', e)
        return render_template_string("Error occurred. We're working on it. Please check back soon."), 500
    logger.error(f'/process_ids error {log_request(request)}')
    return render_template_string(SOMETHING_WENT_WRONG), 400


#######################
# Feedback functionality #
#######################

@app.route('/feedback', methods=['POST'])
def feedback():
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
        return render_template_string("Error occurred. We're working on it. Please check back soon."), 500
    return render_template_string('Thanks you for the feedback!'), 200


#######################
# Admin functionality #
#######################

# Configure flask-admin and flask-sqlalchemy
app.config.from_pyfile('config.py')

# Deployment and development
DATABASE_PATHS = ['/database', os.path.expanduser('~/.pubtrends/database')]
for p in DATABASE_PATHS:
    if os.path.isdir(p):
        DATABASE_PATH = os.path.join(p, app.config['DATABASE_FILE'])
        break
else:
    raise RuntimeError('Failed to configure service db path')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + DATABASE_PATH
db = SQLAlchemy(app)

# Define models
roles_users = db.Table(
    'roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id'))
)


class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))

    def __str__(self):
        return self.name


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(255))
    last_name = db.Column(db.String(255))
    email = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))
    active = db.Column(db.Boolean())
    confirmed_at = db.Column(db.DateTime())
    roles = db.relationship('Role', secondary=roles_users,
                            backref=db.backref('users', lazy='dynamic'))

    def __str__(self):
        return self.email


# Setup Flask-Security
user_datastore = SQLAlchemyUserDatastore(db, User, Role)
security = Security(app, user_datastore)


class AdminStatsView(BaseView):
    @expose('/')
    def index(self):
        stats_data = prepare_stats_data(logfile)
        return self.render('admin/stats.html', version=VERSION, **stats_data)

    def is_accessible(self):
        return current_user.is_active and current_user.is_authenticated and current_user.has_role('admin')

    def _handle_view(self, name, **kwargs):
        """
        Override builtin _handle_view in order to redirect users when a view is not accessible.
        """
        if not self.is_accessible():
            if current_user.is_authenticated:
                # permission denied
                abort(403)
            else:
                # login
                return redirect(url_for('security.login', next=request.url))


# Create admin
admin = flask_admin.Admin(
    app,
    'Pubtrends',
    base_template='master.html',
    template_mode='bootstrap3',
)

# Show only stats view
admin.add_view(AdminStatsView(name='Statistics', endpoint='stats'))


# define a context processor for merging flask-admin's template context into the
# flask-security views.
@security.context_processor
def security_context_processor():
    return dict(
        admin_base_template=admin.base_template,
        admin_view=admin.index_view,
        h=admin_helpers,
        get_url=url_for
    )


def build_users_db():
    """
    Populate a small db with some example entries.
    """
    db.drop_all()
    db.create_all()

    with app.app_context():
        user_role = Role(name='user')
        admin_role = Role(name='admin')
        db.session.add(user_role)
        db.session.add(admin_role)
        db.session.commit()

        user_datastore.create_user(
            first_name='Admin',
            email='admin',
            password=hash_password(PUBTRENDS_CONFIG.admin_password),
            roles=[user_role, admin_role]
        )
        db.session.commit()
    return


def get_app():
    return app


# Build a sample db on the fly, if one does not exist yet.
DB_LOCK = Lock()

DB_LOCK.acquire()
if not os.path.exists(DATABASE_PATH):
    build_users_db()
DB_LOCK.release()

# With debug=True, Flask server will auto-reload on changes
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, extra_files=['templates/'])
