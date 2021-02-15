from multiprocessing import Lock

from flask import abort, url_for, request, redirect
from flask_admin import helpers as admin_helpers, expose, BaseView, Admin
from flask_security import Security, SQLAlchemyUserDatastore, \
    UserMixin, RoleMixin, current_user
from flask_security.utils import hash_password
from flask_sqlalchemy import SQLAlchemy
import os

from pysrc.app.admin.feedback import prepare_feedback_data
from pysrc.app.admin.stats import prepare_stats_data
from pysrc.papers.pubtrends_config import PubtrendsConfig
from pysrc.version import VERSION


PUBTRENDS_CONFIG = PubtrendsConfig(test=False)
DB_LOCK = Lock()

# Admin service database path
SERVICE_DATABASE_PATHS = ['/database', os.path.expanduser('~/.pubtrends/database')]
for p in SERVICE_DATABASE_PATHS:
    if os.path.isdir(p):
        SERVICE_DATABASE_PATH = p
        break
else:
    raise RuntimeError('Failed to configure service db path')


def configure_admin_functions(app, logfile):
    """Configures flask-admin and flask-sqlalchemy"""
    app.config.from_pyfile('config.py')
    service_database_path = os.path.join(SERVICE_DATABASE_PATH, app.config['DATABASE_FILE'])
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + service_database_path

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

    # Build a sample db on the fly, if one does not exist yet.
    try:
        DB_LOCK.acquire()
        if not os.path.exists(service_database_path):
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
    finally:
        DB_LOCK.release()

    # UI
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

    class AdminFeedbackView(BaseView):
        @expose('/')
        def index(self):
            feedback_data = prepare_feedback_data(logfile)
            return self.render('admin/feedback.html', version=VERSION, **feedback_data)

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
    admin = Admin(
        app,
        'Pubtrends',
        base_template='master.html',
        template_mode='bootstrap3',
    )

    # Available views
    admin.add_view(AdminStatsView(name='Statistics', endpoint='stats'))
    admin.add_view(AdminFeedbackView(name='Feedback', endpoint='feedback'))

    # define a context processor for merging flask-admin's template context into the flask-security views.
    @security.context_processor
    def security_context_processor():
        return dict(
            admin_base_template=admin.base_template,
            admin_view=admin.index_view,
            h=admin_helpers,
            get_url=url_for
        )
