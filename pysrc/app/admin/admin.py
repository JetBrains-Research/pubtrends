import os
import uuid
from multiprocessing import Lock

from flask import abort, url_for, request, redirect
from flask_admin import helpers as admin_helpers, expose, BaseView, Admin
from flask_security import Security, current_user
from flask_security.utils import hash_password

from pysrc.app.admin.celery import prepare_celery_data
from pysrc.app.admin.feedback import prepare_feedback_data
from pysrc.app.admin.forms import LoginForm
from pysrc.app.admin.stats import prepare_stats_data
from pysrc.app.db.models import db, Role, User, init_db, get_user_datastore
from pysrc.config import PubtrendsConfig
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


def configure_admin_functions(app, celery_app, logfile):
    """Configures flask-admin and flask-sqlalchemy"""
    with app.app_context():
        app.config.from_pyfile('config.py')
        service_database_path = os.path.join(SERVICE_DATABASE_PATH, app.config['DATABASE_FILE'])
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + service_database_path

        # Bind shared db instance
        init_db(app)

        # Setup Flask-Security using shared models
        user_datastore = get_user_datastore()
        security = Security(app, user_datastore, login_form=LoginForm)

        # Ensure schema exists and seed roles/admin user idempotently.
        try:
            DB_LOCK.acquire()
            db.create_all()

            # Ensure roles
            user_role = Role.query.filter_by(name='user').first()
            if not user_role:
                user_role = Role(name='user')
                db.session.add(user_role)

            admin_role = Role.query.filter_by(name='admin').first()
            if not admin_role:
                admin_role = Role(name='admin')
                db.session.add(admin_role)
            db.session.commit()

            # Ensure admin user exists/updated
            admin_email = PUBTRENDS_CONFIG.admin_email
            admin_password = PUBTRENDS_CONFIG.admin_password
            if admin_email:
                admin_user = User.query.filter_by(email=admin_email).first()
                if not admin_user:
                    user_datastore.create_user(
                        first_name='Admin',
                        email=admin_email,
                        fs_uniquifier=str(uuid.uuid4()),
                        password=hash_password(admin_password) if admin_password else None,
                        roles=[user_role, admin_role]
                    )
                    db.session.commit()
                else:
                    if admin_role not in admin_user.roles:
                        admin_user.roles.append(admin_role)
                        db.session.commit()
                    if admin_password and os.getenv('PUBTRENDS_ADMIN_PASSWORD_RESET', '0') in ('1', 'true', 'True'):
                        admin_user.password = hash_password(admin_password)
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

        class AdminCeleryView(BaseView):
            @expose('/')
            def index(self):
                celery_data = prepare_celery_data(celery_app)
                return self.render('admin/celery.html', version=VERSION, **celery_data)

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
            'PubTrends',
            base_template='master.html',
            template_mode='bootstrap5',
        )

        # Available views
        admin.add_view(AdminStatsView(name='Statistics', endpoint='stats'))
        admin.add_view(AdminFeedbackView(name='Feedback', endpoint='feedback'))
        admin.add_view(AdminCeleryView(name='Celery', endpoint='celery'))

        # define a context processor for merging flask-admin's template context into the flask-security views.
        @security.context_processor
        def security_context_processor():
            return dict(
                admin_base_template=admin.base_template,
                admin_view=admin.index_view,
                h=admin_helpers,
                get_url=url_for
            )


def init_admin(app, celery_app, logfile):
    """Initialize Admin blueprint: DB, Security, seed data."""
    with app.app_context():
        # Load admin-specific config
        app.config.from_pyfile('config.py')

        # Configure DB path
        service_db_base_paths = ['/database', os.path.expanduser('~/.pubtrends/database')]
        for p in service_db_base_paths:
            if os.path.isdir(p):
                base_path = p
                break
        else:
            raise RuntimeError('Failed to configure service db path')

        service_database_path = os.path.join(base_path, app.config['DATABASE_FILE'])
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + service_database_path
        app.config.setdefault('SQLALCHEMY_TRACK_MODIFICATIONS', False)

        # Init db via shared module
        init_db(app)

        # Setup Flask-Security
        user_datastore = get_user_datastore()
        Security(app, user_datastore, login_form=LoginForm)

        # Ensure schema exists and seed roles/admin user idempotently.
        try:
            DB_LOCK.acquire()
            # Always ensure tables exist (no-op if already created)
            db.create_all()

            # Ensure roles exist
            user_role = Role.query.filter_by(name='user').first()
            if not user_role:
                user_role = Role(name='user')
                db.session.add(user_role)

            admin_role = Role.query.filter_by(name='admin').first()
            if not admin_role:
                admin_role = Role(name='admin')
                db.session.add(admin_role)

            db.session.commit()

            # Ensure admin user exists
            admin_email = PUBTRENDS_CONFIG.admin_email
            admin_password = PUBTRENDS_CONFIG.admin_password

            if admin_email:
                admin_user = User.query.filter_by(email=admin_email).first()
                if not admin_user:
                    user_datastore.create_user(
                        first_name='Admin',
                        email=admin_email,
                        fs_uniquifier=str(uuid.uuid4()),
                        password=hash_password(admin_password) if admin_password else None,
                        roles=[user_role, admin_role]
                    )
                    db.session.commit()
                else:
                    # Ensure admin has proper roles
                    if admin_role not in admin_user.roles:
                        admin_user.roles.append(admin_role)
                        db.session.commit()

                    # Optionally rotate password if provided via env flag
                    if admin_password and os.getenv('PUBTRENDS_ADMIN_PASSWORD_RESET', '0') in ('1', 'true', 'True'):
                        admin_user.password = hash_password(admin_password)
                        db.session.commit()
            # If no admin email configured – do nothing but keep tables created
        finally:
            DB_LOCK.release()

        # Stash references for handlers
        app.config['ADMIN_LOGFILE'] = logfile
        app.config['ADMIN_CELERY_APP'] = celery_app
