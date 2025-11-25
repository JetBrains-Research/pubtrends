import os
from functools import wraps
from multiprocessing import Lock

from flask import Blueprint, abort, current_app, render_template, request, redirect, url_for
from flask_security import Security, SQLAlchemyUserDatastore, UserMixin, RoleMixin, current_user
from flask_security.utils import hash_password
from flask_sqlalchemy import SQLAlchemy

from pysrc.app.admin.celery import prepare_celery_data
from pysrc.app.admin.feedback import prepare_feedback_data
from pysrc.app.admin.stats import prepare_stats_data
from pysrc.app.admin.forms import LoginForm
from pysrc.config import PubtrendsConfig
from pysrc.version import VERSION

# Blueprint instance
admin_bp = Blueprint('admin_bp', __name__, url_prefix='/admin')

# Global config/state
PUBTRENDS_CONFIG = PubtrendsConfig(test=False)
DB_LOCK = Lock()

# Flask-SQLAlchemy instance (initialized in init_admin)
db = SQLAlchemy()


# Database models
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
    fs_uniquifier = db.Column(db.String(255), unique=True, nullable=False)
    first_name = db.Column(db.String(255))
    last_name = db.Column(db.String(255))
    email = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))
    active = db.Column(db.Boolean())
    confirmed_at = db.Column(db.DateTime())
    roles = db.relationship('Role', secondary=roles_users, backref=db.backref('users', lazy='dynamic'))

    def __str__(self):
        return self.email


def _admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not (current_user.is_active and current_user.is_authenticated and current_user.has_role('admin')):
            if current_user.is_authenticated:
                abort(403)
            else:
                return redirect(url_for('security.login', next=request.url))
        return fn(*args, **kwargs)

    return wrapper


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

        # Init db
        db.init_app(app)

        # Setup Flask-Security
        user_datastore = SQLAlchemyUserDatastore(db, User, Role)
        Security(app, user_datastore, login_form=LoginForm)

        # Build a sample db on the fly, if one does not exist yet.
        try:
            DB_LOCK.acquire()
            if not os.path.exists(service_database_path):
                db.drop_all()
                db.create_all()
                user_role = Role(name='user')
                admin_role = Role(name='admin')
                db.session.add(user_role)
                db.session.add(admin_role)
                db.session.commit()

                user_datastore.create_user(
                    first_name='Admin',
                    email=PUBTRENDS_CONFIG.admin_email,
                    fs_uniquifier='unique_admin_identifier',
                    password=hash_password(PUBTRENDS_CONFIG.admin_password),
                    roles=[user_role, admin_role]
                )
                db.session.commit()
        finally:
            DB_LOCK.release()

        # Stash references for handlers
        app.config['ADMIN_LOGFILE'] = logfile
        app.config['ADMIN_CELERY_APP'] = celery_app


@admin_bp.route('/')
@_admin_required
def admin_index():
    # Simple index with links; reuse existing template
    return render_template('admin/index.html', version=VERSION)


@admin_bp.route('/stats')
@_admin_required
def admin_stats():
    logfile = current_app.config.get('ADMIN_LOGFILE')
    stats_data = prepare_stats_data(logfile)
    return render_template('admin/stats.html', version=VERSION, **stats_data)


@admin_bp.route('/feedback')
@_admin_required
def admin_feedback():
    logfile = current_app.config.get('ADMIN_LOGFILE')
    feedback_data = prepare_feedback_data(logfile)
    return render_template('admin/feedback.html', version=VERSION, **feedback_data)


@admin_bp.route('/celery')
@_admin_required
def admin_celery():
    celery_app = current_app.config.get('ADMIN_CELERY_APP')
    celery_data = prepare_celery_data(celery_app)
    return render_template('admin/celery.html', version=VERSION, **celery_data)
