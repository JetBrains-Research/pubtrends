from functools import wraps
from multiprocessing import Lock

from flask import Blueprint, abort, current_app, render_template, request, redirect, url_for
from flask_security import current_user

from pysrc.app.admin.celery import prepare_celery_data
from pysrc.app.admin.feedback import prepare_feedback_data
from pysrc.app.admin.stats import prepare_stats_data
from pysrc.config import PubtrendsConfig
from pysrc.version import VERSION

# Blueprint instance
admin_bp = Blueprint('admin_bp', __name__, url_prefix='/admin')

# Global config/state
PUBTRENDS_CONFIG = PubtrendsConfig(test=False)
DB_LOCK = Lock()


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
