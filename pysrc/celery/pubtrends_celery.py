import os

from celery import Celery

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379'),
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379')

# Configure Celery Application
pubtrends_celery = Celery('pubtrends', backend=CELERY_RESULT_BACKEND, broker=CELERY_BROKER_URL)