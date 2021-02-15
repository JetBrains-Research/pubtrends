import gzip
import json
import logging
import os
from threading import Lock

from celery.result import AsyncResult

from pysrc.version import VERSION

logger = logging.getLogger(__name__)

# Configure predefined paths
PREDEFINED_PATHS = ['/predefined', os.path.expanduser('~/.pubtrends/predefined')]
for p in PREDEFINED_PATHS:
    if os.path.isdir(p):
        predefined_path = p
        break
else:
    raise RuntimeError('Failed to configure predefined searches dir')
PREDEFINED_LOCK = Lock()


def save_predefined(viz, data, log, jobid):
    if jobid.startswith('predefined_'):
        logger.info('Saving predefined search')
        path = os.path.join(predefined_path, f"{VERSION.replace(' ', '_')}_{jobid}")
        try:
            PREDEFINED_LOCK.acquire()
            path_viz = f'{path}_viz.json.gz'
            path_data = f'{path}_data.json.gz'
            path_log = f'{path}_log.gz'
            if not os.path.exists(path_viz):
                with gzip.open(path_viz, 'w') as f:
                    f.write(json.dumps(viz).encode('utf-8'))
            if not os.path.exists(path_data):
                with gzip.open(path_data, 'w') as f:
                    f.write(json.dumps(data).encode('utf-8'))
            if not os.path.exists(path_log):
                with gzip.open(path_log, 'w') as f:
                    f.write(log.encode('utf-8'))
        finally:
            PREDEFINED_LOCK.release()


def load_predefined_viz_log(jobid):
    if jobid.startswith('predefined_'):
        logger.info('Trying to load predefined viz, log')
        path = os.path.join(predefined_path, f"{VERSION.replace(' ', '_')}_{jobid}")
        path_viz = f'{path}_viz.json.gz'
        path_log = f'{path}_log.gz'
        try:
            PREDEFINED_LOCK.acquire()
            if os.path.exists(path_viz) and os.path.exists(path_log):
                with gzip.open(path_viz, 'r') as f:
                    viz = json.loads(f.read().decode('utf-8'))
                with gzip.open(path_log, 'r') as f:
                    log = f.read().decode('utf-8')
                return viz, log
        finally:
            PREDEFINED_LOCK.release()
    return None


def load_predefined_or_result_data(jobid, app):
    if jobid.startswith('predefined_'):
        logger.info('Trying to load predefined data')
        path = os.path.join(predefined_path, f"{VERSION.replace(' ', '_')}_{jobid}")
        path_data = f'{path}_data.json.gz'
        try:
            PREDEFINED_LOCK.acquire()
            if os.path.exists(path_data):
                with gzip.open(path_data, 'r') as f:
                    return json.loads(f.read().decode('utf-8'))
        finally:
            PREDEFINED_LOCK.release()
    job = AsyncResult(jobid, app=app)
    if job and job.state == 'SUCCESS':
        viz, data, log = job.result
        return data
    return None

