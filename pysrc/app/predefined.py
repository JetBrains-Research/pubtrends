import gzip
import json
import logging
import os
from threading import Lock

from celery.result import AsyncResult

from pysrc.papers.analysis.text import preprocess_text
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


def save_predefined(viz, data, log, jobid, source, query, sort, limit):
    if jobid.startswith('predefined_'):
        logger.info('Saving predefined search')
        folder = os.path.join(predefined_path, f"{VERSION.replace(' ', '_')}",
                              query_to_folder(source, query, sort, limit))
        if not os.path.exists(folder):
            os.makedirs(folder)
        try:
            PREDEFINED_LOCK.acquire()
            path_viz = os.path.join(folder, 'viz.json.gz')
            path_data = os.path.join(folder, 'data.json.gz')
            path_log = os.path.join(folder, 'log.gz')
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


def load_predefined_viz_log(source, query, sort, limit, jobid):
    if jobid.startswith('predefined_'):
        logger.info('Trying to load predefined viz, log')
        folder = os.path.join(predefined_path, f"{VERSION.replace(' ', '_')}",
                              query_to_folder(source, query, sort, limit))
        path_viz = os.path.join(folder, 'viz.json.gz')
        path_log = os.path.join(folder, 'log.gz')
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


def load_predefined_or_result_data(source, query, sort, limit, jobid, app):
    if jobid.startswith('predefined_'):
        logger.info('Trying to load predefined data')
        folder = os.path.join(predefined_path, f"{VERSION.replace(' ', '_')}",
                              query_to_folder(source, query, sort, limit))
        path_data = os.path.join(folder, 'data.json.gz')
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


def query_to_folder(source, query, sort, limit, max_folder_length=100):
    folder_name = preprocess_text(f'{source}_{query}_{sort}_{limit}').replace(' ', '_').replace('__', '_')
    if len(folder_name) > max_folder_length:
        folder_name = folder_name[:(max_folder_length - 32 - 1)] + '_' + \
            hashlib.md5(folder_name.encode('utf-8')).hexdigest()
    return folder_name
