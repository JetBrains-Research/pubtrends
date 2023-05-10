import gzip
import hashlib
import json
import logging
import os
import re
from threading import Lock

from celery.result import AsyncResult

from pysrc.papers.utils import SORT_MOST_CITED
from pysrc.version import VERSION

logger = logging.getLogger(__name__)

# Configure predefined path
PREDEFINED_PATHS = ['/predefined', os.path.expanduser('~/.pubtrends/predefined')]
for p in PREDEFINED_PATHS:
    if os.path.isdir(p):
        predefined_path = p
        break
else:
    raise RuntimeError('Failed to configure predefined searches dir')
PREDEFINED_LOCK = Lock()


def get_predefined_path():
    return predefined_path


# Configure search results path
RESULTS_PATH = ['/results', os.path.expanduser('~/.pubtrends/results')]
for path in RESULTS_PATH:
    if os.path.exists(path):
        logger.info(f'Search results will be stored at {path}')
        search_path = path
        break
else:
    raise RuntimeError(f'Search results folder not found among: {RESULTS_PATH}')


def search_results_folder():
    return search_path


def get_predefined_jobs(config):
    result = {}
    if config.pm_enabled:
        result['Pubmed'] = _predefined_info(config.pm_search_example_terms)
    if config.ss_enabled:
        result['Semantic Scholar'] = _predefined_info(config.ss_search_example_terms)
    return result


def _save_predefined(viz, data, log, source, jobid, predefined_jobs):
    query, sort, limit = _example_by_jobid(source, jobid, predefined_jobs)
    logger.info(f'Saving predefined search for source={source} query={query} sort={sort} limit={limit}')
    folder = os.path.join(get_predefined_path(), f"{VERSION.replace(' ', '_')}",
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


def load_predefined_viz_log(source, jobid, predefined_jobs, app):
    is_predefined = jobid.startswith('predefined_')
    if is_predefined:
        query, sort, limit = _example_by_jobid(source, jobid, predefined_jobs)
        logger.info(f'Trying to load predefined viz, log for source={source} query={query} sort={sort} limit={limit}')
        folder = os.path.join(get_predefined_path(), f"{VERSION.replace(' ', '_')}",
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
                return viz, log, True
        finally:
            PREDEFINED_LOCK.release()
    job = AsyncResult(jobid, app=app)
    if job and job.state == 'SUCCESS':
        viz, data, log = job.result
        if is_predefined:
            _save_predefined(viz, data, log, source, jobid, predefined_jobs)
        return viz, log, is_predefined
    return None, None, is_predefined


def load_predefined_or_result_data(source, jobid, predefined_jobs, app):
    is_predefined = jobid.startswith('predefined_')
    if is_predefined:
        query, sort, limit = _example_by_jobid(source, jobid, predefined_jobs)
        logger.info(f'Trying to load predefined data for source={source} query={query} sort={sort} limit={limit}')
        folder = os.path.join(get_predefined_path(), f"{VERSION.replace(' ', '_')}",
                              query_to_folder(source, query, sort, limit))
        path_data = os.path.join(folder, 'data.json.gz')
        try:
            PREDEFINED_LOCK.acquire()
            if os.path.exists(path_data):
                with gzip.open(path_data, 'r') as f:
                    return json.loads(f.read().decode('utf-8')), True
        finally:
            PREDEFINED_LOCK.release()
    job = AsyncResult(jobid, app=app)
    if job and job.state == 'SUCCESS':
        viz, data, log = job.result
        if is_predefined:
            _save_predefined(viz, data, log, source, jobid, predefined_jobs)
        return data, is_predefined
    return None, is_predefined


def query_to_folder(source, query, sort, limit, max_folder_length=100):
    folder_name = re.sub(r'[^a-z0-9_]+', '_', f'{source}_{query}_{sort}_{limit}'.lower()).replace('__', '_')
    if len(folder_name) > max_folder_length:
        folder_name = folder_name[:(max_folder_length - 32 - 1)] + '_' + \
                      hashlib.md5(folder_name.encode('utf-8')).hexdigest()
    return folder_name


def _predefined_info(examples):
    return [(t, hashlib.md5(t.encode('utf-8')).hexdigest()) for t in examples]


def _example_by_jobid(source, jobid, predefined_jobs):
    logger.debug(f'Lookup search example by jobid: {jobid}')
    for (t, hash) in predefined_jobs[source]:
        if jobid == f'predefined_{hash}':
            logger.debug(f'Example: {t}')
            return t, SORT_MOST_CITED, 1000
    raise Exception(f'Cannot find search example for jobid: {jobid}')

