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

FILES_LOCK = Lock()

# Configure results path
RESULTS_PATH = ['/results', os.path.expanduser('~/.pubtrends/results')]
for path in RESULTS_PATH:
    if os.path.exists(path):
        logger.info(f'Search results will be stored at {path}')
        results_path = path
        break
else:
    raise RuntimeError(f'Search results folder not found among: {RESULTS_PATH}')


def get_results_path():
    return results_path


PREDEFINED_PREFIX = 'predefined-'


def get_predefined_jobs(config):
    result = {}
    if config.pm_enabled:
        result['Pubmed'] = _predefined_examples_jobid(config.pm_search_example_terms)
    if config.ss_enabled:
        result['Semantic Scholar'] = _predefined_examples_jobid(config.ss_search_example_terms)
    return result


def _save_data(viz, data, log, jobid, source, query, sort, limit):
    logger.info(f'Saving search for source={source} query={query} sort={sort} limit={limit}')
    folder = os.path.join(get_results_path(), preprocess_string(VERSION),
                          query_to_folder(source, query, sort, limit, jobid))
    if not os.path.exists(folder):
        os.makedirs(folder)
    try:
        FILES_LOCK.acquire()
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
        FILES_LOCK.release()


def load_result_viz_log(jobid, source, query, sort, limit, app):
    logger.info(f'Trying to load viz, log for source={source} query={query} sort={sort} limit={limit}')
    folder = os.path.join(get_results_path(), preprocess_string(VERSION),
                          query_to_folder(source, query, sort, limit, jobid))
    path_viz = os.path.join(folder, 'viz.json.gz')
    path_log = os.path.join(folder, 'log.gz')
    try:
        FILES_LOCK.acquire()
        if os.path.exists(path_viz) and os.path.exists(path_log):
            with gzip.open(path_viz, 'r') as f:
                viz = json.loads(f.read().decode('utf-8'))
            with gzip.open(path_log, 'r') as f:
                log = f.read().decode('utf-8')
            return viz, log
    finally:
        FILES_LOCK.release()
    job = AsyncResult(jobid, app=app)
    if job and job.state == 'SUCCESS':
        viz, data, log = job.result
        _save_data(viz, data, log, jobid, source, query, sort, limit)
        return viz, log
    return None, None


def load_result_data(jobid, source, query, sort, limit, app):
    logger.info(f'Trying to load data for source={source} query={query} sort={sort} limit={limit}')
    folder = os.path.join(get_results_path(), preprocess_string(VERSION),
                          query_to_folder(source, query, sort, limit, jobid))
    path_data = os.path.join(folder, 'data.json.gz')
    try:
        FILES_LOCK.acquire()
        if os.path.exists(path_data):
            with gzip.open(path_data, 'r') as f:
                return json.loads(f.read().decode('utf-8'))
    finally:
        FILES_LOCK.release()
    job = AsyncResult(jobid, app=app)
    if job and job.state == 'SUCCESS':
        viz, data, log = job.result
        _save_data(viz, data, log, jobid, source, query, sort, limit)
        return data
    return None


def preprocess_string(s):
    return re.sub(r'-{2,}', '-', re.sub(r'[^a-z0-9_]+', '-', s.lower())).strip('-')


def query_to_folder(source, query, sort, limit, jobid, max_folder_length=100):
    folder_name = preprocess_string(f'{source}-{query}-{sort}-{limit}')
    if len(folder_name) > max_folder_length:
        folder_name = folder_name[:(max_folder_length - 32 - 1)]
    if jobid is not None:
        folder_name += '-' + jobid[:32]
    else:
        folder_name += '-' + hashlib.md5(folder_name.encode('utf-8')).hexdigest()[:32]
    return folder_name


def _predefined_examples_jobid(examples):

    return [(t, PREDEFINED_PREFIX + hashlib.md5(t.encode('utf-8')).hexdigest()) for t in examples]


def _predefined_example_params_by_jobid(source, jobid, predefined_jobs):
    logger.debug(f'Lookup search example by jobid: {jobid}')
    for (t, predefined_jobid) in predefined_jobs[source]:
        if jobid == predefined_jobid:
            logger.debug(f'Example={t}, jobid={jobid}')
            return t, SORT_MOST_CITED, 1000
    raise Exception(f'Cannot find search example for jobid: {jobid}')

