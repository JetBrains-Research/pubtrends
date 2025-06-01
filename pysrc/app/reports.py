import gzip
import hashlib
import json
import logging
import os
import re
from threading import Lock

from celery.result import AsyncResult

from pysrc.papers.data import AnalysisData
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


def _save_result_data(data, log, jobid, source, query, sort, limit, noreviews, min_year, max_year):
    _save_data(result_folder_name(source, query, sort, limit, noreviews, min_year, max_year, jobid), data, log)


def _save_paper_data(data, log, jobid, source, query):
    _save_data(paper_folder_name(source, query, jobid), data, log)


def _save_data(name, data, log):
    folder = os.path.join(get_results_path(), preprocess_string(VERSION), name)
    try:
        FILES_LOCK.acquire()
        if not os.path.exists(folder):
            os.makedirs(folder)
        path_data = os.path.join(folder, 'data.json.gz')
        path_log = os.path.join(folder, 'log.gz')
        if not os.path.exists(path_data):
            with gzip.open(path_data, 'w') as f:
                f.write(json.dumps(data).encode('utf-8'))
        if not os.path.exists(path_log):
            with gzip.open(path_log, 'w') as f:
                f.write(log.encode('utf-8'))
    finally:
        FILES_LOCK.release()


def load_result_data(jobid, source, query, sort, limit, noreviews, min_year, max_year,
                     celery_app) -> AnalysisData | None:
    logger.info(f'Trying to load data for source={source} query={query} sort={sort} limit={limit} '
                f'noreviews={noreviews} min_year={min_year} max_year={max_year}')
    try:
        FILES_LOCK.acquire()
        folder = _find_folder(jobid)
        if folder is not None:
            path_data = os.path.join(folder, 'data.json.gz')
            if os.path.exists(path_data):
                with gzip.open(path_data, 'r') as f:
                    return AnalysisData.from_json(json.loads(f.read().decode('utf-8')))
    finally:
        FILES_LOCK.release()
    job = AsyncResult(jobid, app=celery_app)
    if job and job.state == 'SUCCESS':
        data, log = job.result
        _save_result_data(data, log, jobid, source, query, sort, limit, noreviews, min_year, max_year)
        return AnalysisData.from_json(data)
    return None


def load_paper_data(jobid, source, query, celery_app) -> AnalysisData | None:
    logger.info(f'Trying to load paper data for source={source} query={query}')
    try:
        FILES_LOCK.acquire()
        folder = _find_folder(jobid)
        if folder is not None:
            path_data = os.path.join(folder, 'data.json.gz')
            if os.path.exists(path_data):
                with gzip.open(path_data, 'r') as f:
                    return AnalysisData.from_json(json.loads(f.read().decode('utf-8')))
    finally:
        FILES_LOCK.release()
    job = AsyncResult(jobid, app=celery_app)
    if job and job.state == 'SUCCESS':
        data, log = job.result
        _save_paper_data(data, log, jobid, source, query)
        return AnalysisData.from_json(data)
    return None


def preprocess_string(s):
    return re.sub(r'-{2,}', '-', re.sub(r'[^a-z0-9_]+', '-', s.lower())).strip('-')


def result_folder_name(source, query, sort, limit, noreviews, min_year, max_year, jobid):
    return get_folder_name(jobid, f'{source}-{query}-{sort}-{limit}-{noreviews}-{min_year}-{max_year}')


def paper_folder_name(source, query, jobid):
    return get_folder_name(jobid, f'paper-{source}-{query}')


def get_folder_name(jobid, name, max_folder_length=100):
    folder_name = preprocess_string(name)
    if len(folder_name) > max_folder_length:
        folder_name = folder_name[:(max_folder_length - 32 - 1)]
    if jobid is not None:
        folder_name = jobid[:32] + '-' + folder_name
    else:
        folder_name = hashlib.md5(folder_name.encode('utf-8')).hexdigest()[:32] + '-' + folder_name
    return folder_name


def _find_folder(jobid):
    lookup_folder = os.path.join(get_results_path(), preprocess_string(VERSION))
    if not os.path.exists(lookup_folder):
        os.makedirs(lookup_folder)
    for p in os.listdir(lookup_folder):
        candidate_folder = os.path.join(lookup_folder, p)
        if p.startswith(jobid[:32]) and os.path.isdir(candidate_folder):
            return candidate_folder
    return None
