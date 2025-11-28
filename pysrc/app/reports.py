import gzip
import hashlib
import json
import logging
import os
import re
from threading import Lock

from celery.result import AsyncResult

from pysrc.papers.utils import IDS_ANALYSIS_TYPE, PAPER_ANALYSIS_TYPE
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


def _save_data(name: str, data_json: dict, log: str):
    folder = os.path.join(results_path, preprocess_string(VERSION), name)
    try:
        FILES_LOCK.acquire()
        if not os.path.exists(folder):
            os.makedirs(folder)
        path_data = os.path.join(folder, 'data.json.gz')
        path_log = os.path.join(folder, 'log.gz')
        if not os.path.exists(path_data):
            with gzip.open(path_data, 'w') as f:
                f.write(json.dumps(data_json).encode('utf-8'))
        if not os.path.exists(path_log):
            with gzip.open(path_log, 'w') as f:
                f.write(log.encode('utf-8'))
    finally:
        FILES_LOCK.release()

def is_data_saved(jobid: str) -> bool:
    try:
        FILES_LOCK.acquire()
        return _data_path(jobid) is not None
    finally:
        FILES_LOCK.release()

def _data_path(jobid: str) -> str | None:
    folder = _find_data_folder(jobid)
    if folder is None:
        return None
    path_data = os.path.join(folder, 'data.json.gz')
    return path_data if os.path.exists(path_data) else None


def load_or_save_result_data(celery_app, jobid) -> AnalysisData | None:
    logger.info(f'Trying to load data for job_id={jobid}')
    try:
        FILES_LOCK.acquire()
        p = _data_path(jobid)
        if p is not None:
            with gzip.open(p, 'r') as f:
                return AnalysisData.from_json(json.loads(f.read().decode('utf-8')))
    finally:
        FILES_LOCK.release()
    job = AsyncResult(jobid, app=celery_app)
    if job and job.state == 'SUCCESS':
        data_json, log = job.result
        analysis_type = data_json['analysis_type']
        if analysis_type == IDS_ANALYSIS_TYPE:
            _save_data(result_folder_name(jobid, data_json), data_json, log)
        elif analysis_type == PAPER_ANALYSIS_TYPE:
            _save_data(paper_folder_name(jobid, data_json), data_json, log)
        else:
            raise Exception(f'Unknown analysis type {jobid} {analysis_type}')
        return AnalysisData.from_json(data_json)
    return None


def preprocess_string(s):
    return re.sub(r'-{2,}', '-', re.sub(r'[^a-z0-9_]+', '-', s.lower())).strip('-')


def result_folder_name(jobid: str, data: dict) -> str:
    fields = ['source', 'search_query', 'sort', 'limit', 'noreviews', 'min_year', 'max_year']
    text = '-'.join(map(str, filter(None, map(lambda x: data[x], fields))))
    return get_folder_name(jobid, text)


def paper_folder_name(jobid: str, data: dict):
    return get_folder_name(jobid, f"paper-{data['source']}-{data['search_query']}")


def get_folder_name(jobid, name, max_folder_length=100):
    folder_name = preprocess_string(name)
    if len(folder_name) > max_folder_length:
        folder_name = folder_name[:(max_folder_length - 32 - 1)]
    if jobid is not None:
        folder_name = jobid[:32] + '-' + folder_name
    else:
        folder_name = hashlib.md5(folder_name.encode('utf-8')).hexdigest()[:32] + '-' + folder_name
    return folder_name


def _find_data_folder(jobid):
    lookup_folder = os.path.join(results_path, preprocess_string(VERSION))
    if not os.path.exists(lookup_folder):
        os.makedirs(lookup_folder)
    for p in os.listdir(lookup_folder):
        candidate_folder = os.path.join(lookup_folder, p)
        if p.startswith(jobid[:32]) and os.path.isdir(candidate_folder):
            return candidate_folder
    return None
