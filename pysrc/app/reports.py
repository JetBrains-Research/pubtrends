import logging
import re

from celery.result import AsyncResult

from pysrc.papers.data import AnalysisData
from pysrc.papers.redis_cache import get_cache

logger = logging.getLogger(__name__)

def is_data_saved(jobid: str) -> bool:
    """
    Check if analysis data is saved in the Redis cache.

    Args:
        jobid: Job identifier

    Returns:
        True if data exists in Redis cache, False otherwise
    """
    try:
        cache = get_cache()
        return cache.exists(jobid)
    except Exception as e:
        logger.exception(f'Error checking if data is saved for jobid={jobid}: {e}')
        return False


def load_or_save_result_data(celery_app, jobid) -> AnalysisData | None:
    """
    Load analysis data from Redis cache or Celery result and save to cache if needed.

    Priority order:
    1. Redis cache (primary storage)
    2. Celery result backend (for newly completed jobs)

    Args:
        celery_app: Celery application instance
        jobid: Job identifier

    Returns:
        AnalysisData object if found, None otherwise
    """
    logger.info(f'Trying to load data for job_id={jobid}')
    cache = get_cache()

    # Try loading from Redis cache first
    try:
        data = cache.load(jobid)
        if data is not None:
            logger.info(f'Loaded data from Redis cache for job_id={jobid}')
            return data
    except Exception as e:
        logger.exception(f'Error loading from Redis cache for job_id={jobid}: {e}')

    # Try loading from Celery result backend
    try:
        job = AsyncResult(jobid, app=celery_app)
        if job and job.state == 'SUCCESS':
            data_json, _log = job.result

            # Save to Redis cache
            data = AnalysisData.from_json(data_json)
            cache.save(jobid, data)
            logger.info(f'Saved Celery result to Redis cache for job_id={jobid}')

            return data
    except Exception as e:
        logger.exception(f'Error loading from Celery result for job_id={jobid}: {e}')

    logger.warning(f'No data found for job_id={jobid}')
    return None


def preprocess_string(s):
    """
    Preprocess string for use in filenames and URLs.

    Args:
        s: String to preprocess

    Returns:
        Preprocessed string with special characters replaced by dashes
    """
    return re.sub(r'-{2,}', '-', re.sub(r'[^a-z0-9_]+', '-', s.lower())).strip('-')
