import logging

from cachetools import LRUCache, Cache, cached
from celery.result import AsyncResult

from models.celery.lru_ttl_cache_with_callback import lru_ttl_cache_with_callback
from models.celery.tasks import celery
from models.keypaper.config import PubtrendsConfig


PUBTRENDS_CONFIG = PubtrendsConfig(test=False)


def get_or_cancel_task(jobid):
    """
    This cache function is used to cancel tasks not polled for some time, i.e.
    when user requests to many searches and closes the tabs before getting the results.
    """
    logging.debug(f'get_or_cancel_task access: {jobid}')
    return _get_or_cancel_task(jobid)


def _celery_revoke_pending_task(jobid):
    task = AsyncResult(jobid, app=celery)
    if task is not None and task.state in {'STARTED', 'PENDING'}:
        logging.debug(f'REVOKE Celery task: {jobid}')
        celery.control.revoke(jobid, terminate=True)


@lru_ttl_cache_with_callback(maxsize=PUBTRENDS_CONFIG.celery_max_pending_tasks,
                             timeout=PUBTRENDS_CONFIG.celery_pending_tasks_timeout,
                             remove_callback=_celery_revoke_pending_task)
def _get_or_cancel_task(jobid):
    logging.debug(f'get_or_cancel_task new: {jobid}')
    return AsyncResult(jobid, app=celery)
