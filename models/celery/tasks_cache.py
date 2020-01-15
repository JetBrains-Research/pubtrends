import logging

from cachetools import LRUCache, Cache, cached
from celery.result import AsyncResult

from models.celery.lru_ttl_cache_with_callback import lru_ttl_cache_with_callback
from models.celery.tasks import celery
from models.keypaper.config import PubtrendsConfig


PUBTRENDS_CONFIG = PubtrendsConfig(test=False)


def _celery_revoke_completed_task(jobid):
    logging.debug(f'Revoke Celery complete task: {jobid}')
    celery.control.revoke(jobid, terminate=True)


class LRUCacheRemoveCallback(LRUCache):
    """LRU cache with callbacks on remove event."""

    def __init__(self, maxsize, remove_callback):
        super().__init__(maxsize)
        self.remove_callback = remove_callback

    def __delitem__(self, key, cache_delitem=Cache.__delitem__):
        self.remove_callback()
        super().__delitem__(key, cache_delitem)


# This cache is used to story already completed tasks
_completed_tasks_cache = LRUCacheRemoveCallback(maxsize=PUBTRENDS_CONFIG.celery_max_completed_tasks,
                                                remove_callback=_celery_revoke_completed_task)


def complete_task(jobid):
    """
    This cache function marks Celery task as completed and
    prevents TTL cache revoking after no polling timeout
    """
    logging.debug(f'complete_task access: {jobid}')
    return _complete_task(jobid)


@cached(cache=_completed_tasks_cache)
def _complete_task(jobid):
    logging.debug(f'complete_task new: {jobid}')
    return AsyncResult(jobid, app=celery)


def _celery_revoke_pending_task(jobid):
    if jobid not in _completed_tasks_cache:
        logging.debug(f'REVOKE Celery pending task: {jobid}')
        celery.control.revoke(jobid, terminate=True)


def get_or_cancel_task(jobid):
    """
    This cache function is used to cancel tasks not polled for some time, i.e.
    when user requests to many searches and closes the tabs before getting the results.
    """
    logging.debug(f'get_or_cancel_task access: {jobid}')
    return _get_or_cancel_task(jobid)


@lru_ttl_cache_with_callback(maxsize=PUBTRENDS_CONFIG.celery_max_pending_tasks,
                             timeout=PUBTRENDS_CONFIG.celery_pending_tasks_timeout,
                             remove_callback=_celery_revoke_pending_task)
def _get_or_cancel_task(jobid):
    logging.debug(f'get_or_cancel_task new: {jobid}')
    return AsyncResult(jobid, app=celery)
