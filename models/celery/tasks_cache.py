import logging
from multiprocessing import RLock

from cachetools import LRUCache, Cache, cached
from celery.result import AsyncResult
from datetime import timedelta, datetime

from models.celery.tasks import celery


MAX_PENDING_TASKS = 50  # Max allowed pending tasks
PENDING_TASKS_TIMEOUT = 60  # Seconds, pending task will be revoked after no polling activity
MAX_COMPLETED_TASKS = 1000  # Max completed tasks to store


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
_completed_tasks_cache = LRUCacheRemoveCallback(maxsize=MAX_COMPLETED_TASKS,
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


def lru_ttl_cache_with_callback(maxsize, timeout, remove_callback):
    assert maxsize is not None and maxsize > 0, \
        f"lru_ttl_cache_with_callback: expected maxsize>0: {maxsize}"

    def decorating_function(user_function):
        return _lru_ttl_cache_with_callback(user_function,
                                            maxsize,
                                            timedelta(seconds=timeout),
                                            remove_callback)

    return decorating_function


def _lru_ttl_cache_with_callback(user_function, maxsize, update_delta, remove_callback):
    """
    NOTE: this is a slightly modified version of generic _lru_cache from functools

    Major differences are:
    * This version is specified for given maxsize
    * It uses TIMESTAMP to update access time for items
    * All the items older than update_delta are removed and callback is executed
    * Single argument user function is supported

    """
    # Constants shared by all lru cache instances:
    PREV, NEXT, KEY, RESULT, TIMESTAMP = 0, 1, 2, 3, 4  # names for the link fields

    cache = {}
    full = False
    lock = RLock()  # because linkedlist updates aren't threadsafe
    root = []  # root of the circular doubly linked list
    root[:] = [root, root, None, None, None]  # initialize by pointing to self

    # init first update
    next_update = datetime.utcnow() - update_delta

    def wrapper(key):
        # Size limited caching that tracks accesses by recency
        nonlocal root, full, next_update
        with lock:
            now = datetime.utcnow()
            if now >= next_update:
                logging.debug('_lru_ttl_cache_with_callback: expire cleanup')
                expire(now)
                next_update = now + update_delta

            link = cache.get(key)
            if link is not None:
                # Move the link to the front of the circular queue
                link_prev, link_next, _key, result, _ = link
                link_prev[NEXT] = link_next
                link_next[PREV] = link_prev
                last = root[PREV]
                last[NEXT] = root[PREV] = link
                link[PREV] = last
                link[NEXT] = root
                logging.debug(f'_lru_ttl_cache_with_callback: update timestamp for {key}')
                link[TIMESTAMP] = now
                return result

            result = user_function(key)
            logging.debug(f'_lru_ttl_cache_with_callback: new key {key}')
            if full:
                logging.debug('Full')
                # Use the old root to store the new key and result.
                oldroot = root
                oldroot[KEY] = key
                oldroot[RESULT] = result
                root = oldroot[NEXT]
                oldkey = root[KEY]
                root[KEY] = root[RESULT] = root[TIMESTAMP] = None
                # Now update the cache dictionary.
                del cache[oldkey]
                logging.debug(f'_celery_tasks_cache: full remove {oldkey}')
                remove_callback(oldkey)

                # Save the potentially reentrant cache[key] assignment
                # for last, after the root and links have been put in
                # a consistent state.
                cache[key] = oldroot
            else:
                # Put result in a new link at the front of the queue.
                last = root[PREV]
                link = [last, root, key, result, now]
                last[NEXT] = root[PREV] = cache[key] = link
                # Use the cache_len bound method instead of the len() function
                # which could potentially be wrapped in an lru_cache itself.
                full = (cache.__len__() >= maxsize)
            return result

    def size():
        # Use the cache_len bound method instead of the len() function
        # which could potentially be wrapped in an lru_cache itself.
        return cache.__len__()

    def keys():
        return list(cache.keys())

    def expire(now):
        nonlocal root, full
        link = root
        link_next = link[NEXT]
        while link_next is not root:
            if link_next[KEY] is not None and now > link_next[TIMESTAMP] + update_delta:
                key = link_next[KEY]
                logging.debug(f'_celery_tasks_cache: expired {key}')
                del cache[key]
                remove_callback(key)
                link_next_next = link_next[NEXT]
                link[NEXT] = link_next_next
                link_next_next[PREV] = link
                link_next = link_next_next
            else:
                link = link_next
                link_next = link_next[NEXT]

        assert link_next is root, '_lru_ttl_cache_with_callback: expire procedure failed, expected root'
        assert root[KEY] is None, '_lru_ttl_cache_with_callback: root should not contain any key'

        # Use the cache_len bound method instead of the len() function
        # which could potentially be wrapped in an lru_cache itself.
        full = cache.__len__() >= maxsize

    wrapper.size = size
    wrapper.keys = keys
    return wrapper


def get_or_cancel_task(jobid):
    """
    This cache function is used to cancel tasks not polled for some time, i.e.
    when user requests to many searches and closes the tabs before getting the results.
    """
    logging.debug(f'get_or_cancel_task access: {jobid}')
    return _get_or_cancel_task(jobid)


@lru_ttl_cache_with_callback(maxsize=MAX_PENDING_TASKS,
                             timeout=PENDING_TASKS_TIMEOUT,
                             remove_callback=_celery_revoke_pending_task)
def _get_or_cancel_task(jobid):
    logging.debug(f'get_or_cancel_task new: {jobid}')
    return AsyncResult(jobid, app=celery)
