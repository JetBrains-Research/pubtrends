import hashlib
import logging
import time
from threading import Lock

from pysrc.app.reports import load_result_data
from pysrc.celery.tasks_main import analyze_search_terms, analyze_semantic_search
from pysrc.config import PubtrendsConfig
from pysrc.papers.utils import SORT_MOST_CITED
from pysrc.services.semantic_search_service import is_semantic_search_service_available

logger = logging.getLogger(__name__)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

SEMANTIC_SEARCH_AVAILABLE = PUBTRENDS_CONFIG.feature_semantic_search_enabled and is_semantic_search_service_available()

PREDEFINED_PREFIX = 'predefined-'

PREDEFINED_SEMANTIC_PREFIX = 'predefined-semantic-'

PREDEFINED_SORT = SORT_MOST_CITED

PREDEFINED_LIMIT = 1000

PREDEFINED_NOREVIEWS = True

PREDEFINED_TASKS_READY_KEY = 'PREDEFINED_TASKS_READY'


def is_semantic_predefined(jobid):
    return jobid.startswith(PREDEFINED_SEMANTIC_PREFIX)


def get_predefined_jobs(config, semantic_search_available):
    result = {}
    if config.pm_enabled:
        result['Pubmed'] = _predefined_examples_jobid(config.pm_search_example_terms, semantic_search_available)
    if config.ss_enabled:
        result['Semantic Scholar'] = _predefined_examples_jobid(config.ss_search_example_terms,
                                                                semantic_search_available)
    return result

def _predefined_examples_jobid(examples, semantic_search_available):
    predefined = [(t, PREDEFINED_PREFIX + hashlib.md5(t.encode('utf-8')).hexdigest()) for t in examples]
    if not semantic_search_available:
        return predefined
    semantic_search_predefined = \
        [(t.replace('"', ''), PREDEFINED_SEMANTIC_PREFIX + hashlib.md5(t.replace('"', '').encode('utf-8')).hexdigest())
         for t in examples]
    return predefined + semantic_search_predefined


PREDEFINED_JOBS = get_predefined_jobs(PUBTRENDS_CONFIG, SEMANTIC_SEARCH_AVAILABLE)

PREDEFINED_JOBS_LOCK = Lock()

# Throttle expensive readiness checks against Celery.
# During start-up many concurrent '/' requests may arrive and each would call
# `are_predefined_jobs_ready`, which performs multiple `celery.control.inspect()`
# calls and filesystem reads. To avoid hammering Celery and disk, we only perform
# a full check at most once per `READY_CHECK_INTERVAL_SEC` seconds while the
# system is still becoming ready. Once ready, the result is persisted in
# `pubtrends_app.config[PREDEFINED_TASKS_READY_KEY]` and no further checks are made.
READY_CHECK_INTERVAL_SEC = 5
_last_ready_check_ts = 0.0


def are_predefined_jobs_ready(pubtrends_app, pubtrends_celery):
    """ Checks if all the precomputed examples are available and gensim fasttext model is loaded """
    if len(PREDEFINED_JOBS) == 0:
        return True
    try:
        PREDEFINED_JOBS_LOCK.acquire()
        # If we already marked the app as ready, exit fast without any extra work.
        if pubtrends_app.config.get(PREDEFINED_TASKS_READY_KEY, False):
            return True
        # Throttle expensive checks when not yet ready.
        global _last_ready_check_ts
        now = time.time()
        if now - _last_ready_check_ts < READY_CHECK_INTERVAL_SEC:
            return False
        _last_ready_check_ts = now
        ready = True
        inspect = pubtrends_celery.control.inspect()
        active = inspect.active()
        if active is None:
            return False
        active_jobs = [j['id'] for j in list(active.items())[0][1]]
        reserved = inspect.reserved()
        if reserved is None:
            return False
        scheduled_jobs = [j['id'] for j in list(reserved.items())[0][1]]

        for source, predefine_info in PREDEFINED_JOBS.items():
            for query, jobid in predefine_info:
                logger.info(f'Check predefined search for source={source} query={query} jobid={jobid}')
                # Check celery queue
                if jobid in active_jobs or jobid in scheduled_jobs:
                    ready = False
                    continue

                if not is_semantic_predefined(jobid):
                    data = load_result_data(
                        jobid, source, query, PREDEFINED_SORT, PREDEFINED_LIMIT, PREDEFINED_NOREVIEWS,
                        None, None, pubtrends_celery
                    )
                    if data is None:
                        logger.info(f'No job or out-of-date job for source={source} query={query}, launch it')
                        analyze_search_terms.apply_async(
                            args=[source, query, PREDEFINED_SORT, PREDEFINED_LIMIT, PREDEFINED_NOREVIEWS,
                                  None, None,
                                  PUBTRENDS_CONFIG.show_topics_default_value, pubtrends_app.config['TESTING']],
                            task_id=jobid
                        )
                        ready = False
                else:
                    data = load_result_data(
                        jobid, source, query, '', PREDEFINED_LIMIT, PREDEFINED_NOREVIEWS,
                        None, None,
                        pubtrends_celery
                    )
                    if data is None:
                        logger.info(f'No job or out-of-date job for source={source} semantic query={query}, launch it')
                        analyze_semantic_search.apply_async(
                            args=[source, query, PREDEFINED_LIMIT, PREDEFINED_NOREVIEWS,
                                  PUBTRENDS_CONFIG.show_topics_default_value, pubtrends_app.config['TESTING']],
                            task_id=jobid
                        )
                        ready = False
        if ready:
            pubtrends_app.config[PREDEFINED_TASKS_READY_KEY] = True
        return ready
    finally:
        PREDEFINED_JOBS_LOCK.release()
