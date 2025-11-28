import hashlib
import logging
from threading import Lock

from pysrc.app.reports import is_data_saved
from pysrc.app.reports import load_or_save_result_data
from pysrc.celery.tasks_main import analyze_search_terms, analyze_semantic_search, analyze_search_paper
from pysrc.config import PubtrendsConfig
from pysrc.papers.utils import SORT_MOST_CITED
from pysrc.services.semantic_search_service import is_semantic_search_service_available

logger = logging.getLogger(__name__)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

SEMANTIC_SEARCH_AVAILABLE = PUBTRENDS_CONFIG.feature_semantic_search_enabled and is_semantic_search_service_available()

PREDEFINED_TERMS_PREFIX = 'predefined-terms-'

PREDEFINED_PAPER_PREFIX = 'predefined-paper-'

PREDEFINED_SEMANTIC_PREFIX = 'predefined-semantic-'

PREDEFINED_SORT = SORT_MOST_CITED

PREDEFINED_LIMIT = 1000

PREDEFINED_NOREVIEWS = True

PREDEFINED_TASKS_READY_KEY = 'PREDEFINED_TASKS_READY'


def is_terms_predefined(jobid):
    return jobid.startswith(PREDEFINED_TERMS_PREFIX)

def is_paper_predefined(jobid):
    return jobid.startswith(PREDEFINED_PAPER_PREFIX)

def is_semantic_predefined(jobid):
    return jobid.startswith(PREDEFINED_SEMANTIC_PREFIX)


def get_predefined_jobs(config, semantic_search_available):
    result = {}
    if config.pm_enabled:
        result['Pubmed'] = _predefined_examples_jobid(
            config.pm_search_example_terms, semantic_search_available, config.pm_paper_examples
        )
    if config.ss_enabled:
        result['Semantic Scholar'] = _predefined_examples_jobid(config.ss_search_example_terms,
                                                                semantic_search_available)
    return result

def _predefined_examples_jobid(examples, semantic_search_available, papers=()):
    predefined = [(t, PREDEFINED_TERMS_PREFIX + hashlib.md5(t.encode('utf-8')).hexdigest()) for t in examples]
    papers_predefined = [(f'{k}={v}', PREDEFINED_PAPER_PREFIX + hashlib.md5(f'{k}={v}'.encode('utf-8')).hexdigest())
                         for (k, v) in papers]
    if semantic_search_available:
        semantic_search_predefined = \
            [(t.replace('"', ''), PREDEFINED_SEMANTIC_PREFIX + hashlib.md5(t.replace('"', '').encode('utf-8')).hexdigest())
            for t in examples]
    else:
        semantic_search_predefined = []
    return predefined + papers_predefined + semantic_search_predefined


PREDEFINED_JOBS = get_predefined_jobs(PUBTRENDS_CONFIG, SEMANTIC_SEARCH_AVAILABLE)

PREDEFINED_JOBS_LOCK = Lock()

def are_predefined_jobs_ready(pubtrends_app, pubtrends_celery):
    """ Checks if all the precomputed examples are available and gensim fasttext model is loaded """
    if len(PREDEFINED_JOBS) == 0:
        return True
    try:
        PREDEFINED_JOBS_LOCK.acquire()
        # If we already marked the app as ready, exit fast without any extra work.
        if pubtrends_app.config.get(PREDEFINED_TASKS_READY_KEY, False):
            return True
        ready = True
        inspect = pubtrends_celery.control.inspect()
        active = inspect.active()
        active_jobs = [j['id'] for j in list(active.items())[0][1]] if active is not None else []
        reserved = inspect.reserved()
        scheduled_jobs = [j['id'] for j in list(reserved.items())[0][1]] if reserved is not None else []
        for source, predefine_info in PREDEFINED_JOBS.items():
            for query, jobid in predefine_info:
                logger.info(f'Check predefined search for source={source} query={query} jobid={jobid}')
                if is_data_saved(jobid):
                    continue
                if load_or_save_result_data(pubtrends_celery, jobid) is None:
                    ready = False
                    if jobid in active_jobs or jobid in scheduled_jobs:
                        continue
                    else:
                        logger.info(f'Initiating predefined copmutation source={source} query={query}')
                        if is_terms_predefined(jobid):
                            analyze_search_terms.apply_async(
                                args=[source, query, PREDEFINED_SORT, PREDEFINED_LIMIT, PREDEFINED_NOREVIEWS,
                                      None, None,
                                      PUBTRENDS_CONFIG.show_topics_default_value,
                                      pubtrends_app.config['TESTING']],
                                task_id=jobid
                                )
                        elif is_paper_predefined(jobid):
                            key, value = query.split('=')
                            analyze_search_paper.apply_async(
                                args=[source, None, key, value, PUBTRENDS_CONFIG.paper_expands_steps,
                                      PREDEFINED_LIMIT,
                                      PREDEFINED_NOREVIEWS, PUBTRENDS_CONFIG.show_topics_default_value,
                                      pubtrends_app.config['TESTING']],
                                task_id=jobid
                            )
                        elif is_semantic_predefined(jobid):
                            analyze_semantic_search.apply_async(
                                args=[source, query, PREDEFINED_LIMIT, PREDEFINED_NOREVIEWS,
                                      PUBTRENDS_CONFIG.show_topics_default_value,
                                      pubtrends_app.config['TESTING']],
                                task_id=jobid
                            )
                        else:
                            raise Exception(f'Unknown predefined jobid {jobid}')
        if ready:
            pubtrends_app.config[PREDEFINED_TASKS_READY_KEY] = True
        return ready
    finally:
        PREDEFINED_JOBS_LOCK.release()
