from logging import getLogger

from celery import current_task

from pysrc.app.messages import DOI_WRONG_SEARCH
from pysrc.celery.pubtrends_celery import pubtrends_celery
from pysrc.config import *
from pysrc.papers.analysis.expand import expand_ids
from pysrc.papers.analysis.pubmed import pubmed_search
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.utils import SORT_MOST_CITED, preprocess_doi, is_doi

logger = getLogger(__name__)


@pubtrends_celery.task(name='analyze_search_terms')
def analyze_search_terms(source, query, sort, limit, noreviews, min_year, max_year, topics, test=False):
    if is_doi(query):
        raise SearchError(DOI_WRONG_SEARCH)
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader(source, config)
    analyzer = PapersAnalyzer(loader, config)
    last_update = loader.last_update()
    if last_update is not None:
        analyzer.progress.info(f'Last papers update {last_update}', current=0, task=current_task)
    try:
        sort = sort or SORT_MOST_CITED
        limit = int(limit) if limit is not None and limit != '' else analyzer.config.show_max_articles_default_value
        topics = int(topics) if topics is not None and topics != '' else analyzer.config.show_topics_default_value
        ids = analyzer.search_terms(query, limit=limit, sort=sort,
                                    noreviews=noreviews, min_year=min_year, max_year=max_year,
                                    task=current_task)
        analyzer.analyze_papers(ids, topics, test=test, task=current_task)
    finally:
        loader.close_connection()

    data = analyzer.save(None, query, source, sort, limit, noreviews, min_year, max_year)
    analyzer.progress.done(task=current_task)
    analyzer.teardown()
    return data.to_json(), analyzer.progress.log()


@pubtrends_celery.task(name='analyze_id_list')
def analyze_id_list(source, query, ids, topics, test=False):
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader(source, config)
    analyzer = PapersAnalyzer(loader, config)
    sort = SORT_MOST_CITED
    limit = analyzer.config.show_max_articles_default_value
    return _analyze_id_list(analyzer, query, ids, ids, source, sort, limit, False, None, None, topics,
                            test=test, task=current_task)


def _analyze_id_list(analyzer, query , search_ids,
                     ids, source, sort, limit, noreviews, min_year, max_year, topics,
                     test=False, task=None):
    if len(ids) == 0:
        raise RuntimeError('Empty papers list')
    analyzer.progress.info(f'Analyzing {len(ids)} paper(s) from {source}', current=1, task=task)
    try:
        analyzer.analyze_papers(ids, topics, test=test, task=current_task)
    finally:
        analyzer.loader.close_connection()

    data = analyzer.save(search_ids, query, source, sort, limit, noreviews, min_year, max_year)
    analyzer.progress.done(task=task)
    analyzer.teardown()
    return data.to_json(), analyzer.progress.log()


@pubtrends_celery.task(name='analyze_search_paper')
def analyze_search_paper(source, pid, key, value, expand, limit, noreviews, topics, test=False):
    if key is not None and key != 'doi' and is_doi(preprocess_doi(value)):
        raise SearchError(DOI_WRONG_SEARCH)
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader(source, config)
    analyzer = PapersAnalyzer(loader, config)
    last_update = analyzer.loader.last_update()
    if last_update is not None:
        analyzer.progress.info(f'Last papers update {last_update}', current=0, task=current_task)
    analyzer.progress.info(f'Searching for publications with {key}={value}', current=1, task=current_task)
    try:
        if pid is not None:
            result = [pid]
        else:
            result = loader.search_key_value(key, value)

        if len(result) == 0:
            raise SearchError(f'No papers found with {key}={value}')

        analyzer.progress.info('Expanding related papers by references', current=1, task=current_task)
        limit = int(limit) if limit is not None and limit != '' else analyzer.config.show_max_articles_default_value
        topics = int(topics) if topics is not None and topics != '' else analyzer.config.show_topics_default_value
        expand = int(expand) if expand is not None and expand != '' else analyzer.config.paper_expands_steps
        ids = expand_ids(loader=analyzer.loader, search_ids=result,
                         expand_steps=expand, limit=limit, noreviews=noreviews,
                         max_expand=analyzer.config.paper_expand_limit)
        return _analyze_id_list(
            analyzer, f'Paper {key}={value}', result,
            ids, source, None, limit, False, None, None,
            topics,
            test=test, task=current_task
        )
    finally:
        loader.close_connection()


@pubtrends_celery.task(name='analyze_pubmed_search')
def analyze_pubmed_search(query, sort, limit, topics, test=False):
    if is_doi(preprocess_doi(query)):
        raise SearchError(DOI_WRONG_SEARCH)
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader('Pubmed', config)
    analyzer = PapersAnalyzer(loader, config)
    last_update = analyzer.loader.last_update()
    if last_update is not None:
        analyzer.progress.info(f'Last papers update {last_update}', current=0, task=current_task)
    analyzer.progress.info(f"Searching Pubmed query: {query}, {sort.lower()} limit {limit}",
                           current=1, task=current_task)
    sort = sort or SORT_MOST_CITED
    limit = int(limit) if limit is not None and limit != '' else analyzer.config.show_max_articles_default_value
    topics = int(topics) if topics is not None and topics != '' else analyzer.config.show_topics_default_value
    ids = pubmed_search(query, sort, limit)
    return _analyze_id_list(
        analyzer, query, ids, ids, 'Pubmed', sort, limit, False, None, None,
        topics,
        test=test, task=current_task
    )
