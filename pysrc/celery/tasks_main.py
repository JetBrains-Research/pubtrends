from celery import current_task
from logging import getLogger

from pysrc.app.messages import DOI_WRONG_SEARCH
from pysrc.celery.pubtrends_celery import pubtrends_celery
from pysrc.papers.analysis.expand import expand_ids
from pysrc.papers.analysis.pubmed import pubmed_search
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.analyzer_files import AnalyzerFiles
from pysrc.papers.config import *
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.plot.plotter import visualize_analysis
from pysrc.papers.utils import SORT_MOST_CITED, preprocess_doi, is_doi

logger = getLogger(__name__)


@pubtrends_celery.task(name='analyze_search_terms')
def analyze_search_terms(source, query, sort, limit, noreviews, topics, test=False):
    if is_doi(query):
        raise SearchError(DOI_WRONG_SEARCH)
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader(source, config)
    analyzer = PapersAnalyzer(loader, config)
    last_update = loader.last_update()
    if last_update is not None:
        analyzer.progress.info(f'Last papers update {last_update}', current=0, task=current_task)
    analyzer.progress.info('Analyzing search query', current=1, task=current_task)
    try:
        sort = sort or SORT_MOST_CITED
        limit = int(limit) if limit is not None and limit != '' else analyzer.config.show_max_articles_default_value
        topics = int(topics) if topics is not None and topics != '' else analyzer.config.show_topics_default_value
        ids = analyzer.search_terms(query, limit=limit, sort=sort,
                                    noreviews=noreviews,
                                    task=current_task)
        analyzer.analyze_papers(ids, query, topics, test=test, task=current_task)
    finally:
        loader.close_connection()

    analyzer.progress.info('Visualizing results', current=analyzer.progress.total - 1, task=current_task)

    visualization = visualize_analysis(analyzer)
    dump = analyzer.dump()
    analyzer.progress.done(task=current_task)
    analyzer.teardown()
    return visualization, dump, analyzer.progress.log()


@pubtrends_celery.task(name='analyze_search_terms_files')
def analyze_search_terms_files(source, query, sort, limit, noreviews, topics, test=False):
    if is_doi(preprocess_doi(query)):
        raise SearchError(DOI_WRONG_SEARCH)
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader(source, config)
    analyzer = AnalyzerFiles(loader, config)
    last_update = loader.last_update()
    if last_update is not None:
        analyzer.progress.info(f'Last papers update {last_update}', current=0, task=current_task)
    analyzer.progress.info('Analyzing search query', current=1, task=current_task)
    try:
        sort = sort or SORT_MOST_CITED
        limit = int(limit) if limit is not None and limit != '' else analyzer.config.show_max_articles_default_value
        topics = int(topics) if topics is not None and topics != '' else analyzer.config.show_topics_default_value
        ids = analyzer.search_terms(query, limit=limit, sort=sort,
                                    noreviews=noreviews,
                                    task=current_task)
        analyzer.analyze_ids(ids, source, query, sort, limit, topics, test=test, task=current_task)
        analyzer.progress.done(task=current_task)
        analyzer.teardown()
        return analyzer.query_folder
    finally:
        loader.close_connection()


def _analyze_id_list(analyzer, source, ids,
                     query, topics,
                     test=False, task=None):
    if len(ids) == 0:
        raise RuntimeError('Empty papers list')
    analyzer.progress.info(f'Analyzing {len(ids)} paper(s) from {source}', current=1, task=task)
    try:
        analyzer.analyze_papers(ids, query, topics, test=test, task=task)
    finally:
        analyzer.loader.close_connection()

    analyzer.progress.info('Visualizing results', current=analyzer.progress.total - 1, task=task)
    visualization = visualize_analysis(analyzer)
    dump = analyzer.dump()
    analyzer.progress.done(task=task)
    analyzer.teardown()
    return visualization, dump, analyzer.progress.log()


@pubtrends_celery.task(name='analyze_search_paper')
def analyze_search_paper(source, pid, key, value, limit, topics, test=False):
    if key is not None and key != 'doi' and is_doi(preprocess_doi(value)):
        raise SearchError(DOI_WRONG_SEARCH)
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader(source, config)
    analyzer = PapersAnalyzer(loader, config)
    analyzer.progress.info(f'Searching for a publication with {key}={value}', current=1, task=current_task)
    try:
        if pid is not None:
            result = [pid]
        else:
            result = loader.find(key, value)
        if len(result) == 1:
            analyzer.progress.info('Expanding related papers by references', current=1, task=current_task)
            limit = int(limit) if limit is not None and limit != '' else analyzer.config.show_max_articles_default_value
            topics = int(topics) if topics is not None and topics != '' else analyzer.config.show_topics_default_value
            ids = expand_ids(loader=analyzer.loader, pid=result[0], limit=limit,
                             expand_steps=analyzer.config.paper_expands_steps,
                             max_expand=analyzer.config.paper_expand_limit,
                             citations_q_low=EXPAND_CITATIONS_Q_LOW,
                             citations_q_high=EXPAND_CITATIONS_Q_HIGH,
                             citations_sigma=EXPAND_CITATIONS_SIGMA,
                             mesh_similarity_threshold=EXPAND_MESH_SIMILARITY,
                             single_paper_impact=EXPAND_SINGLE_PAPER_IMPACT)
            return _analyze_id_list(
                analyzer, source, ids=ids, query=f'Paper {key}={value}',
                topics=topics,
                test=test, task=current_task
            )
        elif len(result) == 0:
            raise SearchError(f'No papers found with {key}={value}')
        else:
            raise SearchError(f'Multiple papers found matching {key}={value}, please be more specific')
    finally:
        loader.close_connection()


@pubtrends_celery.task(name='analyze_pubmed_search')
def analyze_pubmed_search(query, sort, limit, topics, test=False):
    if is_doi(preprocess_doi(query)):
        raise SearchError(DOI_WRONG_SEARCH)
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader('Pubmed', config)
    analyzer = PapersAnalyzer(loader, config)
    analyzer.progress.info(f"Searching Pubmed query: {query}, {sort.lower()} limit {limit}",
                           current=1, task=current_task)
    sort = sort or SORT_MOST_CITED
    limit = int(limit) if limit is not None and limit != '' else analyzer.config.show_max_articles_default_value
    topics = int(topics) if topics is not None and topics != '' else analyzer.config.show_topics_default_value
    ids = pubmed_search(query, sort, limit)
    return _analyze_id_list(analyzer, 'Pubmed',
                            ids, query=query,
                            topics=topics,
                            test=test, task=current_task)


@pubtrends_celery.task(name='analyze_pubmed_search_files')
def analyze_pubmed_search_files(query, sort, limit, topics, test=False):
    if is_doi(preprocess_doi(query)):
        raise SearchError(DOI_WRONG_SEARCH)
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader('Pubmed', config)
    analyzer = AnalyzerFiles(loader, config)
    try:
        sort = sort or SORT_MOST_CITED
        limit = int(limit) if limit is not None and limit != '' else analyzer.config.show_max_articles_default_value
        topics = int(topics) if topics is not None and topics != '' else analyzer.config.show_topics_default_value
        analyzer.progress.info(f"Searching Pubmed query: {query}, {sort.lower()} limit {limit} topics {topics}",
                               current=1, task=current_task)
        ids = pubmed_search(query, sort, limit)
        analyzer.progress.info(f'Analysing {len(ids)} paper(s) from Pubmed', current=1, task=current_task)
        analyzer.analyze_ids(ids, 'Pubmed', query, sort, limit, topics, test=test, task=current_task)
        analyzer.progress.done(task=current_task)
        analyzer.teardown()
        return analyzer.query_folder
    finally:
        loader.close_connection()
