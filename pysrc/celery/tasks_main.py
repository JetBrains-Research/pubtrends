from logging import getLogger

from celery import current_task
from pysrc.papers.analyzer_files import AnalyzerFiles

from pysrc.celery.pubtrends_celery import pubtrends_celery
from pysrc.papers.analysis.expand import expand_ids
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.plot.plotter import visualize_analysis
from pysrc.papers.utils import SORT_MOST_CITED, pubmed_search, PAPER_ANALYSIS_TYPE, IDS_ANALYSIS_TYPE

logger = getLogger(__name__)


@pubtrends_celery.task(name='analyze_search_terms')
def analyze_search_terms(source, query, sort=None, limit=None, noreviews=True, expand=0.5, test=False):
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader(source, config)
    analyzer = PapersAnalyzer(loader, config)
    analyzer.progress.info('Analyzing search query', current=1, task=current_task)
    try:
        sort = sort or SORT_MOST_CITED
        limit = limit or analyzer.config.show_max_articles_default_value
        ids = analyzer.search_terms(query, limit=limit, sort=sort,
                                    noreviews=noreviews,
                                    task=current_task)
        if ids and expand != 0:
            ids = expand_ids(
                ids,
                min(int(min(len(ids), limit) * (1 + expand)), analyzer.config.max_number_to_expand),
                loader,
                PapersAnalyzer.EXPAND_LIMIT,
                PapersAnalyzer.EXPAND_CITATIONS_Q_LOW,
                PapersAnalyzer.EXPAND_CITATIONS_Q_HIGH,
                PapersAnalyzer.EXPAND_CITATIONS_SIGMA,
                PapersAnalyzer.EXPAND_SIMILARITY_THRESHOLD,
                analyzer.progress, current=2, task=current_task
            )
        analyzer.analyze_papers(ids, query, test=test, task=current_task)
    finally:
        loader.close_connection()

    analyzer.progress.info('Visualizing', current=analyzer.progress.total - 1, task=current_task)

    visualization = visualize_analysis(analyzer)
    dump = analyzer.dump()
    analyzer.progress.done(task=current_task)
    analyzer.teardown()
    return visualization, dump, analyzer.progress.log()


@pubtrends_celery.task(name='analyze_search_terms_files')
def analyze_search_terms_files(source, query, sort=None, limit=None, noreviews=True, expand=0.5, test=False):
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader(source, config)
    analyzer = AnalyzerFiles(loader, config)
    analyzer.progress.info('Analyzing search query', current=1, task=current_task)
    try:
        sort = sort or SORT_MOST_CITED
        limit = limit or analyzer.config.show_max_articles_default_value
        ids = analyzer.search_terms(query, limit=limit, sort=sort,
                                    noreviews=noreviews,
                                    task=current_task)
        if ids and expand != 0:
            ids = expand_ids(
                ids,
                min(int(min(len(ids), limit) * (1 + expand)), analyzer.config.max_number_to_expand),
                loader,
                PapersAnalyzer.EXPAND_LIMIT,
                PapersAnalyzer.EXPAND_CITATIONS_Q_LOW,
                PapersAnalyzer.EXPAND_CITATIONS_Q_HIGH,
                PapersAnalyzer.EXPAND_CITATIONS_SIGMA,
                PapersAnalyzer.EXPAND_SIMILARITY_THRESHOLD,
                analyzer.progress, current=2, task=current_task
            )
        analyzer.analyze_ids(ids, source, query, limit, test=test, task=current_task)
        analyzer.progress.done(task=current_task)
        analyzer.teardown()
        return analyzer.query_folder
    finally:
        loader.close_connection()


@pubtrends_celery.task(name='analyze_id_list')
def analyze_id_list(source, ids, query, analysis_type, limit=None, test=False):
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader(source, config)
    analyzer = PapersAnalyzer(loader, config)
    return _analyze_id_list(analyzer, source, ids, query, analysis_type, limit, test, current_task)


def _analyze_id_list(analyzer, source, ids, query, analysis_type=IDS_ANALYSIS_TYPE, limit=None, test=False, task=None):
    if len(ids) == 0:
        raise RuntimeError("Empty papers list")
    analyzer.progress.info(f'Analyzing paper(s) from {source}', current=1, task=task)
    try:
        if analysis_type == PAPER_ANALYSIS_TYPE:
            limit = int(limit) if limit else analyzer.config.max_number_to_expand
            # Fetch references at first, but in some cases paper may have empty references
            ids = ids + analyzer.load_references(ids[0], limit=limit)
            # And then expand
            ids = expand_ids(
                ids,
                limit,
                analyzer.loader,
                PapersAnalyzer.EXPAND_LIMIT,
                PapersAnalyzer.EXPAND_CITATIONS_Q_LOW,
                PapersAnalyzer.EXPAND_CITATIONS_Q_HIGH,
                PapersAnalyzer.EXPAND_CITATIONS_SIGMA,
                PapersAnalyzer.EXPAND_SIMILARITY_THRESHOLD,
                analyzer.progress, current=1, task=task
            )
        elif analysis_type == IDS_ANALYSIS_TYPE:
            ids = ids  # Leave intact
        else:
            raise Exception(f'Illegal analysis type {analysis_type}')
        analyzer.analyze_papers(ids, query, test=test, task=task)
    finally:
        analyzer.loader.close_connection()

    analyzer.progress.info('Visualizing', current=analyzer.progress.total - 1, task=task)
    visualization = visualize_analysis(analyzer)
    dump = analyzer.dump()
    analyzer.progress.done(task=task)
    analyzer.teardown()
    return visualization, dump, analyzer.progress.log()


@pubtrends_celery.task(name='analyze_search_paper')
def analyze_search_paper(source, key, value, test=False):
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader(source, config)
    analyzer = PapersAnalyzer(loader, config)
    analyzer.progress.info(f"Searching for a publication with {key} '{value}'", current=1, task=current_task)
    try:
        result = loader.find(key, value)
        if len(result) == 1:
            return _analyze_id_list(
                analyzer,
                source, ids=result, query=f'Paper {key}={value}', analysis_type=PAPER_ANALYSIS_TYPE, limit='',
                test=test, task=current_task
            )
        elif len(result) == 0:
            raise SearchError(f'Found no papers with {key}={value}')
        else:
            raise SearchError('Found multiple papers matching your search, please try to be more specific')
    finally:
        loader.close_connection()


@pubtrends_celery.task(name='analyze_pubmed_search')
def analyze_pubmed_search(query, limit=None, test=False):
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader('Pubmed', config)
    analyzer = PapersAnalyzer(loader, config)
    analyzer.progress.info(f"Searching Pubmed query: '{query}', limit {limit}", current=1, task=current_task)
    ids = pubmed_search(query, limit)
    return _analyze_id_list(analyzer, 'Pubmed', ids, query=query, analysis_type=IDS_ANALYSIS_TYPE, limit=limit,
                            test=test, task=current_task)


@pubtrends_celery.task(name='analyze_pubmed_search_files')
def analyze_pubmed_search_files(query, limit, test=False):
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader('Pubmed', config)
    analyzer = AnalyzerFiles(loader, config)
    try:
        analyzer.progress.info(f"Searching Pubmed query: '{query}', limit {limit}", current=1, task=current_task)
        ids = pubmed_search(query, limit)
        analyzer.analyze_ids(ids, 'Pubmed', query, limit, test=test, task=current_task)
        analyzer.progress.done(task=current_task)
        analyzer.teardown()
        return analyzer.query_folder
    finally:
        loader.close_connection()
