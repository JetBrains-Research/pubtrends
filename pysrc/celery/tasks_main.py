from logging import getLogger

from celery import current_task

from pysrc.celery.pubtrends_celery import pubtrends_celery
from pysrc.papers.analysis.expand import expand_ids
from pysrc.papers.analysis.pm_advanced import PubmedAdvancedAnalyzer, pubmed_search
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.plot.plotter import visualize_analysis
from pysrc.papers.utils import SORT_MOST_CITED, ZOOM_OUT, PAPER_ANALYSIS

logger = getLogger(__name__)


@pubtrends_celery.task(name='analyze_search_terms')
def analyze_search_terms(source, query, sort=None, limit=None, noreviews=True, expand=0.5, test=False):
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader(source, config)
    analyzer = PapersAnalyzer(loader, config)
    analyzer.progress.info('Analyzing search query', current=0, task=current_task)
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
        analyzer.analyze_papers(ids, query, task=current_task)
    finally:
        loader.close_connection()

    analyzer.progress.info('Visualizing', current=analyzer.progress.total - 1, task=current_task)

    visualization = visualize_analysis(analyzer)
    dump = analyzer.dump()
    analyzer.progress.done(task=current_task)
    analyzer.teardown()
    return visualization, dump, analyzer.progress.log()


@pubtrends_celery.task(name='analyze_id_list')
def analyze_id_list(source, ids, zoom, query, limit=None, test=False):
    if len(ids) == 0:
        raise RuntimeError("Empty papers list")

    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader(source, config)
    analyzer = PapersAnalyzer(loader, config)
    analyzer.progress.info('Analyzing paper(s)', current=0, task=current_task)
    try:
        if zoom == ZOOM_OUT:
            ids = expand_ids(
                ids,
                min(len(ids) + PapersAnalyzer.EXPAND_ZOOM_OUT, analyzer.config.max_number_to_expand),
                loader,
                PapersAnalyzer.EXPAND_LIMIT,
                PapersAnalyzer.EXPAND_CITATIONS_Q_LOW,
                PapersAnalyzer.EXPAND_CITATIONS_Q_HIGH,
                PapersAnalyzer.EXPAND_CITATIONS_SIGMA,
                PapersAnalyzer.EXPAND_SIMILARITY_THRESHOLD,
                analyzer.progress, current=1, task=current_task
            )
        elif zoom == PAPER_ANALYSIS:
            if limit:
                limit = int(limit)
            else:
                limit = 0
            # Fetch references at first, but in some cases paper may have empty references
            ids = ids + analyzer.load_references(
                ids[0], limit=limit if limit > 0 else analyzer.config.max_number_to_expand
            )
            # And then expand
            ids = expand_ids(
                ids,
                limit if limit > 0 else analyzer.config.max_number_to_expand,
                loader,
                PapersAnalyzer.EXPAND_LIMIT,
                PapersAnalyzer.EXPAND_CITATIONS_Q_LOW,
                PapersAnalyzer.EXPAND_CITATIONS_Q_HIGH,
                PapersAnalyzer.EXPAND_CITATIONS_SIGMA,
                PapersAnalyzer.EXPAND_SIMILARITY_THRESHOLD,
                analyzer.progress, current=1, task=current_task
            )
        else:
            ids = ids  # Leave intact
        analyzer.analyze_papers(ids, query, task=current_task)
    finally:
        loader.close_connection()

    analyzer.progress.info('Visualizing', current=analyzer.progress.total - 1, task=current_task)
    visualization = visualize_analysis(analyzer)
    dump = analyzer.dump()
    analyzer.progress.done(task=current_task)
    analyzer.teardown()
    return visualization, dump, analyzer.progress.log()


@pubtrends_celery.task(name='analyze_search_paper')
def analyze_search_paper(source, key, value, test=False):
    try:
        logger.info(f"Searching for a publication with {key} '{value}'")
        loader = Loaders.get_loader(source, PubtrendsConfig(test=test))
        result = loader.find(key, value)
        if len(result) == 1:
            return analyze_id_list.delay(
                source, ids=result, zoom=PAPER_ANALYSIS, query='Paper', limit='',
                test=test
            )
        elif len(result) == 0:
            raise SearchError('Found no papers matching specified key - value pair')
        else:
            raise SearchError('Found multiple papers matching your search, please try to be more specific')
    finally:
        loader.close_connection()


@pubtrends_celery.task(name='analyze_pubmed_search')
def analyze_pubmed_search(query, limit=None, test=False):
    logger.info(f"Searching Pubmed query: '{query}', limit {limit}")
    ids = pubmed_search(query, limit)
    return analyze_id_list('Pubmed', ids, zoom=None, query=query, limit=limit, test=test)


@pubtrends_celery.task(name='analyze_pubmed_advanced_search')
def analyze_pubmed_advanced_search(query, limit, test=False):
    config = PubtrendsConfig(test=test)
    loader = Loaders.get_loader('Pubmed', config)
    analyzer = PubmedAdvancedAnalyzer(loader, config)
    try:
        analyzer.analyze_pubmed_advanced_search(query, limit, task=current_task)
        analyzer.progress.done(task=current_task)
        analyzer.teardown()
        return analyzer.query_folder
    finally:
        loader.close_connection()
