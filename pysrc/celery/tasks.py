import html
import os

from celery import Celery, current_task

from pysrc.papers.analyzer import KeyPaperAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.plot.plotter import visualize_analysis
from pysrc.papers.progress import Progress
from pysrc.papers.utils import SORT_MOST_CITED, ZOOM_OUT, PAPER_ANALYSIS

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379'),
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379')

# Configure Celery
celery = Celery("tasks", backend=CELERY_RESULT_BACKEND, broker=CELERY_BROKER_URL)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)


@celery.task(name='analyze_search_terms')
def analyze_search_terms(source, query, sort=None, limit=None, noreviews=True, expand=0.5):
    loader = Loaders.get_loader(source, PUBTRENDS_CONFIG)
    analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
    try:
        sort = sort or SORT_MOST_CITED
        limit = limit or analyzer.config.show_max_articles_default_value
        ids = analyzer.search_terms(query, limit=limit, sort=sort,
                                    noreviews=noreviews,
                                    task=current_task)
        if ids and expand != 0:
            ids = analyzer.expand_ids(
                ids,
                limit=min(int(min(len(ids), limit) * (1 + expand)), analyzer.config.max_number_to_expand),
                steps=KeyPaperAnalyzer.EXPAND_STEPS, keep_citations=True,
                current=2, task=current_task
            )
        analyzer.analyze_papers(ids, query, noreviews=noreviews, task=current_task)
    finally:
        loader.close_connection()

    analyzer.progress.info('Visualizing', current=analyzer.progress.total - 1, task=current_task)

    visualization = visualize_analysis(analyzer)
    dump = analyzer.dump()
    analyzer.progress.done(task=current_task)
    analyzer.teardown()
    return visualization, dump, analyzer.progress.log()


@celery.task(name='analyze_id_list')
def analyze_id_list(source, ids, zoom, query, limit=None):
    if len(ids) == 0:
        raise RuntimeError("Empty papers list")

    loader = Loaders.get_loader(source, PUBTRENDS_CONFIG)
    analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
    try:
        if zoom == ZOOM_OUT:
            ids = analyzer.expand_ids(
                ids,
                limit=min(len(ids) + KeyPaperAnalyzer.EXPAND_ZOOM_OUT, analyzer.config.max_number_to_expand),
                steps=1, keep_citations=True,
                current=1, task=current_task
            )
        elif zoom == PAPER_ANALYSIS:
            if limit:
                limit = int(limit)
            else:
                limit = 0
            ids = analyzer.expand_ids(
                ids,
                limit=limit if limit > 0 else analyzer.config.max_number_to_expand,
                steps=KeyPaperAnalyzer.EXPAND_STEPS, keep_citations=False,
                current=1, task=current_task)
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


@celery.task(name='find_paper_async')
def find_paper_async(source, key, value):
    loader = Loaders.get_loader(source, PUBTRENDS_CONFIG)
    progress = Progress(total=2)
    try:
        progress.info(f"Searching for a publication with {key} '{value}'", current=1, task=None)
        result = loader.find(key, value)
        progress.info(f'Found {len(result)} publications in the local database', current=1, task=None)

        if len(result) == 1:
            progress.info('Done', current=2)
            return result
        elif len(result) == 0:
            raise SearchError('Found no papers matching specified key - value pair')
        else:
            raise SearchError('Found multiple papers matching your search, please try to be more specific')
    finally:
        loader.close_connection()
