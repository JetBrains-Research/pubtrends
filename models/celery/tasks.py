import html
import os

from celery import Celery, current_task

from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.keypaper.pm_loader import PubmedLoader
from models.keypaper.progress_logger import ProgressLogger
from models.keypaper.ss_loader import SemanticScholarLoader
from models.keypaper.utils import SORT_MOST_CITED
from models.keypaper.visualization import visualize_analysis

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379'),
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379')

# Configure Celery
celery = Celery("tasks", backend=CELERY_RESULT_BACKEND, broker=CELERY_BROKER_URL)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)


@celery.task(name='analyze_search_terms')
def analyze_search_terms(source, query, sort=None, limit=None):
    loader = get_loader(source, PUBTRENDS_CONFIG)
    analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
    try:
        sort = sort or SORT_MOST_CITED
        ids = analyzer.search_terms(query, limit=limit, sort=sort, task=current_task)
        analyzer.analyze_papers(ids, query, current_task)
    finally:
        loader.close_connection()

    analyzer.progress.info('Visualizing', current=analyzer.progress.total - 1, task=current_task)
    visualization = visualize_analysis(analyzer)
    dump = analyzer.dump()
    analyzer.progress.done(task=current_task)
    analyzer.teardown()
    return visualization, dump, html.unescape(analyzer.progress.log())


@celery.task(name='analyze_id_list')
def analyze_id_list(source, id_list, zoom, query):
    loader = get_loader(source, PUBTRENDS_CONFIG)
    analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
    try:
        ids = analyzer.process_id_list(id_list, zoom, 1, current_task)
        analyzer.analyze_papers(ids, query, current_task)
    finally:
        loader.close_connection()

    analyzer.progress.info('Visualizing', current=analyzer.progress.total - 1, task=current_task)
    visualization = visualize_analysis(analyzer)
    dump = analyzer.dump()
    analyzer.progress.done(task=current_task)
    analyzer.teardown()
    return visualization, dump, html.unescape(analyzer.progress.log())


@celery.task(name='find_paper_async')
def find_paper_async(source, key, value):
    loader = get_loader(source, PUBTRENDS_CONFIG)
    progress = ProgressLogger(total=2)
    loader.set_progress_logger(progress)
    try:
        result = loader.find(key, value)
        progress.info('Done', current=2)
        return result
    finally:
        loader.close_connection()


def get_loader(source, config):
    if source == 'Pubmed':
        return PubmedLoader(config)
    elif source == 'Semantic Scholar':
        return SemanticScholarLoader(config)
    else:
        raise ValueError(f"Unknown source {source}")
