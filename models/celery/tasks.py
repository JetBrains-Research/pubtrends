import html
import os

import numpy as np
from bokeh.embed import components
from celery import Celery, current_task

from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.keypaper.pm_loader import PubmedLoader
from models.keypaper.progress_logger import ProgressLogger
from models.keypaper.ss_loader import SemanticScholarLoader
from models.keypaper.utils import PUBMED_ARTICLE_BASE_URL, SEMANTIC_SCHOLAR_BASE_URL
from models.keypaper.visualization import Plotter

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379'),
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379')

# Configure Celery
celery = Celery("tasks", backend=CELERY_RESULT_BACKEND, broker=CELERY_BROKER_URL)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)
SORT_METHODS = {'Most Cited': 'citations', 'Most Relevant': 'relevance', 'Most Recent': 'year'}


# Tasks will be served by Celery,
# specify task name explicitly to avoid problems with modules
@celery.task(name='analyze_topic_async')
def analyze_topic_async(source, terms=None, id_list=None, zoom=None, sort='Most Cited', amount=None):
    if source == 'Pubmed':
        loader = PubmedLoader(PUBTRENDS_CONFIG)
    elif source == 'Semantic Scholar':
        loader = SemanticScholarLoader(PUBTRENDS_CONFIG)
    else:
        raise ValueError(f"Unknown source {source}")

    if not sort:
        sort = 'Most Cited'

    analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
    # current_task is from @celery.task
    log = analyzer.launch(search_query=terms, id_list=id_list, zoom=zoom, limit=str(amount),
                          sort=SORT_METHODS[sort], task=current_task)

    # Initialize plotter after completion of analysis
    plotter = Plotter(analyzer=analyzer)

    # Order is important here!
    paper_statistics, zoom_out_callback = plotter.papers_statistics_and_zoom_out_callback()
    result = {
        'log': html.unescape(log),
        'experimental': PUBTRENDS_CONFIG.experimental,
        'n_papers': analyzer.n_papers,
        'n_citations': int(analyzer.df['total'].sum()),
        'n_subtopics': len(analyzer.components),
        'comp_other': analyzer.comp_other,
        'cocitations_clusters': [components(plotter.cocitations_clustering())],
        'component_size_summary': [components(plotter.component_size_summary())],
        'component_years_summary_boxplots': [components(plotter.component_years_summary_boxplots())],
        'subtopics_infos_and_zoom_in_callbacks':
            [(components(p), zoom_in_callback) for
             (p, zoom_in_callback) in plotter.subtopics_infos_and_zoom_in_callbacks()],
        'top_cited_papers': [components(plotter.top_cited_papers())],
        'max_gain_papers': [components(plotter.max_gain_papers())],
        'max_relative_gain_papers': [components(plotter.max_relative_gain_papers())],
        'component_sizes': plotter.component_sizes(),
        'component_ratio': [components(plotter.component_ratio())],
        'papers_stats': [components(paper_statistics)],
        'papers_zoom_out_callback': zoom_out_callback,
        'clusters_info_message': html.unescape(plotter.clusters_info_message),
        'author_statistics': plotter.author_statistics(),
        'journal_statistics': plotter.journal_statistics()
    }

    # Experimental features
    if PUBTRENDS_CONFIG.experimental:
        subtopic_evolution = plotter.subtopic_evolution()
        # Pass subtopic evolution only if not None
        if subtopic_evolution:
            result['subtopic_evolution'] = [components(subtopic_evolution)]

    return result, analyzer.dump()


@celery.task(name='find_paper_async')
def find_paper_async(source, key, value):
    if source == 'Pubmed':
        loader = PubmedLoader(PUBTRENDS_CONFIG)
    elif source == 'Semantic Scholar':
        loader = SemanticScholarLoader(PUBTRENDS_CONFIG)
    else:
        raise ValueError(f"Unknown source {source}")

    loader.set_logger(ProgressLogger(total=1))

    return loader.find(key, value)
