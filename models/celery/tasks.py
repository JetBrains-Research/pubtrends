import os

from bokeh.embed import components
from celery import Celery, current_task

from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.keypaper.pm_loader import PubmedLoader
from models.keypaper.ss_loader import SemanticScholarLoader
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
        raise Exception(f"Unknown source {source}")

    if not sort:
        sort = 'Most Cited'

    analyzer = KeyPaperAnalyzer(loader)
    # current_task is from @celery.task
    log = analyzer.launch(search_query=terms, id_list=id_list, zoom=zoom, limit=str(amount),
                          sort=SORT_METHODS[sort], task=current_task)

    # Initialize plotter after completion of analysis
    plotter = Plotter(analyzer)

    # Order is important here!
    result = {
        'log': log,
        'n_papers': analyzer.n_papers,
        'n_citations': int(analyzer.df['total'].sum()),
        'n_subtopics': len(analyzer.components),
        'comp_other': analyzer.comp_other,
        'cocitations_clusters': [components(plotter.cocitations_clustering())],
        'component_size_summary': [components(plotter.component_size_summary())],
        'subtopic_timeline_graphs': [components(p) for p in plotter.subtopic_timeline_graphs()],
        'top_cited_papers': [components(plotter.top_cited_papers())],
        'max_gain_papers': [components(plotter.max_gain_papers())],
        'max_relative_gain_papers': [components(plotter.max_relative_gain_papers())],
        'component_ratio': [components(plotter.component_ratio())],
        'papers_stats': [components(plotter.papers_statistics())],
        'clusters_info_message': plotter.clusters_info_message,
        'author_statistics': [components(plotter.author_statistics())],
        'journal_statistics': [components(plotter.journal_statistics())]
        # TODO: this doesn't work
        # 'citations_dynamics': [components(plotter.article_citation_dynamics())],
    }

    # Pass subtopic evolution only if not None
    subtopic_evolution = plotter.subtopic_evolution()
    if subtopic_evolution:
        result['subtopic_evolution'] = [components(subtopic_evolution)]

    return result


@celery.task(name='analyze_paper_async')
def analyze_paper_async(source, key, value):
    if source == 'Pubmed':
        loader = PubmedLoader(PUBTRENDS_CONFIG)
    elif source == 'Semantic Scholar':
        loader = SemanticScholarLoader(PUBTRENDS_CONFIG)
    else:
        raise Exception(f"Unknown source {source}")

    analyzer = KeyPaperAnalyzer(loader)
    log = analyzer.launch_paper(key, value, task=current_task)

    result = {
        'log': log,
        'ids': analyzer.ids
    }

    return result
