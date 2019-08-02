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


# Tasks will be served by Celery,
# specify task name explicitly to avoid problems with modules
@celery.task(name='analyze_async')
def analyze_async(source, terms):
    if source == 'Pubmed':
        loader = PubmedLoader(PUBTRENDS_CONFIG)
        amount_of_papers = '29 million'
    elif source == 'Semantic Scholar':
        loader = SemanticScholarLoader(PUBTRENDS_CONFIG)
        amount_of_papers = '45 million'
    else:
        raise Exception(f"Unknown source {source}")
    analyzer = KeyPaperAnalyzer(loader)
    # current_task is from @celery.task
    log = analyzer.launch(*terms, task=current_task)

    # Initialize plotter after completion of analysis
    plotter = Plotter(analyzer)

    # Subtopic evolution is ignored for now.
    # Order is important here!
    return {
        'log': log,
        'n_papers': len(analyzer.df),
        'n_citations': int(analyzer.df['total'].sum()),
        'n_subtopics': len(analyzer.components),
        'cocitations_clusters': [components(plotter.cocitations_clustering())],
        'component_size_summary': [components(plotter.component_size_summary())],
        'subtopic_timeline_graphs': [components(p) for p in plotter.subtopic_timeline_graphs()],
        'top_cited_papers': [components(plotter.top_cited_papers())],
        'max_gain_papers': [components(plotter.max_gain_papers())],
        'max_relative_gain_papers': [components(plotter.max_relative_gain_papers())],
        'component_ratio': [components(plotter.component_ratio())],
        'papers_stats': [components(plotter.papers_statistics())],
        'found_papers': str(analyzer.articles_found),
        'number_of_papers': amount_of_papers,
        'clusters_info_message': plotter.clusters_info_message,
        'subtopic_evolution': [components(plotter.subtopic_evolution())],
        'author_statistics': [components(plotter.author_statistics())],
        'journal_statistics': [components(plotter.journal_statistics())]
        # TODO: this doesn't work
        # 'citations_dynamics': [components(plotter.article_citation_dynamics())],
    }
