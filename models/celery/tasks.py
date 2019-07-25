import os
from bokeh.embed import components
from celery import Celery, current_task

from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.pm_loader import PubmedLoader
from models.keypaper.ss_loader import SemanticScholarLoader
from models.keypaper.visualization import Plotter
from models.keypaper.config import PubtrendsConfig


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
        amount_of_papers = '20 million'
    elif source == 'Semantic Scholar':
        loader = SemanticScholarLoader(PUBTRENDS_CONFIG)
        amount_of_papers = '45 million'
    else:
        raise Exception(f"Unknown source {source}")
    analyzer = KeyPaperAnalyzer(loader)
    plotter = Plotter(analyzer)
    # current_task is from @celery.task
    log = analyzer.launch(*terms, task=current_task)

    # Subtopic evolution is ignored for now.
    # Order is important here!
    return {
        'log': log,
        'chord_cocitations': [components(plotter.chord_diagram_components())],
        'component_size_summary': [components(plotter.component_size_summary())],
        'subtopic_timeline_graphs': [components(p) for p in plotter.subtopic_timeline_graphs()],
        'top_cited_papers': [components(plotter.top_cited_papers())],
        'max_gain_papers': [components(plotter.max_gain_papers())],
        'max_relative_gain_papers': [components(plotter.max_relative_gain_papers())],
        'papers_stats': [components(plotter.papers_statistics())],
        'founded_papers': str(loader.articles_found),
        'number_of_papers': amount_of_papers
        # TODO: this doesn't work
        # 'citations_dynamics': [components(plotter.article_citation_dynamics())],
    }
