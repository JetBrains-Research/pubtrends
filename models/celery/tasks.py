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
    result = {
        'log': html.unescape(log),
        'experimental': PUBTRENDS_CONFIG.run_experimental,
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
        'clusters_info_message': html.unescape(plotter.clusters_info_message),
        'author_statistics': [components(plotter.author_statistics())],
        'journal_statistics': [components(plotter.journal_statistics())]
    }

    # Experimental features
    if PUBTRENDS_CONFIG.run_experimental:
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


def get_top_papers(papers, df, key, n=10):
    citing_papers = map(lambda v: (df[df['id'] == v]['title'].values[0],
                                   df[df['id'] == v][key].values[0]), list(papers))
    return [el[0] for el in sorted(citing_papers, key=lambda x: x[1], reverse=True)[:n]]


def prepare_paper_data(data, source, pid):
    if source == 'Pubmed':
        loader = PubmedLoader(PUBTRENDS_CONFIG)
        url = PUBMED_ARTICLE_BASE_URL + pid
    elif source == 'Semantic Scholar':
        loader = SemanticScholarLoader(PUBTRENDS_CONFIG)
        url = SEMANTIC_SCHOLAR_BASE_URL + pid
    else:
        raise ValueError(f"Unknown source {source}")

    analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
    analyzer.load(data)

    plotter = Plotter()

    # Extract data for the current paper
    sel = analyzer.df[analyzer.df['id'] == pid]
    title = sel['title'].values[0]
    journal = sel['journal'].values[0]
    year = sel['year'].values[0]

    # Trim title to fit in UI
    max_title_length = 100
    trimmed_title = f'{title[:max_title_length]}...' if len(title) > max_title_length else title

    # Generate info about publication year and journal
    if journal == '':
        if year == np.nan:
            citation = ''
        else:
            citation = f'Published in {int(float(year))}'
    else:
        if year == np.nan:
            citation = journal
        else:
            citation = f'{journal} ({int(float(year))})'

    # Estimate related topics for the paper
    related_topics = {}
    for v in analyzer.CG[pid]:
        c = analyzer.df[analyzer.df['id'] == v]['comp'].values[0]
        if c in related_topics:
            related_topics[c] += 1
        else:
            related_topics[c] = 1
    related_topics = map(lambda el: (', '.join([w[0] for w in
                                                analyzer.df_kwd[analyzer.df_kwd['comp'] == el[0]]['kwd'].values[0][
                                                :10]]), el[1]),
                         sorted(related_topics.items(), key=lambda el: el[1], reverse=True))

    # Determine top references (papers that are cited by current), citations (papers that cite current),
    # and co-citations
    top_references = get_top_papers(analyzer.G.successors(pid), analyzer.df, key='pagerank')
    top_citations = get_top_papers(analyzer.G.predecessors(pid), analyzer.df, key='pagerank')

    cocited_papers = map(lambda v: (analyzer.df[analyzer.df['id'] == v]['title'].values[0],
                                    analyzer.CG.edges[pid, v]['weight']), list(analyzer.CG[pid]))
    top_cocited_papers = sorted(cocited_papers, key=lambda x: x[1], reverse=True)[:10]

    result = {
        'title': title,
        'trimmed_title': trimmed_title,
        'authors': sel['authors'].values[0],
        'citation': citation,
        'url': url,
        'source': source,
        'citation_dynamics': [components(plotter.article_citation_dynamics(analyzer.df, str(pid)))],
        'related_topics': related_topics,
        'cocited_papers': top_cocited_papers
    }

    abstract = sel['abstract'].values[0]
    if abstract != '':
        result['abstract'] = abstract

    if len(top_references) > 0:
        result['citing_papers'] = top_references

    if len(top_citations) > 0:
        result['cited_papers'] = top_citations

    return result
