import numpy as np
from bokeh.embed import components

from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.keypaper.pm_loader import PubmedLoader
from models.keypaper.ss_loader import SemanticScholarLoader
from models.keypaper.utils import PUBMED_ARTICLE_BASE_URL, SEMANTIC_SCHOLAR_BASE_URL
from models.keypaper.visualization import Plotter

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)


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

    # Determine top references (papers that are cited by current),
    # citations (papers that cite current), and co-citations
    # Citations graph is limited by only the nodes in pub_df, so not all the nodes might present
    if analyzer.G.has_node(pid):
        top_references = get_top_papers(analyzer.G.successors(pid), analyzer.df, key='pagerank')
        top_citations = get_top_papers(analyzer.G.predecessors(pid), analyzer.df, key='pagerank')
    else:
        top_references = top_citations = []

    cocited_papers = map(lambda v: (analyzer.df[analyzer.df['id'] == v]['title'].values[0],
                                    analyzer.CG.edges[pid, v]['weight']), list(analyzer.CG[pid]))
    top10_cocited_papers = sorted(cocited_papers, key=lambda x: x[1], reverse=True)[:10]

    result = {
        'title': title,
        'trimmed_title': trimmed_title,
        'authors': sel['authors'].values[0],
        'citation': citation,
        'url': url,
        'source': source,
        'citation_dynamics': [components(plotter.article_citation_dynamics(analyzer.df, str(pid)))],
        'related_topics': related_topics,
        'cocited_papers': top10_cocited_papers
    }

    abstract = sel['abstract'].values[0]
    if abstract != '':
        result['abstract'] = abstract

    if len(top_references) > 0:
        result['citing_papers'] = top_references

    if len(top_citations) > 0:
        result['cited_papers'] = top_citations

    return result
