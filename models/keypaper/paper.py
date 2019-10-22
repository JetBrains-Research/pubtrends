import numpy as np
from bokeh.embed import components

from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.keypaper.pm_loader import PubmedLoader
from models.keypaper.ss_loader import SemanticScholarLoader
from models.keypaper.utils import cut_authors_list, trim, PUBMED_ARTICLE_BASE_URL, SEMANTIC_SCHOLAR_BASE_URL
from models.keypaper.visualization import Plotter

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)


def get_loader_and_url_prefix(source, config):
    if source == 'Pubmed':
        loader = PubmedLoader(config)
        url_prefix = PUBMED_ARTICLE_BASE_URL
    elif source == 'Semantic Scholar':
        loader = SemanticScholarLoader(config)
        url_prefix = SEMANTIC_SCHOLAR_BASE_URL
    else:
        raise ValueError(f"Unknown source {source}")
    return loader, url_prefix


def get_top_papers_id_title(papers, df, key, n=50):
    citing_papers = map(lambda v: (df[df['id'] == v], df[df['id'] == v][key].values[0]), list(papers))
    return [(el[0]['id'].values[0], el[0]['title'].values[0])
            for el in sorted(citing_papers, key=lambda x: x[1], reverse=True)[:n]]


def prepare_paper_data(data, source, pid):
    loader, url_prefix = get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
    analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
    analyzer.load(data)

    plotter = Plotter()

    # Extract data for the current paper
    sel = analyzer.df[analyzer.df['id'] == pid]
    title = sel['title'].values[0]
    journal = sel['journal'].values[0]
    year = sel['year'].values[0]

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
        top_references = get_top_papers_id_title(analyzer.G.successors(pid), analyzer.df, key='pagerank')
        top_citations = get_top_papers_id_title(analyzer.G.predecessors(pid), analyzer.df, key='pagerank')
    else:
        top_references = top_citations = []

    cocited_papers = map(lambda v: (analyzer.df[analyzer.df['id'] == v]['id'].values[0],
                                    analyzer.df[analyzer.df['id'] == v]['title'].values[0],
                                    analyzer.CG.edges[pid, v]['weight']), list(analyzer.CG[pid]))
    top_cocited_papers = sorted(cocited_papers, key=lambda x: x[1], reverse=True)[:50]

    result = {
        'title': title,
        'trimmed_title': trim(title, 200),
        'authors': sel['authors'].values[0],
        'citation': citation,
        'url': url_prefix + pid,
        'source': source,
        'citation_dynamics': [components(plotter.article_citation_dynamics(analyzer.df, str(pid)))],
        'related_topics': related_topics,
        'cocited_papers': [(pid, trim(title, 200), url_prefix + pid, cw) for pid, title, cw in top_cocited_papers]
    }

    abstract = sel['abstract'].values[0]
    if abstract != '':
        result['abstract'] = abstract

    if len(top_references) > 0:
        result['citing_papers'] = [(pid, trim(title, 200), url_prefix + pid) for pid, title in top_references]

    if len(top_citations) > 0:
        result['cited_papers'] = [(pid, trim(title, 200), url_prefix + pid) for pid, title in top_citations]

    return result


def prepare_papers_data(data, source, comp=None, words=None, author=None, journal=None):
    loader, url_prefix = get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
    analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
    analyzer.load(data)

    df = analyzer.df.copy()
    # Filter by component
    if comp is not None:
        df = df.loc[df['comp'].astype(int) == comp]
    # Filter by words
    if words is not None and len(words) > 0:
        df = df.loc[[all([w in title for w in words]) for title in df['title']]]
    # Filter by author
    if author is not None:
        df = df.loc[[author in authors for authors in df['authors']]]
    # Filter by journal
    if journal is not None:
        df = df.loc[df['journal'] == journal]

    result = []
    for _, row in df.iterrows():
        pid, title, abstract, authors, journal, year \
            = row['id'], row['title'], row['abstract'], row['authors'], row['journal'], row['year']
        authors = cut_authors_list(authors, limit=2)  # Take only first/last author
        result.append((pid, (trim(title, 200)), authors, url_prefix + pid, trim(journal, 50), year))

    # Return list sorted by year
    return sorted(result, key=lambda t: t[5], reverse=True)
