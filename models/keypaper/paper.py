import numpy as np
from bokeh.embed import components

from models.keypaper.analyzer import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.keypaper.plotter import Plotter
from models.keypaper.pm_loader import PubmedLoader
from models.keypaper.ss_loader import SemanticScholarLoader
from models.keypaper.utils import (
    cut_authors_list, trim, preprocess_text,
    PUBMED_ARTICLE_BASE_URL, SEMANTIC_SCHOLAR_BASE_URL)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

MAX_TITLE_LENGTH = 200


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
    analyzer.init(data)

    plotter = Plotter()

    # Extract data for the current paper
    sel = analyzer.df[analyzer.df['id'] == pid]
    title = sel['title'].values[0]
    authors = sel['authors'].values[0]
    journal = sel['journal'].values[0]
    year = sel['year'].values[0]
    topic = sel['comp'].values[0]
    doi = str(sel['doi'].values[0])
    if doi == 'None' or doi == 'nan':
        doi = ''

    # Estimate related topics for the paper
    if analyzer.similarity_graph.nodes() and analyzer.similarity_graph.has_node(pid):
        related_topics = {}
        for v in analyzer.similarity_graph[pid]:
            c = analyzer.df[analyzer.df['id'] == v]['comp'].values[0]
            if c in related_topics:
                related_topics[c] += 1
            else:
                related_topics[c] = 1
        related_topics = map(
            lambda el: (', '.join(
                [w[0] for w in analyzer.df_kwd[analyzer.df_kwd['comp'] == el[0]]['kwd'].values[0][:10]]
            ), el[1]),
            sorted(related_topics.items(), key=lambda el: el[1], reverse=True)
        )
    else:
        related_topics = None

    # Determine top references (papers that are cited by current),
    # citations (papers that cite current), and co-citations
    # Citations graph is limited by only the nodes in pub_df, so not all the nodes might present
    if analyzer.citations_graph.has_node(pid):
        top_references = get_top_papers_id_title(analyzer.citations_graph.successors(pid), analyzer.df, key='pagerank')
        top_citations = get_top_papers_id_title(analyzer.citations_graph.predecessors(pid), analyzer.df, key='pagerank')
    else:
        top_references = top_citations = []

    if analyzer.similarity_graph.nodes() and analyzer.similarity_graph.has_node(pid):
        related_papers = map(
            lambda v: (analyzer.df[analyzer.df['id'] == v]['id'].values[0],
                       analyzer.df[analyzer.df['id'] == v]['title'].values[0],
                       analyzer.similarity_graph.edges[pid, v]['similarity']),
            list(analyzer.similarity_graph[pid])
        )
        related_papers = [(pid, trim(title, MAX_TITLE_LENGTH), url_prefix + pid, cw)
                          for pid, title, cw in sorted(related_papers, key=lambda x: x[2], reverse=True)[:50]]
    else:
        related_papers = None

    result = {
        'title': title,
        'trimmed_title': trim(title, MAX_TITLE_LENGTH),
        'authors': authors,
        'journal': journal,
        'year': year,
        'topic': topic,
        'doi': doi,
        'url': url_prefix + pid,
        'source': source,
        'citation_dynamics': [components(plotter.article_citation_dynamics(analyzer.df, str(pid)))],
    }
    if related_papers:
        result['related_papers'] = related_papers
    if related_topics:
        result['related_topics'] = related_topics

    abstract = sel['abstract'].values[0]
    if abstract != '':
        result['abstract'] = abstract

    if len(top_references) > 0:
        result['citing_papers'] = [(pid, trim(title, MAX_TITLE_LENGTH), url_prefix + pid)
                                   for pid, title in top_references]

    if len(top_citations) > 0:
        result['cited_papers'] = [(pid, trim(title, MAX_TITLE_LENGTH), url_prefix + pid)
                                  for pid, title in top_citations]

    return result


def prepare_papers_data(data, source, comp=None, word=None, author=None, journal=None, papers_list=None):
    loader, url_prefix = get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
    analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
    analyzer.init(data)

    df = analyzer.df.copy()
    # Filter by component
    if comp is not None:
        df = df.loc[df['comp'].astype(int) == comp]
    # Filter by word
    if word is not None:
        df = df.loc[[word.lower() in preprocess_text(f'{t} {a}')
                     for (t, a) in zip(df['title'], df['abstract'])]]
    # Filter by author
    if author is not None:
        # Check if string was trimmed
        if author.endswith('...'):
            author = author[:-3]
            df = df.loc[[any([a.startswith(author) for a in authors]) for authors in df['authors']]]
        else:
            df = df.loc[[author in authors for authors in df['authors']]]

    # Filter by journal
    if journal is not None:
        # Check if string was trimmed
        if journal.endswith('...'):
            journal = journal[:-3]
            df = df.loc[[j.startswith(journal) for j in df['journal']]]
        else:
            df = df.loc[df['journal'] == journal]

    if papers_list == 'top':
        df = df.loc[[pid in analyzer.top_cited_papers for pid in df['id']]]
    if papers_list == 'year':
        df = df.loc[[pid in analyzer.max_gain_papers for pid in df['id']]]
    if papers_list == 'hot':
        df = df.loc[[pid in analyzer.max_rel_gain_papers for pid in df['id']]]

    result = []
    for _, row in df.iterrows():
        pid, title, abstract, authors, journal, year, total, doi = \
            row['id'], row['title'], row['abstract'], row['authors'], row['journal'], \
            row['year'], row['total'], str(row['doi'])
        if doi == 'None' or doi == 'nan':
            doi = ''
        authors = cut_authors_list(authors, limit=2)  # Take only first/last author
        result.append((pid, (trim(title, MAX_TITLE_LENGTH)), authors, url_prefix + pid, trim(journal, 50),
                       year, total, doi))

    # Return list sorted by year
    return sorted(result, key=lambda t: t[5], reverse=True)
