from bokeh.embed import components

from pysrc.papers.analyzer import KeyPaperAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.plot.plotter import Plotter
from pysrc.papers.utils import (
    trim, build_corpus, MAX_TITLE_LENGTH)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

def get_top_papers_id_title(papers, df, key, n=50):
    citing_papers = map(lambda v: (df[df['id'] == v], df[df['id'] == v][key].values[0]), list(papers))
    return [(el[0]['id'].values[0], el[0]['title'].values[0])
            for el in sorted(citing_papers, key=lambda x: x[1], reverse=True)[:n]]


def prepare_paper_data(data, source, pid):
    loader, url_prefix = Loaders.get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
    analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
    analyzer.init(data)

    plotter = Plotter()

    # Extract data for the current paper
    sel = analyzer.df[analyzer.df['id'] == pid]
    title = sel['title'].values[0]
    authors = sel['authors'].values[0]
    journal = sel['journal'].values[0]
    year = sel['year'].values[0]
    topic = sel['comp'].values[0] + 1
    doi = str(sel['doi'].values[0])
    mesh = str(sel['mesh'].values[0])
    keywords = str(sel['keywords'].values[0])
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
        related_papers = [[pid, trim(title, MAX_TITLE_LENGTH), url_prefix + pid, f'{similarity:.3f}']
                          for pid, title, similarity in sorted(related_papers, key=lambda x: x[2], reverse=True)[:50]]
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
        'mesh': mesh,
        'keywords': keywords,
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
    loader, url_prefix = Loaders.get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
    analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
    analyzer.init(data)

    df = analyzer.df.copy()
    # Filter by component
    if comp is not None:
        df = df.loc[df['comp'].astype(int) == comp]
    # Filter by word
    if word is not None:
        corpus = build_corpus(df)
        df = df.loc[[word.lower() in text for text in corpus]]
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
        pid, title, authors, journal, year, total, doi, topic = \
            row['id'], row['title'], row['authors'], row['journal'], \
            row['year'], row['total'], str(row['doi']), row['comp'] + 1
        if doi == 'None' or doi == 'nan':
            doi = ''
        # Don't trim or cut anything here, because this information can be exported
        result.append((pid, title, authors, url_prefix + pid, journal, year, total, doi, topic))

    # Return list sorted by year
    return sorted(result, key=lambda t: t[5], reverse=True)
