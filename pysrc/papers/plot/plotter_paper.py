from bokeh.embed import components

from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.plot.plotter import Plotter
from pysrc.papers.utils import trim, MAX_TITLE_LENGTH

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)


def get_top_papers_id_title_year(papers, df, key, n=50):
    citing_papers = map(lambda v: (df[df['id'] == v], df[df['id'] == v][key].values[0]), papers)
    return [(el[0]['id'].values[0], el[0]['title'].values[0], el[0]['year'].values[0])
            for el in sorted(citing_papers, key=lambda x: x[1], reverse=True)[:n]]


def prepare_paper_data(data, source, pid):
    loader, url_prefix = Loaders.get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
    analyzer = PapersAnalyzer(loader, PUBTRENDS_CONFIG)
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

    # Citations graph is limited by only the nodes in pub_df
    if analyzer.citations_graph.has_node(pid):
        derivative_papers = get_top_papers_id_title_year(
            analyzer.citations_graph.predecessors(pid), analyzer.df, key='pagerank'
        )
        prior_papers = get_top_papers_id_title_year(
            analyzer.citations_graph.successors(pid), analyzer.df, key='pagerank'
        )
    else:
        prior_papers = derivative_papers = []

    if analyzer.similarity_graph.nodes() and analyzer.similarity_graph.has_node(pid):
        similar_papers = map(
            lambda v: (analyzer.df[analyzer.df['id'] == v]['id'].values[0],
                       analyzer.df[analyzer.df['id'] == v]['title'].values[0],
                       analyzer.df[analyzer.df['id'] == v]['year'].values[0],
                       analyzer.similarity_graph.edges[pid, v]['similarity']),
            list(analyzer.similarity_graph[pid])
        )
        similar_papers = sorted(similar_papers, key=lambda x: x[3], reverse=True)[:50]
    else:
        similar_papers = None

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
        'citation_dynamics': [components(plotter.paper_citations_per_year(analyzer.df, str(pid)))],
    }
    if similar_papers:
        result['similar_papers'] = [(pid, trim(title, MAX_TITLE_LENGTH), url_prefix + pid, year, f'{similarity:.3f}')
                                    for pid, title, year, similarity in similar_papers]

    if related_topics:
        result['related_topics'] = related_topics

    abstract = sel['abstract'].values[0]
    if abstract != '':
        result['abstract'] = abstract

    if len(prior_papers) > 0:
        result['prior_papers'] = [(pid, trim(title, MAX_TITLE_LENGTH), url_prefix + pid, year)
                                  for pid, title, year in prior_papers]

    if len(derivative_papers) > 0:
        result['derivative_papers'] = [(pid, trim(title, MAX_TITLE_LENGTH), url_prefix + pid, year)
                                       for pid, title, year in derivative_papers]

    return result
