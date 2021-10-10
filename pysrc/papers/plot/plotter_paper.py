import logging
from bokeh.embed import components

from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.plot.plotter import Plotter
from pysrc.papers.utils import trim, MAX_TITLE_LENGTH

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

logger = logging.getLogger(__name__)


def get_top_papers_id_title_year(papers, df, key, n=50):
    citing_papers = map(lambda v: (df[df['id'] == v], df[df['id'] == v][key].values[0]), papers)
    return [(el[0]['id'].values[0], el[0]['title'].values[0], el[0]['year'].values[0])
            for el in sorted(citing_papers, key=lambda x: x[1], reverse=True)[:n]]


def prepare_paper_data(data, source):
    loader, url_prefix = Loaders.get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
    analyzer = PapersAnalyzer(loader, PUBTRENDS_CONFIG)
    analyzer.init(data)
    plotter = Plotter()

    logger.debug('Extracting data for the current paper')
    pid = analyzer.df['id'].values[0]  # Analyzed paper is number one
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

    logger.debug('Estimate related topics for the paper')
    sg = analyzer.papers_graph
    if sg.nodes() and sg.has_node(pid):
        related_topics = {}
        for v in sg[pid]:
            c = analyzer.df[analyzer.df['id'] == v]['comp'].values[0]
            if c in related_topics:
                related_topics[c] += 1
            else:
                related_topics[c] = 1
        related_topics = map(
            lambda el: (', '.join(
                [w[0] for w in analyzer.kwd_df[analyzer.kwd_df['comp'] == el[0]]['kwd'].values[0][:10]]
            ), el[1]),
            sorted(related_topics.items(), key=lambda el: el[1], reverse=True)
        )
    else:
        related_topics = None

    # Citations is limited by only the nodes in pub_df
    derivative_papers = get_top_papers_id_title_year(
        analyzer.cit_df.loc[analyzer.cit_df['id_in'] == pid]['id_out'], analyzer.df, key='total'
    )
    prior_papers = get_top_papers_id_title_year(
        analyzer.cit_df.loc[analyzer.cit_df['id_out'] == pid]['id_in'], analyzer.df, key='total'
    )

    logger.debug('Computing aggregated graph')
    sg = analyzer.papers_graph.copy()
    for _, _, d in sg.edges(data=True):
        d['similarity'] = PapersAnalyzer.similarity(d)

    if sg.nodes() and sg.has_node(pid):
        similar_papers = map(
            lambda v: (analyzer.df[analyzer.df['id'] == v]['id'].values[0],
                       analyzer.df[analyzer.df['id'] == v]['title'].values[0],
                       analyzer.df[analyzer.df['id'] == v]['year'].values[0],
                       sg.edges[pid, v]['similarity']),
            list(sg[pid])
        )
        similar_papers = sorted(similar_papers, key=lambda x: x[3], reverse=True)[:50]
    else:
        similar_papers = None

    result = dict(title=title, trimmed_title=trim(title, MAX_TITLE_LENGTH), authors=authors, journal=journal, year=year,
                  topic=topic, doi=doi, mesh=mesh, keywords=keywords, url=url_prefix + pid, source=source,
                  n_papers=len(analyzer.df),
                  citation_dynamics=[components(plotter.paper_citations_per_year(analyzer.df, str(pid)))])
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
