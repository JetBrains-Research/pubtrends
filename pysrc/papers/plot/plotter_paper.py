import logging
import json
from scipy.spatial import distance

from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.plot.plotter import Plotter, components_list, TOPIC_KEYWORDS
from pysrc.papers.utils import trim, MAX_TITLE_LENGTH, topics_palette

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

logger = logging.getLogger(__name__)


def get_top_papers_id_title_year_cited_topic(papers, df, n=50):
    top_papers = map(lambda v: (df[df['id'] == v], df[df['id'] == v]['total'].values[0]), papers)
    return [(el[0]['id'].values[0],
             el[0]['title'].values[0],
             el[0]['year'].values[0],
             el[0]['total'].values[0],
             el[0]['comp'].values[0])
            for el in sorted(top_papers, key=lambda x: x[1], reverse=True)[:n]]


def prepare_paper_data(data, source, pid):
    loader, url_prefix = Loaders.get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
    analyzer = PapersAnalyzer(loader, PUBTRENDS_CONFIG)
    analyzer.init(data)

    logger.debug('Extracting data for the current paper')
    # Use pid if is given or show the very first paper
    pid = pid or analyzer.df['id'].values[0]
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

    topic_tags = ','.join(
        [w[0] for w in analyzer.kwd_df[analyzer.kwd_df['comp'] == topic - 1]['kwd'].values[0][:TOPIC_KEYWORDS]]
    )

    logger.debug('Computing most cited prior and derivative papers')
    derivative_papers = get_top_papers_id_title_year_cited_topic(
        analyzer.cit_df.loc[analyzer.cit_df['id_in'] == pid]['id_out'], analyzer.df
    )
    prior_papers = get_top_papers_id_title_year_cited_topic(
        analyzer.cit_df.loc[analyzer.cit_df['id_out'] == pid]['id_in'], analyzer.df
    )

    logger.debug('Computing most similar papers')
    if len(analyzer.df) > 1:
        indx = {t: i for i, t in enumerate(analyzer.df['id'])}
        similar_papers = map(
            lambda v: (analyzer.df[analyzer.df['id'] == v]['id'].values[0],
                       analyzer.df[analyzer.df['id'] == v]['title'].values[0],
                       analyzer.df[analyzer.df['id'] == v]['year'].values[0],
                       analyzer.df[analyzer.df['id'] == v]['total'].values[0],
                       1 / (1 + distance.euclidean(analyzer.papers_embeddings[indx[pid], :],
                                                   analyzer.papers_embeddings[indx[v], :])),
                       analyzer.df[analyzer.df['id'] == v]['comp'].values[0] + 1),
            [p for p in analyzer.df['id'] if p != pid]
        )
        similar_papers = sorted(similar_papers, key=lambda x: x[4], reverse=True)[:50]
    else:
        similar_papers = None

    result = dict(title=title,
                  trimmed_title=trim(title, MAX_TITLE_LENGTH),
                  authors=authors,
                  journal=journal,
                  year=year,
                  doi=doi,
                  mesh=mesh,
                  url=url_prefix + pid, source=source,
                  topic=topic,
                  keywords=keywords,
                  n_papers=len(analyzer.df),
                  topics_palette=topics_palette(analyzer.df),
                  topic_tags=topic_tags,
                  citation_dynamics=components_list(Plotter._plot_paper_citations_per_year(analyzer.df, str(pid))))

    if similar_papers:
        result['similar_papers'] = [(pid, trim(title, MAX_TITLE_LENGTH), url_prefix + pid,
                                     year, cited, f'{similarity:.3f}', topic)
                                    for pid, title, year, cited, similarity, topic in similar_papers]

    abstract = sel['abstract'].values[0]
    if abstract != '':
        result['abstract'] = abstract

    if len(prior_papers) > 0:
        result['prior_papers'] = [(pid, trim(title, MAX_TITLE_LENGTH), url_prefix + pid, year, cited, topic + 1)
                                  for pid, title, year, cited, topic in prior_papers]

    if len(derivative_papers) > 0:
        result['derivative_papers'] = [(pid, trim(title, MAX_TITLE_LENGTH), url_prefix + pid, year, cited, topic + 1)
                                       for pid, title, year, cited, topic in derivative_papers]

    logger.debug('Papers graph')
    if len(analyzer.df) > 1:
        result['papers_graph'] = components_list(
            Plotter._plot_papers_graph(
                source, analyzer.sparse_papers_graph, analyzer.df,
                analyzed_pid=pid, topics_tags=analyzer.topics_description
            ))

    return result
