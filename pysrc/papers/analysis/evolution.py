import logging

import community
import networkx as nx
import numpy as np
import pandas as pd

from pysrc.papers.analysis.citations import build_cocit_grouped_df
from pysrc.papers.analysis.graph import build_similarity_graph
from pysrc.papers.analysis.text import compute_comps_tfidf
from pysrc.papers.analysis.topics import merge_components
from pysrc.papers.utils import SEED

logger = logging.getLogger(__name__)


def topic_evolution_analysis(
        df,
        cit_df, cocit_df, bibliographic_coupling_df, texts_similarity,
        cocit_min_threshold,
        topic_min_size, max_topics_number,
        similarity_func,
        evolution_step,
        progress, current=0, task=None
):
    """
    Main method of evolution analysis
    :param df: Full dataframe with papers information
    :param cit_df: Citations dataframe
    :param cocit_df: Cocitations dataframe
    :param bibliographic_coupling_df: Bibliographic coupling dataframe, already filtered by min_threshold
    :param texts_similarity: Texts similarity list
    :param cocit_min_threshold: Min cocitations threshold
    :param topic_min_size: 
    :param max_topics_number: 
    :param similarity_func:
    :param evolution_step: Evolution step
    :param progress:
    :param current:
    :param task:
    :return:
    """
    min_year = int(cocit_df['year'].min())
    max_year = int(cocit_df['year'].max())
    year_range = list(np.arange(max_year, min_year - 1, step=-evolution_step).astype(int))

    # Cannot analyze evolution
    if len(year_range) < 2:
        progress.info(f'Year step is too big to analyze evolution of topics in {min_year} - {max_year}',
                      current=current, task=task)
        return None, None

    progress.info(f'Studying evolution of topics in {min_year} - {max_year}', current=current, task=task)

    logger.debug(f"Topics evolution years: {', '.join([str(year) for year in year_range])}")
    years_processed = 1
    evolution_series = [df['comp']]  # Evolution ends with the latest topic separation
    for i, year in enumerate(year_range[1:]):
        progress.info(f'Processing year {year}', current=current, task=task)
        logger.debug(f'Get ids earlier than year {year}')
        ids_year = set(df.loc[df['year'] <= year]['id'])

        logger.debug('Use only citations earlier than year')
        citations_graph_year = filter_citations_graph(cit_df, ids_year)

        logger.debug('Use only co-citations earlier than year')
        cocit_grouped_df_year = filter_cocit_grouped_df(cocit_df, cocit_min_threshold, year)

        logger.debug('Use bibliographic coupling earlier then year')
        bibliographic_coupling_df_year = filter_bibliographic_coupling_df(bibliographic_coupling_df, ids_year)

        logger.debug('Use similarities for papers earlier then year')
        texts_similarity_year = filter_text_similarities(df, texts_similarity, year)

        progress.info('Building papers similarity graph', current=current, task=task)
        similarity_graph = build_similarity_graph(
            df, texts_similarity_year,
            citations_graph_year,
            cocit_grouped_df_year,
            bibliographic_coupling_df_year,
            process_all_papers=False,  # Dont add all the papers to the graph
        )
        progress.info(f'Built similarity graph - {len(similarity_graph.nodes())} nodes and '
                      f'{len(similarity_graph.edges())} edges',
                      current=current, task=task)

        logger.debug('Compute aggregated similarity')
        for _, _, d in similarity_graph.edges(data=True):
            d['similarity'] = similarity_func(d)

        progress.info('Extracting topics from paper similarity graph', current=current, task=task)
        dendrogram = community.generate_dendrogram(
            similarity_graph, weight='similarity', random_state=SEED
        )
        # Smallest communities
        partition_louvain = dendrogram[0]
        logger.debug(f'Found {len(set(partition_louvain.values()))} components')
        # Reorder and merge small components to 'OTHER'
        p, _ = merge_components(
            partition_louvain, topic_min_size=topic_min_size, max_topics_number=max_topics_number
        )
        evolution_series.append(pd.Series(p))
        years_processed += 1

    year_range = year_range[:years_processed]

    evolution_df = pd.concat(evolution_series, axis=1).rename(
        columns=dict(enumerate(year_range)))
    evolution_df['current'] = evolution_df[max_year]
    evolution_df = evolution_df[list(reversed(list(evolution_df.columns)))]

    # Assign -1 to articles not published yet
    evolution_df = evolution_df.fillna(-1.0)

    evolution_df = evolution_df.reset_index().rename(columns={'index': 'id'})
    evolution_df['id'] = evolution_df['id'].astype(str)
    return evolution_df, year_range


def filter_text_similarities(df, texts_similarity, year):
    texts_similarity_year = texts_similarity.copy()
    for idx in np.flatnonzero(df['year'].apply(int) > year):
        texts_similarity_year[idx] = []  # Empty similarities for recent papers
    return texts_similarity_year


def filter_bibliographic_coupling_df(bibliographic_coupling_df, ids_year):
    bibliographic_coupling_df_year = bibliographic_coupling_df.loc[
        np.logical_and(
            bibliographic_coupling_df['citing_1'].isin(ids_year),
            bibliographic_coupling_df['citing_2'].isin(ids_year)
        )
    ]
    return bibliographic_coupling_df_year


def filter_cocit_grouped_df(cocit_df, cocit_min_threshold, year):
    cocit_grouped_df_year = build_cocit_grouped_df(cocit_df.loc[cocit_df['year'] <= year])
    cocit_grouped_df_year = cocit_grouped_df_year[cocit_grouped_df_year['total'] >= cocit_min_threshold]
    return cocit_grouped_df_year


def filter_citations_graph(cit_df, ids_year):
    citations_graph_year = nx.DiGraph()
    for index, row in cit_df.iterrows():
        v, u = row['id_out'], row['id_in']
        if v in ids_year and u in ids_year:
            citations_graph_year.add_edge(v, u)
    return citations_graph_year


def topic_evolution_descriptions(
        df, evolution_df, year_range, corpus_terms, corpus_counts, size,
        progress, current=0, task=None
):
    # Topic evolution failed, no need to generate keywords
    if evolution_df is None or not year_range:
        return None

    progress.info('Generating evolution topics description by top cited papers',
                  current=current, task=task)
    try:  # Workaround for https://github.com/JetBrains-Research/pubtrends/issues/247
        evolution_kwds = {}
        for col in evolution_df:
            if col in year_range:
                progress.info(f'Generating topics descriptions for year {col}',
                              current=current, task=task)
                if isinstance(col, (int, float)):
                    evolution_df[col] = evolution_df[col].apply(int)
                    comps = evolution_df.groupby(col)['id'].apply(list).to_dict()
                    evolution_kwds[col] = get_evolution_topics_description(
                        df, comps, corpus_terms, corpus_counts,
                        size=size
                    )
        return evolution_kwds
    except Exception as e:
        logger.error('Error while computing evolution description', e)
        return None


def get_evolution_topics_description(df, comps, corpus_terms, corpus_counts, size):
    tfidf = compute_comps_tfidf(df, comps, corpus_counts, ignore_comp=-1)
    kwd = {}
    comp_idx = dict(enumerate([c for c in comps if c != -1]))  # -1 Not yet published
    for comp in comps.keys():
        if comp not in comp_idx:
            # Generate no keywords for '-1' component
            kwd[comp] = ''
            continue

        # Sort indices by tfidf value
        # It might be faster to use np.argpartition instead of np.argsort
        ind = np.argsort(tfidf[comp_idx[comp], :].toarray(), axis=1)

        # Take tokens with the largest tfidf
        kwd[comp_idx[comp]] = [corpus_terms[idx] for idx in ind[0, -size:]]
    return kwd