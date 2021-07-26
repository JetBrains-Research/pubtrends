import logging
from queue import PriorityQueue

import networkx as nx
import numpy as np
import pandas as pd

from pysrc.papers.analysis.citations import build_cocit_grouped_df
from pysrc.papers.analysis.graph import build_similarity_graph
from pysrc.papers.analysis.topics import get_topics_description, louvain

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
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    year_range = list(np.arange(max_year, min_year - 1, step=-evolution_step).astype(int))

    # Cannot analyze evolution
    if len(year_range) < 2:
        progress.info(f'Year step is too big to analyze evolution of topics in {min_year} - {max_year}',
                      current=current, task=task)
        return None, None

    progress.info(f'Analyzing evolution of topics in {min_year} - {max_year}', current=current, task=task)

    logger.debug(f"Topics evolution years: {year_range}")
    # Evolution starts with the latest topic separation
    evolution_series = [pd.Series(data=list(df['comp']), index=list(df['id']))]
    for year in year_range[1:]:
        logger.debug(f'Processing year {year}')
        df_year = df.loc[df['year'] <= year]
        ids_year = set(df_year['id'])

        logger.debug('Use only citations earlier than year')
        citations_graph_year = filter_citations_graph(cit_df, ids_year)

        logger.debug('Use only co-citations earlier than year')
        cocit_grouped_df_year = filter_cocit_grouped_df(cocit_df, cocit_min_threshold, year)

        logger.debug('Use bibliographic coupling earlier then year')
        bibliographic_coupling_df_year = filter_bibliographic_coupling_df(bibliographic_coupling_df, ids_year)

        logger.debug('Use similarities for papers earlier then year')
        texts_similarity_year = filter_text_similarities(df_year, texts_similarity, year)

        logger.debug('Building papers similarity graph')
        similarity_graph = build_similarity_graph(
            df_year, texts_similarity_year,
            citations_graph_year,
            cocit_grouped_df_year,
            bibliographic_coupling_df_year,
        )
        logger.debug(f'Built similarity graph - {len(similarity_graph.nodes())} nodes and '
                     f'{len(similarity_graph.edges())} edges')

        merged_partition, _, _ = louvain(
            similarity_graph,
            similarity_func=similarity_func,
            topic_min_size=topic_min_size,
            max_topics_number=max_topics_number
        )
        evolution_series.append(pd.Series(merged_partition))

    evolution_df = pd.concat(evolution_series, axis=1)
    evolution_df.columns = year_range  # Set columns
    evolution_df = evolution_df[reversed(evolution_df.columns)]  # Restore ascending order

    # Assign -1 to articles not published yet
    evolution_df = evolution_df.fillna(-1)

    # Correct types
    evolution_df = evolution_df.astype(int)

    evolution_df = evolution_df.reset_index().rename(columns={'index': 'id'})
    evolution_df['id'] = evolution_df['id'].astype(str)

    logger.debug(f'Successfully created evolution_df {list(evolution_df.columns)} for year_range: {year_range}')
    return evolution_df, year_range


def filter_text_similarities(df, texts_similarity, year):
    texts_similarity_year = texts_similarity.copy()
    for idx in np.flatnonzero(df['year'].apply(int) > year):
        texts_similarity_year[idx] = PriorityQueue(maxsize=0)  # Empty similarities for recent papers
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
    if evolution_df is None or not year_range:
        logger.debug('Topic evolution failed, evolution_df is None, no need to generate keywords')
        return None

    progress.info('Generating evolution topics descriptions', current=current, task=task)
    try:  # Workaround for https://github.com/JetBrains-Research/pubtrends/issues/247
        evolution_kwds = {}
        for col in evolution_df:
            if col in year_range:
                logger.debug(f'Generating topics descriptions for year {col}')
                comps = evolution_df[[col, 'id']].groupby(col)['id'].apply(list).to_dict()
                # Component -1 is not published yet, should be ignored
                evolution_kwds[col] = get_topics_description(df, comps, corpus_terms, corpus_counts,
                                                             query=None, n_words=size, ignore_comp=-1)
        logger.debug(f'Successfully generated evolution_kwds')
        return evolution_kwds
    except Exception as e:
        logger.error('Error while computing evolution description', e)
        return None
