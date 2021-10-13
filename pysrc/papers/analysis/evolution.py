import logging

import networkx as nx
import numpy as np
import pandas as pd

from pysrc.papers.analysis.citations import build_cocit_grouped_df
from pysrc.papers.analysis.graph import build_papers_graph, to_weighted_graph, sparse_graph
from pysrc.papers.analysis.node2vec import node2vec
from pysrc.papers.analysis.text import texts_embeddings
from pysrc.papers.analysis.topics import get_topics_description, cluster_and_sort

logger = logging.getLogger(__name__)


def topic_evolution_analysis(
        df, cit_df, cocit_df, bibliographic_coupling_df, cocit_min_threshold, similarity_func,
        corpus_counts, corpus_tokens_embedding,
        graph_embedding_factor, text_embedding_factor,
        topic_min_size, max_topics_number,
        evolution_step
):
    """
    Main method of evolution analysis
    :return: evolution_df, year_range
    """
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    year_range = list(np.arange(max_year, min_year - 1, step=-evolution_step).astype(int))

    # Cannot analyze evolution
    if len(year_range) < 2:
        logger.info(f'Year step is too big to analyze evolution of topics in {min_year} - {max_year}')
        return None, None

    logger.info(f'Analyzing evolution of topics in {min_year} - {max_year}')

    logger.debug(f"Topics evolution years: {year_range}")
    # Evolution starts with the latest topic separation
    evolution_series = [pd.Series(data=list(df['comp']), index=list(df['id']))]
    for year in year_range[1:]:
        df_year = df.loc[df['year'] <= year]
        logger.debug(f'Processing {len(df_year)} papers older than year {year}')
        ids_year = set(df_year['id'])
        if text_embedding_factor != 0:
            logger.debug('Filtering text counts and embeddings for papers earlier then year')
            corpus_counts_year = corpus_counts[np.flatnonzero(df['year'] <= year), :]
            logger.debug('Analyzing texts embeddings')
            texts_embeddings_year = texts_embeddings(
                corpus_counts_year, corpus_tokens_embedding
            )
        else:
            texts_embeddings_year = np.zeros(shape=(len(df_year), 0))

        if graph_embedding_factor != 0:
            logger.debug('Use only citations earlier than year')
            cit_df_year = filter_cit_df(cit_df, ids_year)

            logger.debug('Use only co-citations earlier than year')
            cocit_grouped_df_year = filter_cocit_grouped_df(cocit_df, cocit_min_threshold, ids_year)

            logger.debug('Use bibliographic coupling earlier then year')
            bibliographic_coupling_df_year = filter_bibliographic_coupling_df(bibliographic_coupling_df, ids_year)

            logger.debug('Building papers graph')
            papers_graph = build_papers_graph(
                df_year, cit_df_year, cocit_grouped_df_year, bibliographic_coupling_df_year,
            )
            logger.debug(f'Built papers graph - {papers_graph.number_of_nodes()} nodes and '
                         f'{papers_graph.number_of_edges()} edges')
            weighted_similarity_graph = to_weighted_graph(papers_graph, similarity_func)
            gs = sparse_graph(weighted_similarity_graph)
            graph_embeddings_year = node2vec(df_year['id'], gs)
        else:
            graph_embeddings_year = np.zeros(shape=(len(df_year), 0))

        logger.debug('Computing aggregated graph and text embeddings for papers')
        papers_embeddings = np.concatenate(
            (graph_embeddings_year * graph_embedding_factor,
             texts_embeddings_year * text_embedding_factor), axis=1)

        logger.debug('Extracting topics from papers')
        clusters, _ = cluster_and_sort(papers_embeddings, topic_min_size, max_topics_number)
        partition = dict(zip(df_year['id'], clusters))
        evolution_series.append(pd.Series(partition))

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


def filter_bibliographic_coupling_df(bibliographic_coupling_df, ids_year):
    return bibliographic_coupling_df.loc[
        (bibliographic_coupling_df['citing_1'].isin(ids_year)) &
        (bibliographic_coupling_df['citing_2'].isin(ids_year))
        ]


def filter_cocit_grouped_df(cocit_df, cocit_min_threshold, ids_year):
    cocit_df_year = cocit_df.loc[
        (cocit_df['citing'].isin(ids_year)) & (cocit_df['cited_1'].isin(ids_year)) &
        (cocit_df['cited_2'].isin(ids_year))]
    cocit_grouped_df_year = build_cocit_grouped_df(cocit_df_year)
    cocit_grouped_df_year = cocit_grouped_df_year[cocit_grouped_df_year['total'] >= cocit_min_threshold]
    return cocit_grouped_df_year


def filter_cit_df(cit_df, ids_year):
    return cit_df.loc[(cit_df['id_out'].isin(ids_year)) & (cit_df['id_in'].isin(ids_year))]


def topic_evolution_descriptions(
        df, evolution_df, year_range, corpus, corpus_tokens, corpus_counts, size,
        progress, current=0, task=None
):
    if evolution_df is None or not year_range:
        logger.debug('Topic evolution failed, evolution_df is None, no need to generate keywords')
        return None

    progress.info('Generating evolution topics descriptions', current=current, task=task)
    evolution_kwds = {}
    for col in evolution_df:
        if col in year_range:
            logger.debug(f'Generating topics descriptions for year {col}')
            comps = evolution_df[[col, 'id']].groupby(col)['id'].apply(list).to_dict()
            # Component -1 is not published yet, should be ignored
            evolution_kwds[col] = get_topics_description(df, comps, corpus, corpus_tokens, corpus_counts,
                                                         n_words=size, ignore_comp=-1)
    logger.debug(f'Successfully generated evolution_kwds')
    return evolution_kwds
