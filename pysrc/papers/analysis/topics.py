import logging
from collections import Counter

import community
import networkx as nx
import numpy as np
import pandas as pd

from pysrc.papers.analysis.graph import to_weighted_graph
from pysrc.papers.analysis.text import get_frequent_tokens, compute_tfidf
from pysrc.papers.utils import SEED

logger = logging.getLogger(__name__)


def louvain(similarity_graph, similarity_func, topic_min_size, max_topics_number):
    """
    Performs clustering of similarity topics, merging small topics into Other component
    :param similarity_graph: Similarity graph
    :param similarity_func: Function to compute aggregated similarity between nodes in similarity graph
    :param topic_min_size:
    :param max_topics_number:
    :return: comp_partition, comp_sizes
    """
    wsg = to_weighted_graph(similarity_graph, similarity_func)
    logger.debug(f'Similarity graph has {nx.number_connected_components(wsg)} connected components')
    logger.debug('Graph clustering via Louvain community algorithm')
    partition = community.best_partition(
        wsg, weight='weight', random_state=SEED
    )
    logger.debug(f'Best partition {len(set(partition.values()))} components')
    components = set(partition.values())
    sizes = {c1: sum([partition[node] == c1 for node in partition.keys()]) for c1 in components}
    logger.debug(f'Components: {sizes}')
    if len(wsg.edges) > 0:
        modularity = community.modularity(partition, wsg)
        logger.debug(f'Graph modularity (possible range is [-1, 1]): {modularity :.3f}')

    logger.debug('Merge small components')
    similarity_matrix = compute_similarity_matrix(similarity_graph, similarity_func, partition)
    comp_partition, comp_sizes = merge_components(
        partition, similarity_matrix,
        topic_min_size=topic_min_size, max_topics_number=max_topics_number
    )
    return comp_partition, comp_sizes


def compute_similarity_matrix(similarity_graph, similarity_func, partition):
    logger.debug('Computing mean similarity of all edges between topics')
    n_comps = len(set(partition.values()))
    edges = len(similarity_graph.edges)
    sources = [None] * edges
    targets = [None] * edges
    similarities = [0.0] * edges
    i = 0
    for u, v, data in similarity_graph.edges(data=True):
        sources[i] = u
        targets[i] = v
        similarities[i] = similarity_func(data)
        i += 1
    df = pd.DataFrame(partition.items(), columns=['id', 'comp'])
    similarity_df = pd.DataFrame(data={'source': sources, 'target': targets, 'similarity': similarities})
    similarity_topics_df = similarity_df.merge(df, how='left', left_on='source', right_on='id') \
        .merge(df, how='left', left_on='target', right_on='id')
    logger.debug('Calculate mean similarity between for topics')
    mean_similarity_topics_df = \
        similarity_topics_df.groupby(['comp_x', 'comp_y'])['similarity'].mean().reset_index()
    similarity_matrix = np.zeros(shape=(n_comps, n_comps))
    for index, row in mean_similarity_topics_df.iterrows():
        cx, cy = int(row['comp_x']), int(row['comp_y'])
        similarity_matrix[cx, cy] = similarity_matrix[cy, cx] = row['similarity']
    return similarity_matrix


def merge_components(partition, similarity_matrix, topic_min_size, max_topics_number):
    """
    Merge small topics to required number of topics and minimal size, reorder topics by size
    :param partition: Partition, dict paper id -> component
    :param similarity_matrix: Mean similarity between components for partition
    :param topic_min_size: Min number of papers in topic
    :param max_topics_number: Max number of topics
    :return: merged_partition, sorted_comp_sizes
    """
    logger.debug(f'Merging: max {max_topics_number} components with min size {topic_min_size}')
    comp_sizes = Counter(partition.values())
    logger.debug(f'{len(comp_sizes)} comps, comp_sizes: {comp_sizes}')

    merge_index = 1
    merge_order = []
    while len(comp_sizes) > 1 and \
            (len(comp_sizes) > max_topics_number or min(comp_sizes.values()) < topic_min_size):
        # logger.debug(f'{merge_index}. Pick minimal and merge it with the closest by similarity')
        merge_index += 1
        min_comp = min(comp_sizes.keys(), key=lambda c: comp_sizes[c])
        comp_to_merge = max([c for c in partition.values() if c != min_comp],
                            key=lambda c: similarity_matrix[min_comp][c])
        # logger.debug(f'Merging with most similar comp {comp_to_merge}')
        comp_update = min(min_comp, comp_to_merge)
        comp_sizes[comp_update] = comp_sizes[min_comp] + comp_sizes[comp_to_merge]
        if min_comp != comp_update:
            merge_order.append((min_comp, comp_update))
            del comp_sizes[min_comp]
        else:
            merge_order.append((comp_to_merge, comp_update))
            del comp_sizes[comp_to_merge]
        # logger.debug(f'Merged comps: {len(comp_sizes)}, updated comp_sizes: {comp_sizes}')
        for (paper, c) in list(partition.items()):
            if c == min_comp or c == comp_to_merge:
                partition[paper] = comp_update

        # logger.debug('Update similarities')
        for i in range(len(similarity_matrix)):
            similarity_matrix[i, comp_update] = \
                (similarity_matrix[i, min_comp] + similarity_matrix[i, comp_to_merge]) / 2
            similarity_matrix[comp_update, i] = \
                (similarity_matrix[min_comp, i] + similarity_matrix[comp_to_merge, i]) / 2
    logger.debug(f'Merge done in {merge_index} steps.\nOrder: {merge_order}')
    logger.debug('Sorting comps by size descending')
    sorted_components = dict(
        (c, i) for i, c in enumerate(sorted(set(comp_sizes), key=lambda c: comp_sizes[c], reverse=True))
    )
    logger.debug(f'Comps reordering by size: {sorted_components}')
    merged_partition = {paper: sorted_components[c] for paper, c in partition.items()}
    sorted_comp_sizes = Counter(merged_partition.values())

    for k, v in sorted_comp_sizes.items():
        logger.debug(f'Component {k}: {v} ({int(100 * v / len(merged_partition))}%)')
    return merged_partition, sorted_comp_sizes


def get_topics_description(df, comps, corpus_terms, corpus_counts, query, n_words, method='tfidf', ignore_comp=None):
    """
    Get words from abstracts that describe the components best using two methods:
    :param method:
      * 'tfidf' - select words with maximal tfidf across components
      * 'cosine' - closest to the 'ideal' frequency vector - [0, ..., 0, 1, 0, ..., 0] in terms of cosine distance
    """
    logger.debug(f'Generating topics description, ignore_comp={ignore_comp}')
    # Since some of the components may be skipped, use this dict for continuous indexes'
    comp_idx = {c: i for i, c in enumerate(c for c in comps if c != ignore_comp)}
    # In cases with less than 2 significant components, return  frequencies
    if len(comp_idx) < 2:
        comp = list(comp_idx.keys())[0]
        if ignore_comp is None:
            most_frequent = get_frequent_tokens(df, query)
            return {comp: list(sorted(most_frequent.items(), key=lambda kv: kv[1], reverse=True))[:n_words]}
        else:
            most_frequent = get_frequent_tokens(df.loc[df['id'].isin(set(comps[comp]))], query)
            return {comp: list(sorted(most_frequent.items(), key=lambda kv: kv[1], reverse=True))[:n_words],
                    ignore_comp: []}

    # Pass paper indices (for corpus_terms and corpus_counts) instead of paper ids
    comps_ids = {comp: list(np.flatnonzero(df['id'].isin(comp_pids))) for comp, comp_pids in comps.items()}

    if method == 'tfidf':
        result = get_topics_description_tfidf(comps_ids, corpus_terms, corpus_counts, n_words, ignore_comp=ignore_comp)
    elif method == 'cosine':
        result = get_topics_description_cosine(comps_ids, corpus_terms, corpus_counts, n_words, ignore_comp=ignore_comp)
    else:
        raise ValueError(f'Bad method for generating topic descriptions: {method}')

    kwds = [(comp, ','.join([f'{t}:{v:.3f}' for t, v in vs])) for comp, vs in result.items()]
    logger.debug('Description\n' + '\n'.join(f'{comp}: {kwd}' for comp, kwd in kwds))

    return result


def get_topics_description_tfidf(comps, corpus_terms, corpus_counts, n_words, ignore_comp=None):
    """
    Select words from abstracts with maximal tfidf across components
    """
    logger.debug('Compute average terms counts per components')
    # Since some of the components may be skipped, use this dict for continuous indexes
    comp_idx = {c: i for i, c in enumerate(c for c in comps if c != ignore_comp)}
    terms_freqs_per_comp = np.zeros(shape=(len(comp_idx), corpus_counts.shape[1]), dtype=float)
    for comp, comp_ids in comps.items():
        if comp != ignore_comp:  # Not ignored
            terms_freqs_per_comp[comp_idx[comp], :] = \
                np.sum(corpus_counts[comp_ids, :], axis=0) / len(comp_ids)

    tfidf = compute_tfidf(terms_freqs_per_comp)

    logger.debug('Take terms with the largest tfidf for topics')
    result = {}
    for comp, _ in comps.items():
        if comp == ignore_comp:
            result[comp] = []  # Ignored component
            continue

        counter = Counter()
        for i, t in enumerate(corpus_terms):
            counter[t] += tfidf[comp_idx[comp], i]
        # Ignore terms with insignificant frequencies
        result[comp] = [(t, f) for t, f in counter.most_common(n_words) if f > 0]

    return result


def get_topics_description_cosine(comps, corpus_terms, corpus_counts, n_words, ignore_comp=None):
    """
    Select words with the frequency closer that is the closest to the 'ideal' frequency vector
    ([0, ..., 0, 1, 0, ..., 0]) in terms of cosine distance
    """
    logger.debug('Compute average terms counts per components')
    # Since some of the components may be skipped, use this dict for continuous indexes
    comp_idx = {c: i for i, c in enumerate(c for c in comps if c != ignore_comp)}
    terms_freqs_per_comp = np.zeros(shape=(len(comp_idx), corpus_counts.shape[1]), dtype=np.float)
    for comp, comp_ids in comps.items():
        if comp != ignore_comp:  # Not ignored
            terms_freqs_per_comp[comp_idx[comp], :] = \
                np.sum(corpus_counts[comp_ids, :], axis=0)

    # Calculate total number of occurrences for each word
    terms_freqs_total = np.sum(terms_freqs_per_comp, axis=0)

    # Normalize frequency vector for each word to have length of 1
    terms_freqs_norm = np.sqrt(np.diag(terms_freqs_per_comp.T @ terms_freqs_per_comp))
    terms_freqs_per_comp = terms_freqs_per_comp / terms_freqs_norm

    logger.debug('Take frequent terms that have the most descriptive frequency vector for topics')
    # Calculate cosine distance between the frequency vector and [0, ..., 0, 1, 0, ..., 0] for each cluster
    cluster_mask = np.eye(len(comp_idx))
    distance = terms_freqs_per_comp.T @ cluster_mask
    # Add some weight for more frequent terms to get rid of extremely rare ones in the top
    adjusted_distance = distance.T * np.log(terms_freqs_total)

    result = {}
    for comp in comps.keys():
        if comp == ignore_comp:
            result[comp] = []  # Ignored component
            continue

        c = comp_idx[comp]   # Get the continuous index
        cluster_terms_idx = np.argsort(-adjusted_distance[c, :])[:n_words].tolist()
        result[comp] = [(corpus_terms[i], adjusted_distance[c, i]) for i in cluster_terms_idx]

    return result
