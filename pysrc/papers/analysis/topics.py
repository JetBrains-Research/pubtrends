import logging
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from pysrc.papers.analysis.text import get_frequent_tokens, compute_tfidf
from pysrc.papers.utils import SEED

logger = logging.getLogger(__name__)


def topic_analysis(similarity_graph, similarity_func, topic_min_size, max_topics_number):
    """
    Performs clustering of similarity topics, merging small topics into Other component
    :param similarity_graph: Similarity graph
    :param similarity_func: Function to compute hybrid similarity between nodes in similarity graph
    :param topic_min_size:
    :param max_topics_number:
    :return: dendrogram, sorted_partition, comp_other, components, sorted_comp_sizes
    """
    connected_components = nx.number_connected_components(similarity_graph)
    logger.debug(f'Similarity graph has {connected_components} connected components')

    logger.debug('Compute aggregated similarity')
    for _, _, d in similarity_graph.edges(data=True):
        d['similarity'] = similarity_func(d)

    logger.debug('Graph clustering via Louvain community algorithm')
    import community
    dendrogram = community.generate_dendrogram(
        similarity_graph, weight='similarity', random_state=SEED
    )

    # Smallest communities by the Louvain algorithm
    logger.debug(f'Found {len(set(dendrogram[0].values()))} components')
    if len(similarity_graph.edges) > 0:
        modularity = community.modularity(dendrogram[0], similarity_graph)
        logger.debug(f'Graph modularity (possible range is [-1, 1]): {modularity :.3f}')

    # Merge small topics
    similarity_matrix = compute_similarity_matrix(similarity_graph, similarity_func, dendrogram[0])
    dendrogram, comp_sizes = merge_components(
        dendrogram, similarity_matrix,
        topic_min_size=topic_min_size, max_topics_number=max_topics_number
    )
    return dendrogram[0], dendrogram[1:], comp_sizes


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


def merge_components(dendrogram, similarity_matrix, topic_min_size, max_topics_number):
    """
    Merge small topics to required number of topics and minimal size, reorder topics by size
    :param dendrogram: Louvain clustering dendrogram
    :param similarity_matrix: Mean similarity between components for partition
    :param topic_min_size: Min number of papers in topic
    :param max_topics_number: Max number of topics
    :return: updated dendrogram, sorted_comp_sizes
    """
    logger.debug(f'Merging: max {max_topics_number} components with min size {topic_min_size}')
    partition = dendrogram[0]
    comp_sizes = {c: sum([partition[paper] == c for paper in partition.keys()])
                  for c in (set(partition.values()))}
    logger.debug(f'Dendrogram 1+ {dendrogram[1:]}')
    logger.debug(f'{len(comp_sizes)} comps, comp_sizes: {comp_sizes}')

    merge_index = 1
    while len(comp_sizes) > 1 and \
            (len(comp_sizes) > max_topics_number or min(comp_sizes.values()) < topic_min_size):
        logger.debug(f'{merge_index}. Pick minimal and merge it with the closest by dendrogram and similarity')
        merge_index += 1
        min_comp = min(comp_sizes.keys(), key=lambda c: comp_sizes[c])
        if len(dendrogram) > 1:
            min_topic_parent = dendrogram[1][min_comp]
            logger.debug(f'Min comp {min_comp} size {comp_sizes[min_comp]} parent {min_topic_parent}')
            comps_same_parent = [c for c, p in dendrogram[1].items() if c != min_comp and p == min_topic_parent]
        else:
            comps_same_parent = []
        if comps_same_parent:
            comp_to_merge = max(comps_same_parent, key=lambda c: similarity_matrix[min_comp][c])
            logger.debug(f'Merging with most similar comp {comp_to_merge} same parent')
        else:
            comp_to_merge = max([c for c in partition.values() if c != min_comp],
                                key=lambda c: similarity_matrix[min_comp][c])
            logger.debug(f'Merging with most similar comp {comp_to_merge}')
        comp_update = min(min_comp, comp_to_merge)
        comp_sizes[comp_update] = comp_sizes[min_comp] + comp_sizes[comp_to_merge]
        if min_comp != comp_update:
            del comp_sizes[min_comp]
        else:
            del comp_sizes[comp_to_merge]
        logger.debug(f'Merged comps: {len(comp_sizes)}, updated comp_sizes: {comp_sizes}')
        for (paper, c) in list(partition.items()):
            if c == min_comp or c == comp_to_merge:
                partition[paper] = comp_update

        logger.debug('Update dendrogram')
        for i in range(1, len(dendrogram)):
            prev_level_values = set(dendrogram[i - 1].values())
            for c in list(dendrogram[i].keys()):
                if c not in prev_level_values:
                    del dendrogram[i][c]
        logger.debug('Update similarities')
        for i in range(len(similarity_matrix)):
            similarity_matrix[i, comp_update] = \
                (similarity_matrix[i, min_comp] + similarity_matrix[i, comp_to_merge]) / 2
            similarity_matrix[comp_update, i] = \
                (similarity_matrix[min_comp, i] + similarity_matrix[comp_to_merge, i]) / 2

    logger.debug('Sorting comps by size descending')
    sorted_components = dict(
        (c, i) for i, c in enumerate(sorted(set(comp_sizes), key=lambda c: comp_sizes[c], reverse=True))
    )
    logger.debug(f'Comps reordering by size: {sorted_components}')
    dendrogram[0] = {paper: sorted_components[c] for paper, c in dendrogram[0].items()}
    if len(dendrogram) > 1:
        dendrogram[1] = {sorted_components[c]: p for c, p in dendrogram[1].items()}
    sorted_comp_sizes = {c: sum([dendrogram[0][p] == c for p in dendrogram[0].keys()])
                         for c in set(dendrogram[0].values())}

    for k, v in sorted_comp_sizes.items():
        logger.debug(f'Component {k}: {v} ({int(100 * v / len(dendrogram[0]))}%)')
    return dendrogram, sorted_comp_sizes


def get_topics_description(df, comps, corpus_terms, corpus_counts, query, n_words, ignore_comp=None):
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

    logger.debug('Compute average terms counts per components')
    # Since some of the components may be skipped, use this dict for continuous indexes
    comp_idx = {c: i for i, c in enumerate(c for c in comps if c != ignore_comp)}
    terms_freqs_per_comp = np.zeros(shape=(len(comp_idx), corpus_counts.shape[1]), dtype=np.float)
    for comp, comp_pids in comps.items():
        if comp != ignore_comp:  # Not ignored
            terms_freqs_per_comp[comp_idx[comp], :] = \
                np.sum(corpus_counts[np.flatnonzero(df['id'].isin(comp_pids)), :], axis=0) / len(comp_pids)

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

    kwds = [(comp, ','.join([f'{t}:{v:.3f}' for t, v in vs])) for comp, vs in result.items()]
    logger.debug('Description\n' + '\n'.join(f'{comp}: {kwd}' for comp, kwd in kwds))

    return result
