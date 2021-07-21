import logging
from collections import Counter

import community
import networkx as nx
import numpy as np
import pandas as pd
from more_itertools import unique_everseen
from sklearn.cluster import AgglomerativeClustering

from pysrc.papers.analysis.graph import to_weighted_graph
from pysrc.papers.analysis.text import get_frequent_tokens, compute_tfidf

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
    g = to_weighted_graph(similarity_graph, similarity_func)
    logger.debug(f'Similarity graph has {nx.number_connected_components(g)} connected components')
    logger.debug('Graph clustering via Louvain community algorithm')
    partition_louvain = community.best_partition(
        g, weight='weight', random_state=42
    )
    logger.debug(f'Best partition {len(set(partition_louvain.values()))} components')
    components = set(partition_louvain.values())
    sizes = {c1: sum([partition_louvain[node] == c1 for node in partition_louvain.keys()]) for c1 in components}
    logger.debug(f'Components: {sizes}')
    if len(g.edges) > 0:
        modularity = community.modularity(partition_louvain, g)
        logger.debug(f'Graph modularity (possible range is [-1, 1]): {modularity :.3f}')

    logger.debug('Merge small components')
    similarity_matrix = compute_similarity_matrix(g, similarity_func, partition_louvain)
    comp_partition, comp_sizes = merge_components(
        partition_louvain, similarity_matrix,
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


def cluster_and_sort(x, min_cluster_size, max_clusters):
    """
    :param x: object representations (X x Features)
    :param min_cluster_size:
    :param max_clusters:
    :return: List[cluster], Hierarchical dendrogram of splits.
    """
    logger.debug('Looking for an appropriate number of clusters,'
                 f'min_cluster_size={min_cluster_size}, max_clusters={max_clusters}')
    r = min(int(x.shape[0] / min_cluster_size), max_clusters) + 1
    l = 1

    if l >= r - 2:
        return [0] * x.shape[0], None

    prev_min_size = None
    while l < r - 2:
        n_clusters = int((l + r) / 2)
        logger.debug(f'l = {l}; r = {r}; n_clusters = {n_clusters}')
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(x)
        clusters_counter = Counter(model.labels_)
        assert len(clusters_counter.keys()) == n_clusters, "Incorrect clusters number"
        min_size = clusters_counter.most_common()[-1][1]
        # Track previous_min_size to cope with situation with super distant tiny clusters
        if prev_min_size != min_size and min_size < min_cluster_size or n_clusters > max_clusters:
            logger.debug(f'prev_min_size({prev_min_size}) != min_size({min_size}) < {min_cluster_size} or '
                         f'n_clusters = {n_clusters}  > {max_clusters}')
            r = n_clusters + 1
        else:
            l = n_clusters
        prev_min_size = min_size

    logger.debug(f'Number of clusters = {n_clusters}')
    logger.debug(f'Min cluster size = {prev_min_size}')
    logger.debug('Reorder clusters by size descending')
    reorder_map = {c: i for i, (c, _) in enumerate(clusters_counter.most_common())}
    return [reorder_map[c] for c in model.labels_], model.children_


def compute_clusters_dendrogram_children(clusters, children):
    leaves_map = dict(enumerate(clusters))
    nodes_map = {}
    clusters_children = []
    for i, (u, v) in enumerate(children):
        u_cluster = leaves_map[u] if u in leaves_map else nodes_map[u]
        v_cluster = leaves_map[v] if v in leaves_map else nodes_map[v]
        node = len(leaves_map) + i
        if u_cluster is not None and v_cluster is not None:
            if u_cluster != v_cluster:
                nodes_map[node] = None  # Different clusters
                clusters_children.append((u, v, node))
            else:
                nodes_map[node] = u_cluster
        else:
            nodes_map[node] = None  # Different clusters
            clusters_children.append((u, v, node))

    def rwc(v):
        if v in leaves_map:
            return leaves_map[v]
        elif v in nodes_map:
            res = nodes_map[v]
            return res if res is not None else v
        else:
            return v

    # Rename nodes to clusters
    result = [(rwc(u), rwc(v), rwc(n)) for u, v, n in clusters_children]
    logger.debug(f'Clusters based dendrogram children {result}')
    return result


def convert_clusters_dendrogram_to_paths(clusters, children):
    logger.debug('Converting agglomerate clustering clusters dendrogram format to path for visualization')
    paths = [[p] for p in sorted(set(clusters))]
    for i, (u, v, n) in enumerate(children):
        for p in paths:
            if p[i] == u or p[i] == v:
                p.append(n)
            else:
                p.append(p[i])
    logger.debug(f'Paths {paths}')
    logger.debug('Radix sort or paths to ensure no overlaps')
    for i in range(len(children)):
        paths.sort(key=lambda p: p[i])
        # Reorder next level to keep order of previous if possible
        if i != len(children):
            order = dict((v, i) for i, v in enumerate(unique_everseen(p[i + 1] for p in paths)))
            for p in paths:
                p[i + 1] = order[p[i + 1]]
    leaves_order = dict((v, i) for i, v in enumerate(unique_everseen(p[0] for p in paths)))
    return paths, leaves_order
