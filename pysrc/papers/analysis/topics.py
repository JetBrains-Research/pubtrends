import logging
from collections import Counter

import networkx as nx

from pysrc.papers.analysis.text import compute_comps_tfidf, get_frequent_tokens
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
    logger.debug(f'Relations graph has {connected_components} connected components')

    logger.debug('Compute aggregated similarity')
    for _, _, d in similarity_graph.edges(data=True):
        d['similarity'] = similarity_func(d)

    logger.debug('Graph clustering via Louvain community algorithm')
    import community
    dendrogram = community.generate_dendrogram(
        similarity_graph, weight='similarity', random_state=SEED
    )
    # Smallest communities
    partition_louvain = dendrogram[0]
    logger.debug(f'Found {len(set(partition_louvain.values()))} components')
    components = set(partition_louvain.values())
    comp_sizes = {c: sum([partition_louvain[node] == c for node in partition_louvain.keys()]) for c in components}
    logger.debug(f'Components: {comp_sizes}')
    if len(similarity_graph.edges) > 0:
        logger.debug('Calculate modularity for partition')
        modularity = community.modularity(partition_louvain, similarity_graph)
        logger.debug(f'Graph modularity (possible range is [-1, 1]): {modularity :.3f}')

    # Reorder and merge small components to 'OTHER'
    partition, n_components_merged = merge_components(
        partition_louvain,
        topic_min_size=topic_min_size,
        max_topics_number=max_topics_number
    )

    logger.debug('Sorting components by size descending')
    components = set(partition.values())
    comp_sizes = {c: sum([partition[node] == c for node in partition.keys()]) for c in components}
    # Hack to sort map values by key
    keysort = lambda seq: sorted(range(len(seq)), key=seq.__getitem__, reverse=True)
    sorted_comps = list(keysort(list(comp_sizes.values())))
    sort_order = dict(zip(sorted_comps, range(len(components))))
    logger.debug(f'Components reordering by size: {sort_order}')
    sorted_partition = {p: sort_order[c] for p, c in partition.items()}
    sorted_comp_sizes = {c: comp_sizes[sort_order[c]] for c in range(len(comp_sizes))}

    if n_components_merged > 0:
        comp_other = sorted_comps.index(0)  # Other component is 0!
    else:
        comp_other = None
    logger.debug(f'Component OTHER: {comp_other}')

    for k, v in sorted_comp_sizes.items():
        logger.debug(f'Component {k}: {v} ({int(100 * v / len(partition))}%)')

    logger.debug('Update components dendrogram according to merged topics')
    if len(dendrogram) >= 2:
        rename_map = {}
        for pid, v in partition_louvain.items():  # Pid -> smallest community
            if v not in rename_map:
                rename_map[v] = sorted_partition[pid]
        comp_level = {rename_map[k]: v for k, v in dendrogram[1].items() if k in rename_map}

        logger.debug('Add artificial path for OTHER component')
        if comp_other is not None:
            comp_level[comp_other] = -1
            for d in dendrogram[2:]:
                d[-1] = -1
        comp_dendrogram = [comp_level] + dendrogram[2:]
    else:
        comp_dendrogram = []

    return comp_dendrogram, sorted_partition, comp_other, components, sorted_comp_sizes


def merge_components(partition, topic_min_size, max_topics_number):
    logger.debug(f'Merging components to get max {max_topics_number} components into to "Other" component')
    components = set(partition.values())
    comp_sizes = {c: sum([partition[node] == c for node in partition.keys()]) for c in components}
    sorted_comps = sorted(comp_sizes.keys(), key=lambda c: comp_sizes[c], reverse=True)
    # Limit max number of topics
    if len(components) > max_topics_number:
        components_to_merge = set(sorted_comps[max_topics_number - 1:])
    else:
        components_to_merge = set()
    # Merge tiny topics
    for c, csize in comp_sizes.items():
        if csize < topic_min_size:
            components_to_merge.add(c)
    if components_to_merge:
        n_components_merged = len(components_to_merge)
        logger.debug('Reassigning components')
        partition_merged = {}
        new_comps = {}
        ci = 1  # Start with 1, OTHER component is 0
        for node, comp in partition.items():
            if comp in components_to_merge:
                partition_merged[node] = 0  # Other
                continue
            if comp not in new_comps:
                new_comps[comp] = ci
                ci += 1
            partition_merged[node] = new_comps[comp]
        logger.debug(f'Got {len(set(partition_merged.values()))} components')
        return partition_merged, n_components_merged
    else:
        logger.debug('No need to reassign components')
        return partition, 0


def get_topics_description(df, comps, corpus_terms, corpus_counts, query, n_words):
    if len(comps) == 1:
        most_frequent = get_frequent_tokens(df, query)
        return {0: list(sorted(most_frequent.items(), key=lambda kv: kv[1], reverse=True))[:n_words]}

    tfidf = compute_comps_tfidf(df, comps, corpus_counts)
    result = {}
    for comp in comps.keys():
        # Generate no keywords for '-1' component
        if comp == -1:
            result[comp] = ''
            continue

        # Take indices with the largest tfidf
        counter = Counter()
        for i, w in enumerate(corpus_terms):
            counter[w] += tfidf[comp, i]
        # Ignore terms with insignificant frequencies
        result[comp] = [(t, f) for t, f in counter.most_common(n_words) if f > 0]
    return result
