import logging
from collections import Counter

import networkx as nx
import numpy as np

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
