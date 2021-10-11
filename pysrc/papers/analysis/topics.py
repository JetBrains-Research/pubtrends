import logging
from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from pysrc.papers.analysis.text import get_frequent_tokens

logger = logging.getLogger(__name__)


def compute_topics_similarity_matrix(papers_embeddings, comps):
    logger.debug('Computing mean cosine similarity between topics embeddings')
    n_comps = len(set(comps))
    cos_similarities = cosine_similarity(papers_embeddings)
    similarity_matrix = np.zeros(shape=(n_comps, n_comps))
    for i in range(n_comps):
        i_indx = np.flatnonzero([c == i for c in comps])
        for j in range(i, n_comps):
            j_indx = np.flatnonzero([c == j for c in comps])
            mean_similarity = np.mean(cos_similarities[np.flatnonzero(cos_similarities[i_indx, j_indx])])
            similarity_matrix[i, j] = similarity_matrix[j, i] = mean_similarity
    return similarity_matrix


def cluster_and_sort(x, min_cluster_size, max_clusters):
    """
    :param x: object representations (X x Features)
    :param min_cluster_size:
    :param max_clusters:
    :return: List[cluster], Hierarchical dendrogram of splits.
    """
    logger.debug('Looking for an appropriate number of clusters,'
                 f'min_cluster_size={min_cluster_size}, max_clusters={max_clusters}')
    if x.shape[1] == 0:
        return [0] * x.shape[0], None
    r = min(int(x.shape[0] / min_cluster_size), max_clusters) + 1
    l = 1

    if l >= r - 2:
        return [0] * x.shape[0], None

    prev_min_size = None
    while l < r - 2:
        n_clusters = int((l + r) / 2)
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(x)
        clusters_counter = Counter(model.labels_)
        min_size = clusters_counter.most_common()[-1][1]
        logger.debug(f'l={l}, r={r}, n_clusters={n_clusters}, min_cluster_size={min_cluster_size}, '
                     f'prev_min_size={prev_min_size}, min_size={min_size}')
        if min_size < min_cluster_size:
            if prev_min_size is not None and min_size <= prev_min_size:
                break
            r = n_clusters + 1
        else:
            l = n_clusters
        prev_min_size = min_size

    logger.debug(f'Number of clusters = {n_clusters}')
    logger.debug(f'Min cluster size = {prev_min_size}')
    logger.debug('Reorder clusters by size descending')
    reorder_map = {c: i for i, (c, _) in enumerate(clusters_counter.most_common())}
    return [reorder_map[c] for c in model.labels_], model.children_


def get_topics_description(df, comps, corpus_terms, corpus_counts, stems_map, n_words, ignore_comp=None):
    """
    Get words from abstracts that describe the components the best way
    using closest to the 'ideal' frequency vector - [0, ..., 0, 1, 0, ..., 0] in terms of cosine distance
    """
    logger.debug(f'Generating topics description, ignore_comp={ignore_comp}')
    # Since some of the components may be skipped, use this dict for continuous indexes'
    comp_idx = {c: i for i, c in enumerate(c for c in comps if c != ignore_comp)}
    # In cases with less than 2 components, return frequencies
    if len(comp_idx) < 2:
        comp = list(comp_idx.keys())[0]
        if ignore_comp is None:
            most_frequent = get_frequent_tokens(df, stems_map)
            return {comp: list(sorted(most_frequent.items(), key=lambda kv: kv[1], reverse=True))[:n_words]}
        else:
            most_frequent = get_frequent_tokens(df.loc[df['id'].isin(set(comps[comp]))], stems_map)
            return {comp: list(sorted(most_frequent.items(), key=lambda kv: kv[1], reverse=True))[:n_words],
                    ignore_comp: []}

    # Pass paper indices (for corpus_terms and corpus_counts) instead of paper ids
    comps_ids = {comp: list(np.flatnonzero(df['id'].isin(comp_pids))) for comp, comp_pids in comps.items()}
    result = _get_topics_description_cosine(comps_ids, corpus_terms, corpus_counts, n_words, ignore_comp=ignore_comp)
    kwds = [(comp, ','.join([f'{t}:{v:.3f}' for t, v in vs])) for comp, vs in result.items()]
    logger.debug('Description\n' + '\n'.join(f'{comp}: {kwd}' for comp, kwd in kwds))

    return result


def _get_topics_description_cosine(comps, corpus_terms, corpus_counts, n_words, ignore_comp=None):
    """
    Select words with the frequency vector that is the closest to the 'ideal' frequency vector
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

        c = comp_idx[comp]  # Get the continuous index
        cluster_terms_idx = np.argsort(-adjusted_distance[c, :])[:n_words].tolist()
        result[comp] = [(corpus_terms[i], adjusted_distance[c, i]) for i in cluster_terms_idx]

    return result
