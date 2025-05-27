import logging
from itertools import chain

import numpy as np

from pysrc.papers.analysis.text import get_frequent_tokens

logger = logging.getLogger(__name__)

def get_topics_description(df, corpus, corpus_tokens, corpus_counts, n_words, ignore_comp=None):
    """
    Get words from abstracts that describe the components the best way
    using closest to the 'ideal' frequency vector - [0, ..., 0, 1, 0, ..., 0] in tokens of cosine distance
    """
    comps = df[['id', 'comp']].groupby('comp')['id'].apply(list).to_dict()
    logger.debug(f'Generating topics description, ignore_comp={ignore_comp}')
    # Since some of the components may be skipped, use this dict for continuous indexes'
    comp_idx = {c: i for i, c in enumerate(c for c in comps if c != ignore_comp)}
    # In cases with less than 2 components, return frequencies
    if len(comp_idx) < 2:
        comp = list(comp_idx.keys())[0]
        if ignore_comp is None:
            most_frequent = get_frequent_tokens(chain(*chain(*corpus)))
            return {comp: list(sorted(most_frequent.items(), key=lambda kv: kv[1], reverse=True))[:n_words]}
        else:
            most_frequent = get_frequent_tokens(
                chain(*chain(*[corpus[i] for i in np.flatnonzero(df['id'].isin(set(comps[comp])))]))
            )
            return {comp: list(sorted(most_frequent.items(), key=lambda kv: kv[1], reverse=True))[:n_words],
                    ignore_comp: []}

    # Pass paper indices (for corpus_tokens and corpus_counts) instead of paper ids
    comps_ids = {comp: list(np.flatnonzero(df['id'].isin(comp_pids))) for comp, comp_pids in comps.items()}
    result = _get_topics_description_cosine(comps_ids, corpus_tokens, corpus_counts, n_words, ignore_comp=ignore_comp)
    kwds = [(comp, ','.join([f'{t}:{v:.3f}' for t, v in vs])) for comp, vs in result.items()]
    logger.debug('Description\n' + '\n'.join(f'{comp}: {kwd}' for comp, kwd in kwds))

    return result


def _get_topics_description_cosine(comps, corpus_tokens, corpus_counts, n_words, ignore_comp=None):
    """
    Select words with the frequency vector that is the closest to the 'ideal' frequency vector
    ([0, ..., 0, 1, 0, ..., 0]) in tokens of cosine distance
    """
    logger.debug('Compute average tokens counts per components')
    # Since some of the components may be skipped, use this dict for continuous indexes
    comp_idx = {c: i for i, c in enumerate(c for c in comps if c != ignore_comp)}
    tokens_freqs_per_comp = np.zeros(shape=(len(comp_idx), corpus_counts.shape[1]), dtype=np.float32)
    for comp, comp_ids in comps.items():
        if comp != ignore_comp:  # Not ignored
            tokens_freqs_per_comp[comp_idx[comp], :] = \
                np.sum(corpus_counts[comp_ids, :], axis=0)

    # Calculate total number of occurrences for each word
    tokens_freqs_total = np.sum(tokens_freqs_per_comp, axis=0)

    # Normalize frequency vector for each word to have length of 1
    tokens_freqs_norm = np.sqrt(np.diag(tokens_freqs_per_comp.T @ tokens_freqs_per_comp))
    tokens_freqs_per_comp = tokens_freqs_per_comp / tokens_freqs_norm

    logger.debug('Take frequent tokens that have the most descriptive frequency vector for topics')
    # Calculate cosine distance between the frequency vector and [0, ..., 0, 1, 0, ..., 0] for each cluster
    cluster_mask = np.eye(len(comp_idx))
    distance = tokens_freqs_per_comp.T @ cluster_mask
    # Add some weight for more frequent tokens to get rid of extremely rare ones in the top
    adjusted_distance = distance.T * np.log1p(tokens_freqs_total)

    result = {}
    for comp in comps.keys():
        if comp == ignore_comp:
            result[comp] = []  # Ignored component
            continue

        c = comp_idx[comp]  # Get the continuous index
        cluster_tokens_idx = np.argsort(-adjusted_distance[c, :])[:n_words].tolist()
        result[comp] = [(corpus_tokens[i], adjusted_distance[c, i]) for i in cluster_tokens_idx]

    return result
