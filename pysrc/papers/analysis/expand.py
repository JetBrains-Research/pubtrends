import logging
from collections import Counter

import numpy as np

from pysrc.papers.analysis.text import tokens_stems

logger = logging.getLogger(__name__)


def expand_ids(
        ids, limit,
        loader, max_expand,
        citations_q_low, citations_q_high, citations_sigma,
        min_keywords_similarity,
        progress, current=1, task=None
):
    """
    Expands list of paper ids to the limit, filtering by citations counts and keywords
    :param ids: Initial ids to expand
    :param limit: Limit of expansion
    :param loader: DB loader
    :param max_expand: Number of papers expanded by references by loader before any filtration
    :param citations_q_low: Minimal percentile for groupwise citations count estimation, removes outliers
    :param citations_q_high: Max percentile for groupwise citations count estimation, removes outliers
    :param citations_sigma: Sigma for citations filtering range mean +- sigma * std
    :param min_keywords_similarity: Min keywords similarity
    :param progress:
    :param current:
    :param task:
    :return:
    """
    progress.info('Expanding related papers by references', current=current, task=task)

    if len(ids) > 1:
        cit_mean, cit_std = estimate_citations(ids, loader, citations_q_low, citations_q_high)
        mesh_stems, mesh_counter = estimate_mesh(ids, loader)
    else:
        # Cannot estimate these characteristics by a single paper
        cit_mean, cit_std = None, None
        mesh_stems, mesh_counter = None, None

    number_to_expand = limit - len(ids)
    expanded_df = loader.expand(ids, max_expand)
    logger.debug(f'Loaded {len(expanded_df)} papers by references')

    new_df = expanded_df.loc[np.logical_not(expanded_df['id'].isin(set(ids)))]
    logger.debug(f'New papers {len(new_df)}')
    if len(new_df) == 0:  # Nothing to add
        logger.debug('Nothing expanded')
        return ids

    if cit_mean is not None and cit_std is not None:
        logger.debug(f'New papers citations min={new_df["total"].min()}, max={new_df["total"].max()}')
        logger.debug(f'Filtering by citations mean({cit_mean}) +- {citations_sigma} * std({cit_std})')
        new_df = new_df.loc[[
            cit_mean - citations_sigma * cit_std <= t <= cit_mean + citations_sigma * cit_std
            for t in new_df['total']
        ]]
        logger.debug(f'Citations filtered: {len(new_df)}')

    logging.debug(f'Limiting new papers to {number_to_expand}')
    new_ids = list(new_df['id'])[:number_to_expand]
    if len(new_ids) == 0:  # Nothing to add
        logger.debug('Nothing expanded after citations filtration')
        return ids

    if mesh_stems is not None:
        new_publications = loader.load_publications(new_ids)
        fcs = []
        for _, row in new_publications.iterrows():
            pid = row['id']
            mesh = row['mesh']
            keywords = row['keywords']
            title = row['title']
            new_mesh_stems = [s for s, _ in tokens_stems((mesh + ' ' + keywords).replace(',', ' '))]
            if new_mesh_stems:
                # Estimate fold change of similarity vs random single paper
                similarity = sum([mesh_counter[s] / (len(mesh_stems) / len(ids)) for s in new_mesh_stems])
                fcs.append([pid, False, similarity, title, ','.join(new_mesh_stems)])
            else:
                fcs.append([pid, True, 0.0, title, ''])

        fcs.sort(key=lambda v: v[2], reverse=True)
        sim_threshold = fcs[0][2] * min_keywords_similarity  # Compute keywords similarity threshold as a fraction of top
        for v in fcs:
            v[1] = v[1] or v[2] > sim_threshold

        logger.debug('Pid\tOk\tSimilarity\tTitle\tMesh\n' +
                     '\n'.join(f'{p}\t{"+" if a else "-"}\t{int(s)}\t{t}\t{m}' for
                               p, a, s, t, m in fcs))
        new_mesh_ids = [v[0] for v in fcs if v[1]][:limit]
        logger.debug(f'Similar by mesh papers: {len(new_mesh_ids)}')
        if len(new_mesh_ids) == 0:  # Nothing to add
            logger.debug('Nothing expanded after mesh filtration')
            return ids
        else:
            logger.debug(f'Expanded to {len(ids) + len(new_ids)} papers')
            return ids + new_mesh_ids
    else:
        logger.debug(f'Expanded to {len(ids) + len(new_ids)} papers')
        return ids + new_ids


def estimate_mesh(ids, loader):
    logger.debug(f'Estimating mesh and keywords terms to keep the theme')
    publications = loader.load_publications(ids)
    mesh_stems = [s for s, _ in tokens_stems(
        ' '.join(publications['mesh'] + ' ' + publications['keywords']).replace(',', ' ')
    )]
    mesh_counter = Counter(mesh_stems)
    logger.debug(f'Mesh most common:\n' + ','.join(f'{k}:{"{0:.3f}".format(v / len(mesh_stems))}'
                                                   for k, v in mesh_counter.most_common(20)))
    return (mesh_stems, mesh_counter) if len(mesh_stems) > 0 else (None, None)


def estimate_citations(ids, loader, q_low, q_high):
    total = loader.estimate_citations(ids)
    logger.debug(f'Citations min={total.min()}, max={total.max()}, '
                 f'mean={total.mean()}, std={total.std()}')
    citations_q_low = np.percentile(total, q_low)
    citations_q_high = np.percentile(total, q_high)
    logger.debug(
        f'Filtering < Q{q_low}={citations_q_low} or > Q{q_high}={citations_q_high}')
    filtered = total[np.logical_and(total >= citations_q_low, total <= citations_q_high)]
    mean = filtered.mean()
    std = filtered.std()
    logger.debug(f'Filtered citations min={filtered.min()}, max={filtered.max()}, mean={mean}, std={std}')
    return mean, std
