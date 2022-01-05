import logging
from collections import Counter

import numpy as np

from pysrc.papers.analysis.text import stemmed_tokens

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

    new_df = loader.expand(ids, max_expand)
    logger.debug(f'Expanded by references: {len(new_df)}')

    if len(new_df) == 0:
        return ids

    if cit_mean is not None and cit_std is not None:
        logger.debug(f'New papers citations min={new_df["total"].min()}, max={new_df["total"].max()}')
        logger.debug(f'Filtering by citations mean({cit_mean}) +- {citations_sigma} * std({cit_std})')
        new_publications_citations_infos = []
        for _, row in new_df.iterrows():
            pid, total = row['id'], row['total']
            new_publications_citations_infos.append([pid, total])
        new_publications_citations_infos.sort(key=lambda i: i[1], reverse=True)
        new_ids = [i[0] for i in new_publications_citations_infos
                   if cit_mean - citations_sigma * cit_std <= i[1] <= cit_mean + citations_sigma * cit_std]
    else:
        new_ids = new_df['id']
    logger.debug(f'Citations filter: {len(new_ids)}')

    if len(new_ids) == 0:
        return ids

    if mesh_stems is not None:
        new_publications = loader.load_publications(new_ids)
        new_publications_mesh_infos = []
        for _, row in new_publications.iterrows():
            pid, title, mesh, keywords = row['id'], row['title'], row['mesh'], row['keywords']
            new_mesh_stems = [s for s, _ in stemmed_tokens((mesh + ' ' + keywords).replace(',', ' '))]
            if new_mesh_stems:
                # Estimate fold change of similarity vs random single paper
                similarity = sum([mesh_counter[s] / (len(mesh_stems) / len(ids)) for s in new_mesh_stems])
                new_publications_mesh_infos.append([pid, False, similarity, title, ','.join(new_mesh_stems)])
            else:
                new_publications_mesh_infos.append([pid, True, 0.0, title, ''])

        new_publications_mesh_infos.sort(key=lambda i: i[2], reverse=True)
        # Compute keywords similarity threshold as a fraction of top
        sim_threshold = new_publications_mesh_infos[0][2] * min_keywords_similarity
        logger.debug(f'Similarity threshold {sim_threshold}')
        logger.debug('Pid\tOk\tSimilarity\tTitle\tMesh\n' +
                     '\n'.join(f'{p}\t{"+" if a else "-"}\t{int(s)}\t{t}\t{m}' for
                               p, a, s, t, m in new_publications_mesh_infos))
        new_ids = [i[0] for i in new_publications_mesh_infos if i[1] or i[2] >= sim_threshold]
        logger.debug(f'Similar by mesh papers: {len(new_ids)}')

    new_ids = new_ids[:limit - len(ids)]
    logger.debug(f'Expanded to {len(ids) + len(new_ids)} papers')
    return ids + new_ids


def estimate_mesh(ids, loader):
    logger.debug(f'Estimating mesh and keywords terms to keep the theme')
    publications = loader.load_publications(ids)
    mesh_stems = [s for s, _ in stemmed_tokens(
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
