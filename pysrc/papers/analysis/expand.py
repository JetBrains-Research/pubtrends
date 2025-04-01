import logging
from collections import Counter

import numpy as np

from pysrc.papers.analysis.text import stemmed_tokens

logger = logging.getLogger(__name__)


def expand_ids(
        loader,
        search_ids,
        limit,
        expand_steps,
        max_expand,
        citations_q_low,
        citations_q_high,
        citations_sigma,
        mesh_similarity_threshold,
        single_paper_impact
):
    """
    Expands list of paper ids to the limit, filtering by citations counts and keywords
    :param search_ids: Initial ids to expand
    :param limit: Limit of expansion
    :param expand_steps: Number of expand steps
    :param loader: DB loader
    :param max_expand: Number of papers expanded by references by loader before any filtration
    :param citations_q_low: Minimal percentile for groupwise citations count estimation, removes outliers
    :param citations_q_high: Max percentile for groupwise citations count estimation, removes outliers
    :param citations_sigma: Sigma for citations filtering range mean +- sigma * std
    :param mesh_similarity_threshold: Similarity fraction of top similar, value 0 - 1
    :param single_paper_impact: Impact of single paper when analyzing citations and mesh terms for single paper
    :return:
    """
    logger.debug(f'Expanding papers {", ".join(search_ids)} with limit={limit} max_expand={max_expand}')
    # Fetch references at first, but in some cases paper may have empty references
    logger.debug('Loading direct references for paper analysis')
    ids = search_ids.copy()
    for pid in search_ids:
        ids.extend(loader.load_references(pid, limit))
    logger.debug(f'Loaded {len(ids) - 1} references')

    if len(ids) > 1:
        cit_mean, cit_std = estimate_citations(ids, True, loader,
                                               citations_q_low, citations_q_high, single_paper_impact)
        if cit_mean is not None and cit_std is not None:
            mesh_stems, mesh_counter = estimate_mesh(ids, True, loader, single_paper_impact)
        else:
            mesh_stems, mesh_counter = None, None
    else:
        # Cannot estimate these characteristics by a single paper
        cit_mean, cit_std = None, None
        mesh_stems, mesh_counter = None, None

    for i in range(expand_steps):
        logger.debug(f'Expanding step {i + 1}/{expand_steps}')
        ids = _expand_ids_step(
            loader, ids, limit, max_expand, cit_mean, cit_std, citations_sigma, mesh_similarity_threshold,
            mesh_stems, mesh_counter)
    # Keep original start ids first
    return search_ids + [x for x in ids if x not in search_ids][:limit - len(search_ids)]


def _expand_ids_step(
        loader,
        ids,
        limit,
        max_expand,
        cit_mean,
        cit_std,
        citations_sigma,
        mesh_similarity_threshold,
        mesh_stems,
        mesh_counter
):
    new_df = loader.expand(ids, max_expand - len(ids))
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
        new_ids = list(new_df['id'])
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
                new_publications_mesh_infos.append([pid, True, similarity, title, ','.join(new_mesh_stems)])
            else:
                new_publications_mesh_infos.append([pid, False, 0.0, title, ''])

        new_publications_mesh_infos.sort(key=lambda i: i[2], reverse=True)
        # Compute keywords similarity threshold as a fraction of top
        sim_threshold = new_publications_mesh_infos[0][2] * mesh_similarity_threshold
        logger.debug(f'Similarity threshold {sim_threshold}')
        # Show top 50 similar papers
        filtered_publications_mesh_infos = [i for i in new_publications_mesh_infos if not i[1] or i[2] >= sim_threshold]
        logger.debug('Pid\tMesh\tSimilarity\tTitle\tMesh\n' +
                     '\n'.join(f'{p}\t{"+" if a else "-"}\t{int(s)}\t{t[:80]}\t{m[:50]}' for
                               p, a, s, t, m in filtered_publications_mesh_infos[:50]))
        new_ids = [i[0] for i in filtered_publications_mesh_infos]
        logger.debug(f'Similar by mesh papers: {len(new_ids)}')

    new_ids = new_ids[:limit - len(ids)]
    logger.debug(f'Expanded to {len(ids) + len(new_ids)} papers')
    return ids + new_ids


def estimate_mesh(ids, single_paper, loader, single_paper_impact):
    logger.debug(f'Estimating mesh and keywords terms to keep the theme')
    publications = loader.load_publications(ids)
    if single_paper:
        # Artificially inflate paper of interest influence
        mesh_stems = [s for s, _ in stemmed_tokens(
            ' '.join(publications['mesh'].values[0] + ' ' + publications['keywords'].values[0]).replace(',', ' ')
        )] * single_paper_impact + \
                     [s for s, _ in stemmed_tokens(
                         ' '.join(publications['mesh'][1:] + ' ' + publications['keywords'][1:]).replace(',', ' ')
                     )]
    else:
        mesh_stems = [s for s, _ in stemmed_tokens(
            ' '.join(publications['mesh'] + ' ' + publications['keywords']).replace(',', ' ')
        )]
    mesh_counter = Counter(mesh_stems)
    logger.debug(f'Mesh most common:\n' + ','.join(f'{k}:{"{0:.3f}".format(v / len(mesh_stems))}'
                                                   for k, v in mesh_counter.most_common(20)))
    return (mesh_stems, mesh_counter) if len(mesh_stems) > 0 else (None, None)


def estimate_citations(ids, single_paper, loader, q_low, q_high, single_paper_impact):
    if single_paper:
        # Artificially inflate paper of interest influence
        total = loader.load_citations_counts(ids[:1]) * single_paper_impact + loader.load_citations_counts(ids[1:])
    else:
        total = loader.load_citations_counts(ids)
    if len(total) == 0:
        return None, None
    logger.debug(f'Citations min={np.min(total)}, max={np.max(total)}, '
                 f'mean={np.mean(total)}, std={np.std(total)}')
    citations_q_low = np.percentile(total, q_low)
    citations_q_high = np.percentile(total, q_high)
    logger.debug(
        f'Filtering < Q{q_low}={citations_q_low} or > Q{q_high}={citations_q_high}'
    )
    filtered = [t for t in total if citations_q_low <= t <= citations_q_high]
    if len(filtered) == 0:
        return None, None
    mean, std = np.mean(filtered), np.std(filtered)
    logger.debug(f'Filtered citations min={np.min(filtered)}, max={np.max(filtered)}, mean={mean}, std={std}')
    return mean, std
