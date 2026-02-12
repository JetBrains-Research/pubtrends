import logging

import numpy as np

from pysrc.config import *
from pysrc.papers.analysis.text import fetch_embeddings_from_db, embeddings_from_service, chunks_to_text_embeddings
from pysrc.services.embeddings_service import is_texts_embeddings_available, is_embeddings_db_available
from pysrc.services.semantic_search_service import is_semantic_search_service_available, \
    fetch_semantic_search_embeddings

logger = logging.getLogger(__name__)


def expand_ids(
        loader,
        search_ids,
        expand_steps,
        limit,
        noreviews,
        max_expand,
        semantic_expand,
        citations_q_low=EXPAND_CITATIONS_Q_LOW,
        citations_q_high=EXPAND_CITATIONS_Q_HIGH,
        citations_sigma=EXPAND_CITATIONS_SIGMA,
        single_paper_impact=EXPAND_SINGLE_PAPER_IMPACT
):
    """
    Expands list of paper ids to the limit, filtering by citations counts and keywords
    :param search_ids: Initial ids to expand
    :param limit: Limit of expansion
    :param expand_steps: Number of expand steps
    :param loader: DB loader
    :param max_expand: Number of papers expanded by references by loader before any filtration
    :type semantic_expand: Fraction of papers expanded by semantic search
    :param citations_q_low: Minimal percentile for groupwise citations count estimation, removes outliers
    :param citations_q_high: Max percentile for groupwise citations count estimation, removes outliers
    :param citations_sigma: Sigma for citations filtering range mean +- sigma * std
    :param single_paper_impact: Impact of single paper when analyzing citations and mesh terms for single paper
    :return:
    """
    if expand_steps == 0:
        return search_ids

    logger.debug(f'Expanding papers {", ".join(search_ids)} with limit={limit} max_expand={max_expand}')

    # Try to estimate citations ranges by references only
    cit_mean, cit_std = None, None
    references = []
    if expand_steps >= 1:
        logger.debug(f'Estimating citations counts by direct references')
        for pid in search_ids:
            references.extend(loader.load_references(pid, limit))
        logger.debug(f'Loaded total {len(references)} references')
        if len(search_ids) + len(references) > 1:
            cit_mean, cit_std = estimate_citations(
                search_ids + references, True, loader,
                citations_q_low, citations_q_high, single_paper_impact
            )

    ids = search_ids.copy()
    if is_semantic_search_service_available() and is_texts_embeddings_available():
        logger.debug(f'Step 0/{expand_steps} - expand by similar papers embeddings')
        df = loader.load_publications(search_ids)
        chunks_embeddings, chunks_idx =\
            fetch_embeddings_from_db(df) if is_embeddings_db_available() else embeddings_from_service(df)
        papers_text_embeddings = chunks_to_text_embeddings(df, chunks_embeddings, chunks_idx)
        logger.debug(f'Loaded {len(papers_text_embeddings)} papers embeddings {papers_text_embeddings.shape}')
        logger.debug('Adding similar papers by embeddings')
        similar_by_embds = fetch_semantic_search_embeddings(
            'Pubmed', papers_text_embeddings, noreviews, int((limit - len(ids)) * semantic_expand)
        )
        logger.debug(f'Loaded {len(similar_by_embds)} similar papers by embeddings')
        for pid in similar_by_embds:
            if pid not in ids:
                ids.append(pid)

    start_step = 1
    if expand_steps >= 1 and len(references) > 0:
        logger.debug(f'Step {start_step}/{expand_steps} - expand by references only')
        ids += references[:limit - len(ids)]
        logger.debug(f'Step {start_step}/{expand_steps} - {len(ids)}')
        start_step +=1

    for step in range(start_step, expand_steps + 1):
        if len(ids) >= limit:
            break
        logger.debug(f'Expanding step {step}/{expand_steps} - expand by both references and citations')
        ids = _expand_ids_step(
            loader, ids, limit, noreviews, max_expand,
            cit_mean, cit_std, citations_sigma
        )
        logger.debug(f'Step {start_step}/{expand_steps} - {len(ids)}')

    # Keep original start ids first
    return search_ids + [x for x in ids if x not in search_ids][:limit - len(search_ids)]


def _expand_ids_step(
        loader,
        ids,
        limit,
        noreviews,
        max_expand,
        cit_mean,
        cit_std,
        citations_sigma
):
    new_df = loader.expand(ids, max_expand - len(ids), noreviews)
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


    new_ids = new_ids[:limit - len(ids)]
    logger.debug(f'Expanded to {len(ids) + len(new_ids)} papers')
    return ids + new_ids


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
