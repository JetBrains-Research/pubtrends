import logging
from functools import cache

import numpy as np
import pandas as pd

from pysrc.config import PubtrendsConfig
from pysrc.faiss.faiss_connector import FaissConnector
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.utils import l2norm
from pysrc.preprocess.embeddings.publications_db_connector import PublicationsDBConnector
from pysrc.services.embeddings_service import fetch_texts_embedding

logger = logging.getLogger(__name__)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)


def semantic_search_faiss_embedding(faiss_index, pids_idx, query_embedding, k):
    # Normalize embeddings if using cosine similarity
    query_embedding = l2norm(query_embedding)

    # Validate embedding dimension matches FAISS index
    expected_dim = faiss_index.d
    actual_dim = query_embedding.shape[1]
    if actual_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: query embedding has dimension {actual_dim}, "
            f"but FAISS index expects dimension {expected_dim}. "
            f"Query embedding shape: {query_embedding.shape}"
        )

    similarities, indices = faiss_index.search(query_embedding.astype('float32'), k)
    t = pids_idx.iloc[indices[0]].copy().reset_index(drop=True)
    t['similarity'] = similarities[0]
    return t


class SemanticSearch:

    def __init__(self, source):
        self.faiss_connector = FaissConnector(
            source, PUBTRENDS_CONFIG.sentence_transformer_model, PUBTRENDS_CONFIG.embeddings_dimension
        )

    def search(self, source, text, noreviews, min_year, max_year, n):
        # Only Pubmed is supported for now
        assert source == 'Pubmed'
        if noreviews:
            n *= 2
        if min_year is not None or max_year is not None:
            n *= 2
        result = self._search_raw(text, n)
        logger.info(f'After _search_raw: columns={result.columns.tolist()}, shape={result.shape}')

        result = self._filter_results(result, noreviews, max_year, min_year)

        result = result[:n]
        logger.info(f'Final result after limiting to {n}: {len(result)} papers')

        # Return list of [pmid, similarity] pairs
        return [[int(row['pmid']), float(row['similarity'])] for _, row in result.iterrows()]

    def search_embeddings(self, source, embeddings, noreviews, min_year, max_year, n):
        # Only Pubmed is supported for now
        assert source == 'Pubmed'
        if noreviews:
            n *= 2
        if min_year is not None or max_year is not None:
            n *= 2
        result = self._search_embeddings_raw(embeddings, n)
        logger.info(f'After _search_embeddings_raw: columns={result.columns.tolist()}, shape={result.shape}')

        result = self._filter_results(result, noreviews, max_year, min_year)

        result = result[:n]
        logger.info(f'Final result after limiting to {n}: {len(result)} papers')

        # Return list of [pmid, similarity] pairs
        return [[int(row['pmid']), float(row['similarity'])] for _, row in result.iterrows()]

    @cache
    def _load_faiss_and_index(self):
        return self.faiss_connector.create_or_load_faiss()

    def _search_raw(self, text, n):
        embeddings_func = lambda t: fetch_texts_embedding([t])[0]
        query_embedding = embeddings_func(text).reshape(1, -1)
        faiss_index, pids_idx = self._load_faiss_and_index()
        result = semantic_search_faiss_embedding(faiss_index, pids_idx, query_embedding, n)
        return result

    def _search_embeddings_raw(self, embeddings, n):
        faiss_index, pids_idx = self._load_faiss_and_index()
        ts = []
        for e in embeddings:
            npe = np.array(e).astype('float32').reshape(1, -1)
            ts.append(semantic_search_faiss_embedding(faiss_index, pids_idx, npe, n))
        result = pd.concat(ts).reset_index(drop=True).sort_values(by='similarity', ascending=False)[:n]
        return result

    @staticmethod
    def _filter_results(result, noreviews, max_year, min_year):
        if not(noreviews or min_year is not None or max_year is not None):
            return result
        connector = PublicationsDBConnector()
        pmids = result['pmid'].tolist()
        df = connector.load_publications(pmids)

        if noreviews:
            noreivew_ids = set(df[df['type'] != 'Review']['pmid'])
            result = result[~result['pmid'].isin(noreivew_ids)]
            logger.info(f'After no_reviews: {len(result)} papers')

        if min_year is not None or max_year is not None:
            filtered_years = set(
                df[df['year'].between(min_year if min_year else 0, max_year if max_year else 9999)]['pmid'])
            result = result[result['pmid'].isin(filtered_years)]
            logger.info(f'After filtered years: {len(result)} papers')

        return result


# TODO: support other sources
SEMANTIC_SEARCH = SemanticSearch("Pubmed")
