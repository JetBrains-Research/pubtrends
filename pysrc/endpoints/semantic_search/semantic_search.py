import logging
from functools import cache

import numpy as np
import pandas as pd

from pysrc.config import PubtrendsConfig
from pysrc.faiss.faiss_connector import FaissConnector
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
        lookup_n = n * 2  # For empty abstracts
        if noreviews:
            lookup_n *= 2
        if min_year is not None or max_year is not None:
            lookup_n *= 2
        result = self._search_raw(text, lookup_n)
        logger.info(f'After _search_raw: columns={result.columns.tolist()}, shape={result.shape}')

        result = self._filter_results(result, noreviews, max_year, min_year)

        result = result['pmid'].unique().tolist()[:n]
        logger.info(f'Final result after limiting to {n}: {len(result)} papers')
        # Return list of [pmid]
        return result

    def search_embeddings(self, source, embeddings, noreviews, min_year, max_year, n):
        # Only Pubmed is supported for now
        assert source == 'Pubmed'
        lookup_n = n * 2  # For empty abstracts
        if noreviews:
            lookup_n *= 2
        if min_year is not None or max_year is not None:
            lookup_n *= 2
        result = self._search_embeddings_raw(embeddings, lookup_n)
        logger.info(f'After _search_embeddings_raw: columns={result.columns.tolist()}, shape={result.shape}')

        result = self._filter_results(result, noreviews, max_year, min_year)

        result = result['pmid'].unique().tolist()[:n]
        logger.info(f'Final result after limiting to {n}: {len(result)} papers')
        # Return list of [pmid]
        return result

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
        connector = PublicationsDBConnector()
        pmids = result['pmid'].tolist()
        df = connector.load_publications(pmids)
        df = df.merge(result, on='pmid', how='left')

        # Make empty abstracts less similar
        empty_abstracts_mask = df['abstract'].isna() | (df['abstract'].str.strip() == '')
        df_non_empty_abstracts = df[~empty_abstracts_mask].copy().reset_index(drop=True)
        df_empty_abstracts = df[empty_abstracts_mask].copy().reset_index(drop=True)
        # Update empty abstracts similarity
        df_empty_abstracts['similarity'] -= (
                df_empty_abstracts['similarity'].max() - df_non_empty_abstracts['similarity'].min())
        df = pd.concat([df_non_empty_abstracts, df_empty_abstracts]).reset_index(drop=True)

        # Filter reviews
        if noreviews:
            df = df[df['type'] != 'Review']
            logger.info(f'After no_reviews: {len(df)} papers')

        # Filter years
        if min_year is not None or max_year is not None:
            df = df[df['year'].between(int(min_year) if min_year else 0,
                                       int(max_year) if max_year else 9999)]
            logger.info(f'After filtered years: {len(df)} papers')

        return df[['pmid', 'similarity']].copy().reset_index(drop=True)


# TODO: support other sources
SEMANTIC_SEARCH = SemanticSearch("Pubmed")
