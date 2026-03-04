import logging
from functools import cache

import numpy as np
import pandas as pd

from pysrc.preprocess.embeddings.publications_db_connector import PublicationsDBConnector
from pysrc.config import PubKConfig
from pysrc.faiss.faiss_connector import FaissConnector
from pysrc.utils import l2norm
from pysrc.services.embeddings_service import fetch_texts_embedding

logger = logging.getLogger(__name__)

PUBK_CONFIG = PubKConfig(test=False)


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
    # Workaround for extra columns in the pidx index table
    return t[['pmid', 'chunk', 'similarity']]


class SemanticSearch:

    def __init__(self, source):
        self.faiss_connector = FaissConnector(
            source, PUBK_CONFIG.sentence_transformer_model, PUBK_CONFIG.embeddings_dimension
        )

    def search(self, source, text, noreviews, min_year, max_year, n):
        # Only Pubmed is supported for now
        assert source == 'Pubmed'
        lookup_n = n * 2  # For empty abstracts
        if SemanticSearch.is_true(noreviews):
            lookup_n *= 2
        if min_year is not None or max_year is not None:
            lookup_n *= 2
        result = self._search_raw(text, lookup_n)
        logger.info(f'After _search_raw: columns={result.columns.tolist()}, shape={result.shape}')

        result = self._process_results(result, noreviews, max_year, min_year)

        # Keep unique PMIDs, taking the first (highest similarity) occurrence
        result = result.drop_duplicates(subset=['pmid'], keep='first')
        result = result[:n]
        logger.info(f'Final result after limiting to {n}: {len(result)} papers')

        # Return list of [pmid, similarity] pairs
        return [[int(row['pmid']), float(row['similarity'])] for _, row in result.iterrows()]

    def search_embeddings(self, source, embeddings, noreviews, min_year, max_year, n):
        # Only Pubmed is supported for now
        assert source == 'Pubmed'
        lookup_n = n * 2  # For empty abstracts
        if SemanticSearch.is_true(noreviews):
            lookup_n *= 2
        if min_year is not None or max_year is not None:
            lookup_n *= 2
        result = self._search_embeddings_raw(embeddings, lookup_n)
        logger.info(f'After _search_embeddings_raw: columns={result.columns.tolist()}, shape={result.shape}')

        result = self._process_results(result, noreviews, max_year, min_year)

        # Keep unique PMIDs, taking the first (highest similarity) occurrence
        result = result.drop_duplicates(subset=['pmid'], keep='first')
        result = result[:n]
        logger.info(f'Final result after limiting to {n}: {len(result)} papers')

        # Return list of [pmid, similarity] pairs
        return [[int(row['pmid']), float(row['similarity'])] for _, row in result.iterrows()]

    @cache
    def _load_faiss_and_index(self):
        return self.faiss_connector.create_or_load_faiss()

    def _search_raw(self, text, n):
        embeddings = fetch_texts_embedding([text])
        if embeddings is None:
            raise RuntimeError("Failed to fetch text embeddings. Ensure the embeddings service is running.")
        query_embedding = embeddings[0].reshape(1, -1)
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
    def _process_results(result, noreviews, max_year, min_year):
        connector = PublicationsDBConnector()
        pmids = result['pmid'].tolist()
        df = connector.load_publications(pmids)
        df = df.merge(result, on='pmid', how='left')
        # Prioritize recent papers with the same relevance
        df = df.sort_values(by=['similarity', 'year'], ascending=False)

        # Make empty abstracts less similar
        empty_abstracts_mask = df['abstract'].isna() | (df['abstract'].str.strip() == '')
        df_non_empty_abstracts = df[~empty_abstracts_mask].copy().reset_index(drop=True)
        df_empty_abstracts = df[empty_abstracts_mask].copy().reset_index(drop=True)
        # Update empty abstracts similarity
        df_empty_abstracts['similarity'] -= (
                df_empty_abstracts['similarity'].max() - df_non_empty_abstracts['similarity'].min())
        df = pd.concat([df_non_empty_abstracts, df_empty_abstracts]).reset_index(drop=True)

        # Filter reviews
        if SemanticSearch.is_true(noreviews):
            df = df[df['type'] != 'Review']
            logger.info(f'After no_reviews: {len(df)} papers')

        # Filter years
        if min_year is not None or max_year is not None:
            df = df[df['year'].between(min_year if min_year else 0,
                                       max_year if max_year else 9999)]
            logger.info(f'After filtered years: {len(df)} papers')

        return df[['pmid', 'chunk', 'similarity']].copy().reset_index(drop=True)

    @staticmethod
    def is_true(noreviews):
        return noreviews == True or str(noreviews).lower() in ['on', 'true', 'yes', '1', 'y']


# TODO: support other sources
SEMANTIC_SEARCH = SemanticSearch("Pubmed")
