import logging
from functools import cache

import numpy as np
import pandas as pd

from pysrc.config import PubtrendsConfig
from pysrc.faiss.faiss_connector import FaissConnector
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.utils import l2norm
from pysrc.services.embeddings_service import fetch_texts_embedding

logger = logging.getLogger(__name__)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)


def semantic_search_faiss_embedding(faiss_index, pids_idx, query_embedding, k):
    # Normalize embeddings if using cosine similarity
    query_embedding = l2norm(query_embedding)
    similarities, indices = faiss_index.search(query_embedding.astype('float32'), k)
    t = pids_idx.iloc[indices[0]].copy().reset_index(drop=True)
    t['similarity'] = similarities[0]
    return t


class SemanticSearch:

    def __init__(self, source):
        self.faiss_connector = FaissConnector(
            source, PUBTRENDS_CONFIG.sentence_transformer_model, PUBTRENDS_CONFIG.embeddings_dimension
        )

    def search(self, source, text, noreviews, n):
        result = self._search_raw(text, (n * 2) if noreviews else n)
        if noreviews:
            result = self._filter_no_reviews(source, result)
        return list(result['pmid'].unique())[:n]

    def search_embeddings(self, source, embeddings, noreviews, n):
        result = self._search_embeddings_raw(embeddings, (n * 2) if noreviews else n)
        if noreviews:
            result = self._filter_no_reviews(source, result)
        return list(result['pmid'].unique())[:n]

    @cache
    def _load_faiss_and_index(self):
        return self.faiss_connector.create_or_load_faiss()

    def _search_raw(self, text, n):
        embeddings_func = lambda t: fetch_texts_embedding([t])[0]
        query_embedding = embeddings_func(text).reshape(1, -1)
        faiss_index, pids_idx = self._load_faiss_and_index()
        result = semantic_search_faiss_embedding(faiss_index, pids_idx, query_embedding, n)
        result['pmid'] = result['pmid'].astype(str)
        return result

    def _search_embeddings_raw(self, embeddings, n):
        faiss_index, pids_idx = self._load_faiss_and_index()
        ts = []
        for e in embeddings:
            npe = np.array(e).astype('float32').reshape(1, -1)
            ts.append(semantic_search_faiss_embedding(faiss_index, pids_idx, npe, n))
        result = pd.concat(ts).reset_index(drop=True).sort_values(by='similarity', ascending=False)[:n]
        result['pmid'] = result['pmid'].astype(str)
        return result

    @staticmethod
    def _filter_no_reviews(source, result):
        loader = Loaders.get_loader(source, PUBTRENDS_CONFIG)
        df = loader.load_publications(result['pmid'].tolist())
        noreview_ids = set(df[df['type'] != 'Review']['id'])
        result = result[result['pmid'].isin(noreview_ids)]
        return result


# TODO: support other sources
SEMANTIC_SEARCH = SemanticSearch("Pubmed")
