import logging
from functools import cache

import numpy as np

from pysrc.papers.utils import l2norm
from pysrc.config import PubtrendsConfig
from pysrc.faiss.faiss_connector import FaissConnector
from pysrc.papers.db.loaders import Loaders
from pysrc.services.embeddings_service import fetch_texts_embedding

logger = logging.getLogger(__name__)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)


def semantic_search_faiss(query_text, faiss_index, pids_idx, embeddings_func, k):
    query_vector = embeddings_func(query_text).reshape(1, -1)
    # Normalize embeddings if using cosine similarity
    query_vector = l2norm(query_vector)
    similarities, indices = faiss_index.search(query_vector.astype('float32'), k)
    t = pids_idx.iloc[indices[0]].copy().reset_index(drop=True)
    t['similarity'] = similarities[0]
    return t


class SemanticSearch:

    def __init__(self, source):
        self.faiss_connector = FaissConnector(
            source, PUBTRENDS_CONFIG.sentence_transformer_model, PUBTRENDS_CONFIG.embeddings_dimension
        )

    @cache
    def load_faiss_and_index(self):
        return self.faiss_connector.create_or_load_faiss()

    def search_raw(self, text, n):
        embeddings_func = lambda t: fetch_texts_embedding([t])[0]
        faiss_index, pids_idx = self.load_faiss_and_index()
        result = semantic_search_faiss(text, faiss_index, pids_idx, embeddings_func, n)
        result['pmid'] = result['pmid'].astype(str)
        return result

    def search(self, source, text, noreviews, n):
        result = self.search_raw(text, n * 2)
        if noreviews:
            loader = Loaders.get_loader(source, PUBTRENDS_CONFIG)
            df = loader.load_publications(result['pmid'].tolist())
            noreview_ids = set(df[df['type'] != 'Review']['id'])
            result = result[result['pmid'].isin(noreview_ids)].head(n)
        return list(result['pmid'].unique())


# TODO: support other sources
SEMANTIC_SEARCH = SemanticSearch("Pubmed")
