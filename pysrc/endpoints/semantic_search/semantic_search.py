import logging
import os
import re
from functools import cache

import faiss
import numpy as np
import pandas as pd

from pysrc.config import PubtrendsConfig
from pysrc.services.embeddings_service import fetch_texts_embedding
from pysrc.papers.db.loaders import Loaders

logger = logging.getLogger(__name__)

FAISS_PATHS = ['/faiss', os.path.expanduser('~/.pubtrends/faiss')]
for p in FAISS_PATHS:
    if os.path.isdir(p):
        faiss_path = p
        break
else:
    raise RuntimeError('Failed to configure faiss directory')

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

def l2norm(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    v /= norm
    return v


def create_faiss(embeddings_dimension, exact=False):
    if exact:
        print('Exact search index')
        index = faiss.IndexFlatIP(embeddings_dimension)
    else:
        print('Approximate search index')
        quantizer = faiss.IndexFlatL2(embeddings_dimension)
        index = faiss.IndexIVFPQ(quantizer, embeddings_dimension, 200, 16, 8)
    return index

def create_or_load_faiss(faiss_index_file, pids_index_file, embeddings_dimension, exact=False):
    if os.path.exists(faiss_index_file):
        print('Loading Faiss index from existing file')
        index = faiss.read_index(faiss_index_file)
        # For accurate search
        index.nprobe = 200
    else:
        print('Creating empty Faiss index')
        index = create_faiss(embeddings_dimension, exact)
    if os.path.exists(pids_index_file):
        pids_idx = pd.read_csv(pids_index_file, compression='gzip')
    else:
        pids_idx = pd.DataFrame(data=[], columns=['pmid', 'chunk', 'year', 'noreview'], dtype=int)
    return index, pids_idx

def semantic_search_faiss(query_text, faiss_index, pids_idx, embeddings_func, k):
    query_vector = embeddings_func(query_text).reshape(1, -1)
    # Normalize embeddings if using cosine similarity
    query_vector = l2norm(query_vector)
    similarities, indices = faiss_index.search(query_vector.astype('float32'), k)
    t = pids_idx.iloc[indices[0]].copy().reset_index(drop=True)
    t['similarity'] = similarities[0]
    return t

class SemanticSearch:

    def __init__(self):
        self.model_name = PUBTRENDS_CONFIG.sentence_transformer_model
        self.folder_name = re.sub('[^a-zA-Z0-9]', '_', self.model_name)



    @cache
    def load_faiss_and_index(self, source, folder_name):
        folder = f'{faiss_path}/faiss_{source}_{folder_name}'
        assert os.path.exists(folder), f"FAISS index folder doesn't exist {folder}"
        faiss_index_file = os.path.expanduser(f'{folder}/embeddings.index')
        assert os.path.exists(faiss_index_file), f"FAISS index file doesn't exist {faiss_index_file}"
        pids_index_file = os.path.expanduser(f'{folder}/pids.csv.gz')
        assert os.path.exists(pids_index_file), f"PIDs index file doesn't exist {pids_index_file}"
        logger.info(f'Loading faiss and index from {faiss_index_file}, {pids_index_file}')

        faiss_index, pids_idx = create_or_load_faiss(faiss_index_file, pids_index_file, None, False)
        return faiss_index, pids_idx

    def search_raw(self, source, text, n):
        embeddings_func = lambda t: fetch_texts_embedding([t])[0]
        faiss_index, pids_idx = self.load_faiss_and_index(source, self.folder_name)
        result = semantic_search_faiss(text, faiss_index, pids_idx, embeddings_func, n)
        result['pmid'] = result['pmid'].astype(str)
        return result

    def search(self, source, text, noreviews, n):
        result = self.search_raw(source, text, n * 2)
        if noreviews:
            loader = Loaders.get_loader(source, PUBTRENDS_CONFIG)
            df = loader.load_publications(result['pmid'].tolist())
            noreview_ids = set(df[df['type'] != 'Review']['id'])
            result = result[result['pmid'].isin(noreview_ids)].head(n)
        return list(result['pmid'].unique())


SEMANTIC_SEARCH = SemanticSearch()