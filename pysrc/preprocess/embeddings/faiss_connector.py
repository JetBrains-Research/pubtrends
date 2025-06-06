import logging
import os

import faiss
import numpy as np
import pandas as pd

from pysrc.config import PubtrendsConfig

config = PubtrendsConfig(test=False)

logger = logging.getLogger(__name__)


class FaissConnector:
    def __init__(self, embeddings_model_name, embeddings_dimension, exact=False):
        self.embeddings_model_name = embeddings_model_name
        self.embeddings_dimension = embeddings_dimension
        self.exact = exact
        self.faiss_dir = os.path.expanduser(f'~/faiss_{embeddings_model_name}')
        if not os.path.exists(self.faiss_dir):
            os.mkdir(self.faiss_dir)
        self.faiss_index_file = os.path.expanduser(f'{self.faiss_dir}/embeddings.index')
        self.pids_index_file = os.path.expanduser(f'{self.faiss_dir}/pids.csv.gz')

    def create_faiss(self):
        if self.exact:
            print('Exact search index')
            index = faiss.IndexFlatIP(self.embeddings_dimension)
        else:
            print('Approximate search index')
            quantizer = faiss.IndexFlatL2(self.embeddings_dimension)
            index = faiss.IndexIVFPQ(quantizer, self.embeddings_dimension, 200, 16, 8)
        return index

    def create_or_load_faiss(self):
        if os.path.exists(self.faiss_index_file):
            print('Loading Faiss index from existing file')
            faiss_index = faiss.read_index(self.faiss_index_file)
            # For accurate search
            faiss_index.nprobe = 200
        else:
            print('Creating empty Faiss index')
            faiss_index = self.create_faiss()
        if os.path.exists(self.pids_index_file):
            pids_idx = pd.read_csv(self.pids_index_file, compression='gzip')
        else:
            pids_idx = pd.DataFrame(data=[], columns=['pmid', 'chunk', 'year', 'noreview'], dtype=int)
        self.pids_idx = pids_idx
        self.faiss_index = faiss_index
        return faiss_index, pids_idx

    def store_embeddings(self, embeddings, index):
        embeddings = np.array(embeddings).astype('float32')
        if (len(embeddings.shape) == 1 or
                embeddings.shape[1] != self.embeddings_dimension or
                len(index) != embeddings.shape[0]):
            print(f'Problematic chunk embeddings, {embeddings.shape}')
            return
        t = pd.DataFrame(data=index, columns=['pmid', 'chunk'])
        self.pids_idx = pd.concat([self.pids_idx, t], ignore_index=True).reset_index(drop=True)
        self.faiss_index.add(embeddings)

    def ntotal(self):
        assert self.faiss_index.ntotal == len(self.pids_idx)
        return len(self.pids_idx)

    def save(self):
        assert len(self.pids_idx) == self.faiss_index.ntotal
        print('Storing FAISS index')
        faiss.write_index(self.faiss_index, self.faiss_index_file)
        print('Storing Ids index')
        self.pids_idx.to_csv(self.pids_index_file, index=False, compression='gzip')
