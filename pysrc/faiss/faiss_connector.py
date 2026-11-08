import logging
import os
import re

import faiss
import numpy as np
import pandas as pd

from pysrc.config import PubtrendsConfig

config = PubtrendsConfig(test=False)

logger = logging.getLogger(__name__)

FAISS_PATHS = ['/faiss', os.path.expanduser('~/.pubtrends/faiss')]
for p in FAISS_PATHS:
    if os.path.isdir(p):
        faiss_path = p
        break
else:
    # Ensure default directory exists
    default_path = os.path.expanduser('~/.pubtrends/faiss')
    os.makedirs(default_path, exist_ok=True)
    faiss_path = default_path


class FaissConnector:
    def __init__(self, source, embeddings_model_name, embeddings_dimension, create=False, exact=False):
        self.embeddings_dimension = embeddings_dimension
        self.create = create
        self.exact = exact
        folder_name = re.sub('[^a-zA-Z0-9]', '_', embeddings_model_name)
        self.faiss_dir = f'{faiss_path}/faiss_{source}_{folder_name}'
        if not os.path.exists(self.faiss_dir):
            if not self.create:
                raise Exception(f'Faiss directory {self.faiss_dir} does not exist')
            os.makedirs(self.faiss_dir)
        self.faiss_index_file = os.path.expanduser(f'{self.faiss_dir}/embeddings.index')
        self.pids_index_file = os.path.expanduser(f'{self.faiss_dir}/pids.pq')
        # Present while the index is being populated from scratch, so that interrupted rebuild resumes
        self.rebuild_marker_file = os.path.expanduser(f'{self.faiss_dir}/rebuild_in_progress')
        self.faiss_index = None
        self._pids_idx = None
        # Ids of added embeddings, concatenated lazily: concat on each add is quadratic for ~100M rows
        self._pending_pids = []

    @property
    def pids_idx(self):
        if self._pending_pids:
            self._pids_idx = pd.concat([self._pids_idx, *self._pending_pids], ignore_index=True)
            self._pending_pids = []
        return self._pids_idx

    def create_faiss(self):
        if self.exact:
            print('Exact search index')
            index = faiss.IndexFlatL2(self.embeddings_dimension)
        else:
            print(f'Approximate search index {config.faiss_index_factory}')
            index = faiss.index_factory(self.embeddings_dimension, config.faiss_index_factory, faiss.METRIC_L2)
        self.configure_search(index)
        return index

    @staticmethod
    def configure_search(index):
        if faiss.try_extract_index_ivf(index) is None:
            return
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, 'nprobe', config.faiss_nprobe)
        if isinstance(faiss.downcast_index(faiss.extract_index_ivf(index).quantizer), faiss.IndexHNSW):
            # HNSW coarse quantizer returns at most efSearch nearest lists, also used to assign lists on add
            ps.set_index_parameter(index, 'quantizer_efSearch', 2 * config.faiss_nprobe)

    def training_size(self):
        """Returns (minimal, recommended) number of training embeddings, 0 if training is not required"""
        ivf = faiss.try_extract_index_ivf(self.faiss_index)
        if ivf is None:
            return 0, 0
        # Faiss warns about less than 39 points per cluster, k-means uses up to 256 points per cluster
        return 39 * ivf.nlist, 64 * ivf.nlist

    def train(self, embeddings):
        embeddings = np.ascontiguousarray(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings)
        ivf = faiss.try_extract_index_ivf(self.faiss_index)
        ngpus = faiss.get_num_gpus()
        if ivf is not None and ngpus > 0:
            # Coarse k-means is the most expensive part of training, run its assignment step on GPUs
            print(f'Training coarse quantizer k-means on {ngpus} GPUs')
            # Keep a Python reference, the C++ field doesn't own it
            gpu_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(self.embeddings_dimension))
            ivf.clustering_index = gpu_index
        try:
            self.faiss_index.train(embeddings)
        finally:
            if ivf is not None:
                ivf.clustering_index = None

    def create_or_load_faiss(self, mmap=False):
        """mmap: map inverted lists instead of reading into memory, read-only,
        lets processes share a single copy of the index in page cache"""
        if os.path.exists(self.faiss_index_file):
            print(f'Loading Faiss index from existing file {self.faiss_index_file}')
            flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY if mmap else 0
            faiss_index = faiss.read_index(self.faiss_index_file, flags)
            self.configure_search(faiss_index)
        else:
            if not self.create:
                raise Exception(f'Faiss index file {self.faiss_index_file} does not exist')
            print(f'Creating empty Faiss index {self.faiss_index_file}')
            faiss_index = self.create_faiss()
        if os.path.exists(self.pids_index_file):
            print(f'Loading Ids index from existing file {self.pids_index_file}')
            pids_idx = pd.read_parquet(self.pids_index_file)
        else:
            if not self.create:
                raise Exception(f'Ids index file {self.pids_index_file} does not exist')
            pids_idx = pd.DataFrame(data=[], columns=['pmid', 'chunk'], dtype=int)
            print(f'Creating empty Ids index {self.pids_index_file}')
        self._pids_idx = pids_idx
        self._pending_pids = []
        self.faiss_index = faiss_index
        return faiss_index, pids_idx

    def store_embeddings(self, index, embeddings):
        if len(index) == 0 or len(embeddings) == 0:
            return
        embeddings = np.array(embeddings).astype('float32')
        if (len(embeddings.shape) == 1 or
                embeddings.shape[1] != self.embeddings_dimension or
                len(index) != embeddings.shape[0]):
            print(f'Problematic chunk embeddings, {embeddings.shape}')
            return
        # Normalize, same as embeddings DB and search queries, so that L2 distance ranks as cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        self._pending_pids.append(pd.DataFrame(data=index, columns=['pmid', 'chunk']))

    def save(self):
        if len(self.pids_idx) != self.faiss_index.ntotal:
            raise RuntimeError(
                f"Index/FAISS size mismatch: {len(self.pids_idx)} pids vs {self.faiss_index.ntotal} vectors"
            )

        # Write to temporary files first for atomic operation
        faiss_temp_file = f'{self.faiss_index_file}.tmp'
        pids_temp_file = f'{self.pids_index_file}.tmp'

        try:
            print(f'Storing FAISS index to temporary file {faiss_temp_file}')
            faiss.write_index(self.faiss_index, faiss_temp_file)

            print(f'Storing Ids index to temporary file {pids_temp_file} with {len(self.pids_idx)} rows')
            self.pids_idx.to_parquet(pids_temp_file, index=False, compression='gzip')

            # Atomic rename operations - both files written successfully
            print(f'Committing FAISS index to {self.faiss_index_file}')
            os.rename(faiss_temp_file, self.faiss_index_file)

            print(f'Committing Ids index to {self.pids_index_file}')
            os.rename(pids_temp_file, self.pids_index_file)

            print('Save completed successfully')
        except Exception as e:
            # Clean up temporary files on failure
            print(f'Save failed: {e}')
            if os.path.exists(faiss_temp_file):
                os.remove(faiss_temp_file)
            if os.path.exists(pids_temp_file):
                os.remove(pids_temp_file)
            raise
