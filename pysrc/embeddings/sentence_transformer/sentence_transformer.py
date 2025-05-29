import concurrent
import logging
import multiprocessing
import numpy as np
import os
import requests
import torch

from lazy import lazy
from math import ceil
from more_itertools import sliced
from sentence_transformers import SentenceTransformer
from threading import Lock

from pysrc.config import PubtrendsConfig

logger = logging.getLogger(__name__)

MODEL_PATHS = ['/sentence-transformers', os.path.expanduser('~/.pubtrends/sentence-transformers')]
for p in MODEL_PATHS:
    if os.path.isdir(p):
        # Configure HuggingFace models cache directory
        os.environ['HF_HOME'] = p
        break
else:
    raise RuntimeError('Failed to configure model cache directory')



class SentenceTransformerModel:

    def __init__(self):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model_name = PubtrendsConfig().sentence_transformer_model

    @lazy
    def download_and_load_model(self):
        logger.info(f'Loading model into {self.device}')
        # Superfast general purpose, acceptable for biomedical texts
        self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def encode(self, texts, show_progress_bar=False):
        return self.model.encode(texts, device=self.device, show_progress_bar=show_progress_bar)

    def encode_parallel(self, texts, max_workers = multiprocessing.cpu_count()):
        if self.device != 'cpu':
            return self.model.encode(texts, show_progress_bar=False)
        # Compute parallel on different threads, since we use the same fasttext model
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(lambda ts: self.model.encode(ts, show_progress_bar=False), ts)
                for ts in sliced(texts, int(ceil(len(texts) / max_workers)))
            ]
            # Important: keep order of results!!!
            return np.vstack([future.result() for future in futures])



SENTENCE_TRANSFORMER_MODEL_CACHE = SentenceTransformerModel()

SENTENCE_TRANSFORMER_MODEL_CACHE_LOCK = Lock()