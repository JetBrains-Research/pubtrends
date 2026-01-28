import logging
import os

import torch
from lazy import lazy
from sentence_transformers import SentenceTransformer

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
        # noinspection PyStatementEffect
        self.download_and_load_model

    @lazy
    def download_and_load_model(self):
        logger.info(f'Loading model {self.model_name} into {self.device}')
        # Superfast general purpose, acceptable for biomedical texts
        self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def encode(self, texts, max_workers = 4, show_progress_bar=False):
        if self.device != 'cpu':
            return self.model.encode(
                texts, device=self.device, show_progress_bar=show_progress_bar
            )
        else:
            return self.model.encode(
                texts, batch_size=max_workers, device=self.device, show_progress_bar=show_progress_bar
            )

logger.info('Prepare embeddings pretrained model')
SENTENCE_TRANSFORMER_MODEL = SentenceTransformerModel()
logger.info('Model is ready')