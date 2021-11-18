import logging
from threading import Lock

import gensim.downloader as api
import numpy as np
from lazy import lazy

logger = logging.getLogger(__name__)


class PretrainedModelCache:
    @lazy
    def download_and_load_model(self):
        model_name = 'fasttext-wiki-news-subwords-300'
        logger.info(f'Loading {model_name} fasttext model by facebook')
        model = api.load(model_name)
        logger.info('Successfully loaded model')
        return model


PRETRAINED_MODEL_CACHE = PretrainedModelCache()

PRETRAINED_MODEL_CACHE_LOCK = Lock()


def tokens_embeddings_fasttext(corpus_tokens):
    logger.info('Compute words embeddings using pretrained fasttext model')
    model = PRETRAINED_MODEL_CACHE.download_and_load_model
    logger.info('Retrieve word embeddings')
    return np.array([
        model.get_vector(t) if model.has_index_for(t)
        else np.zeros(model.vector_size)  # Support out-of-dictionary missing embeddings
        for t in corpus_tokens
    ])
