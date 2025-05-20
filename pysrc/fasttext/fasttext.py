import logging
import os
from threading import Lock

import numpy as np
import requests
from gensim.models import KeyedVectors
from lazy import lazy

logger = logging.getLogger(__name__)

# File URL
MODEL_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin"

MODEL_PATHS = ['/model', os.path.expanduser('~/.pubtrends/model')]
for p in MODEL_PATHS:
    if os.path.isdir(p):
        model_dir = p
        break
else:
    raise RuntimeError('Failed to configure model directory')



class PretrainedModelCache:
    @lazy
    def download_and_load_model(self):
        # Extract filename from URL
        model_name = os.path.basename(MODEL_URL)
        filename = os.path.join(model_dir, model_name)

        logger.info(f'Loading {model_name} fasttext model for biomedical texts')

        # Check if file already exists
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            response = requests.get(MODEL_URL, stream=True)

            if response.status_code == 200:
                with open(filename, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                logger.info(f"Download complete: {filename}")
            else:
                logger.error(f"Failed to download file. Status code: {response.status_code}")
        else:
            logger.info(f"File already exists: {filename}")

        model = KeyedVectors.load_word2vec_format(filename, binary=True)
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

def text_embedding_fasttext(text):
    logger.info('Compute text embedding using pretrained fasttext model')
    model_instance = PRETRAINED_MODEL_CACHE.download_and_load_model
    return np.mean([
        model_instance.get_vector(t) if model_instance.has_index_for(t)
        else np.zeros(model_instance.vector_size)  # Support out-of-dictionary missing embeddings
        for t in text.split()
    ], axis=0).tolist()
