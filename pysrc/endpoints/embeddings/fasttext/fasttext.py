import logging
import os
from threading import Lock

import numpy as np
import requests
from gensim.models import KeyedVectors
from lazy import lazy

logger = logging.getLogger(__name__)

# File URL
FASTTEXT_MODEL_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin"

MODEL_PATHS = ['/fasttext', os.path.expanduser('~/.pubtrends/fasttext')]
for p in MODEL_PATHS:
    if os.path.isdir(p):
        model_dir = p
        break
else:
    raise RuntimeError('Failed to configure model directory')



class FastTextModelCache:
    @lazy
    def download_and_load_model(self):
        # Extract filename from URL
        model_name = os.path.basename(FASTTEXT_MODEL_URL)
        filename = os.path.join(model_dir, model_name)

        logger.info(f'Loading {model_name} fasttext model for biomedical texts')

        # Check if the file already exists
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            response = requests.get(FASTTEXT_MODEL_URL, stream=True)

            if response.status_code == 200:
                with open(filename, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                logger.info(f"Download complete: {filename}")
            else:
                logger.error(f"Failed to download file. Status code: {response.status_code}")
        else:
            logger.info(f"File already exists: {filename}")

        self.model = KeyedVectors.load_word2vec_format(filename, binary=True)
        logger.info('Successfully loaded model')
        return self

    def tokens_embeddings_fasttext(self, tokens):
        logger.info('Compute words embeddings using pretrained fasttext model')
        return np.array([
            self.model.get_vector(t) if self.model.has_index_for(t)
            else np.zeros(self.model.vector_size)  # Support out-of-dictionary missing embeddings
            for t in tokens
        ])

FASTTEXT_MODEL_CACHE = FastTextModelCache()

FASTTEXT_MODEL_CACHE_LOCK = Lock()



