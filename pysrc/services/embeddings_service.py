import logging
import os

import numpy as np
import requests
from sklearn.preprocessing import normalize

from pysrc.preprocess.embeddings.embeddings_db_connector import EmbeddingsDBConnector

logger = logging.getLogger(__name__)

# Launch with a Docker address or locally
EMBEDDINGS_SERVICE_URL = os.getenv('EMBEDDINGS_SERVICE_URL', 'http://localhost:5001')

EMBEDDINGS_DB_CONNECTOR = EmbeddingsDBConnector()


def is_embeddings_service_available():
    logger.debug(f'Check if embeddings service endpoint is available')
    try:
        r = requests.request(url=EMBEDDINGS_SERVICE_URL, method='GET')
        return r.status_code == 200
    except Exception as e:
        logger.debug(f'Embeddings service is not available: {e}')
        return False


def is_embeddings_service_ready():
    logger.debug(f'Check if embeddings service endpoint is ready')
    try:
        r = requests.request(url=EMBEDDINGS_SERVICE_URL, method='GET')
        if r.status_code != 200:
            return False
        r = requests.request(url=f'{EMBEDDINGS_SERVICE_URL}/check', method='GET',
                             headers={'Accept': 'application/json'})
        if r.status_code != 200 or r.json() is not True:
            return False
        return True
    except Exception as e:
        logger.debug(f'Embeddings service is not ready: {e}')
        return False


def fetch_tokens_embeddings(tokens):
    # Don't use the model as is, since each celery process will load its own copy.
    # Shared model is available via additional service with a single model.
    logger.debug(f'Fetch tokens embeddings')
    try:
        r = requests.request(
            url=f'{EMBEDDINGS_SERVICE_URL}/embeddings_tokens',
            method='GET',
            json=tokens,
            headers={'Accept': 'application/json'}
        )
        if r.status_code == 200:
            return np.array(r.json()).reshape(len(tokens), -1)
        else:
            logger.debug(f'Wrong response code {r.status_code}')
    except Exception as e:
        logger.debug(f'Failed to fetch tokens embeddings ${e}')
    return None


def is_texts_embeddings_available():
    return fetch_texts_embedding(['Test sentence 1.', 'Test sentence 2.']) is not None


def fetch_texts_embedding(texts):
    # Don't use the model as is, since each celery process will load its own copy.
    # Shared model is available via additional service with a single model.
    logger.debug(f'Fetch texts embeddings')
    try:
        r = requests.request(
            url=f'{EMBEDDINGS_SERVICE_URL}/embeddings_texts',
            method='GET',
            json=texts,
            headers={'Accept': 'application/json'}
        )
        if r.status_code == 200:
            return np.array(r.json()).reshape(len(texts), -1)
        else:
            logger.debug(f'Wrong response code {r.status_code}')
    except Exception as e:
        logger.debug(f'Failed to fetch texts embeddings ${e}')
    return None


def is_embeddings_db_available():
    logger.debug(f'Check if embeddings db is available')
    return EMBEDDINGS_DB_CONNECTOR.is_embeddings_db_available()


def load_embeddings_from_df(pids):
    logger.debug("Fetching embeddings from DB")
    index, embeddings = EMBEDDINGS_DB_CONNECTOR.load_embeddings_by_ids(pids)
    return normalize(np.array(embeddings).reshape(len(index), -1)), [(str(p), c) for p, c in index]
