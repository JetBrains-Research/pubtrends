import logging
import os
import requests

import numpy as np

logger = logging.getLogger(__name__)

# Launch with a Docker address or locally
EMBEDDINGS_SERVICE_URL = os.getenv('EMBEDDINGS_SERVICE_URL', 'http://localhost:5001')


def is_embeddings_service_ready():
    logger.debug(f'Check if embeddings service endpoint is ready')
    try:
        r = requests.request(url=EMBEDDINGS_SERVICE_URL, method='GET')
        if r.status_code != 200:
            return False
        r = requests.request(url=f'{EMBEDDINGS_SERVICE_URL}/check', method='GET', headers={'Accept': 'application/json'})
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


def fetch_text_embedding(text):
    # Don't use the model as is, since each celery process will load its own copy.
    # Shared model is available via additional service with a single model.
    logger.debug(f'Fetch text embedding')
    try:
        r = requests.request(
            url=f'{EMBEDDINGS_SERVICE_URL}/embeddings_text',
            method='GET',
            json=text,
            headers={'Accept': 'application/json'}
        )
        if r.status_code == 200:
            return r.json()
        else:
            logger.debug(f'Wrong response code {r.status_code}')
    except Exception as e:
        logger.debug(f'Failed to fetch text embedding ${e}')
    return None
