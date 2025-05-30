import logging
import os

import requests

from pysrc.services.embeddings_service import is_embeddings_service_available

logger = logging.getLogger(__name__)

# Launch with a Docker address or locally
SEMANTIC_SEARCH_SERVICE_URL = os.getenv('SEMANTIC_SEARCH_SERVICE_URL', 'http://localhost:5002')


def is_semantic_search_service_available():
    if not is_embeddings_service_available():
        logger.debug('Semantic search service is not functional without embeddings')
        return False
    logger.debug(f'Check if semantics search service endpoint is available')
    try:
        r = requests.request(url=SEMANTIC_SEARCH_SERVICE_URL, method='GET')
        return r.status_code == 200
    except Exception as e:
        logger.debug(f'Semantic search service is not available: {e}')
        return False


def fetch_semantic_search(source, text, noreviews, limit):
    logger.debug(f'Fetching semantic search results')
    args = dict(
        source=source,
        text=text,
        noreviews=noreviews,
        limit=limit
    )
    try:
        r = requests.request(
            url=f'{SEMANTIC_SEARCH_SERVICE_URL}/semantic_search',
            method='GET',
            json=args,
            headers={'Accept': 'application/json'}
        )
        if r.status_code == 200:
            return r.json()
        else:
            logger.debug(f'Wrong response code {r.status_code}')
    except Exception as e:
        logger.debug(f'Failed to fetch semantic search results ${e}')
    return None
