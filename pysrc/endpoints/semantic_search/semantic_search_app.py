import json
import logging
import os

from flask import Flask, request
from flasgger import Swagger, swag_from

from pysrc.endpoints.semantic_search.semantic_search import SEMANTIC_SEARCH

semantic_search_app = Flask(__name__)
Swagger(semantic_search_app)

#####################
# Configure logging #
#####################

# Deployment and development
LOG_PATHS = ['/logs', os.path.expanduser('~/.pubtrends/logs')]
for p in LOG_PATHS:
    if os.path.isdir(p):
        logfile = os.path.join(p, 'semantic_search_app.log')
        break
else:
    raise RuntimeError('Failed to configure main log file')

logging.basicConfig(filename=logfile,
                    filemode='a',
                    format='[%(asctime)s,%(msecs)03d: %(levelname)s/%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Check to see if our Flask application is being run directly or through Gunicorn,
# and then set your Flask application log level the same as Gunicorn’s.
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    semantic_search_app.logger.setLevel(gunicorn_logger.level)

logger = semantic_search_app.logger


@semantic_search_app.route('/semantic_search', methods=['POST'])
def semantic_search():
    """
    Perform semantic search on scientific papers
    ---
    tags:
      - Semantic Search
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - source
            - text
            - noreviews
            - limit
          properties:
            source:
              type: string
              description: Data source (e.g., "Pubmed")
              example: "Pubmed"
            text:
              type: string
              description: Search query text
              example: "cancer treatment"
            noreviews:
              type: boolean
              description: Exclude review articles
              example: true
            minyear:
              type: integer
              description: Minimum publication year (optional)
              example: 2020
            maxyear:
              type: integer
              description: Maximum publication year (optional)
              example: 2024
            limit:
              type: integer
              description: Maximum number of results to return
              example: 100
    responses:
      200:
        description: List of search results with PMIDs and similarity scores
        schema:
          type: array
          items:
            type: object
            properties:
              pmid:
                type: string
                description: PubMed ID
              similarity:
                type: number
                format: float
                description: Similarity score
    """
    # Please ensure that already initialized. Otherwise, model loading may take some time
    data = request.get_json()
    logger.info('Search')
    source = data['source']
    text = data['text']
    noreviews = data['noreviews']
    min_year = data.get('minyear', None)
    max_year = data.get('maxyear', None)
    limit = data['limit']
    logger.info(f'Search parameters: source={source}, text={text}, noreviews={noreviews}, '
                f'min_year={min_year}, max_year={max_year}, limit={limit}')
    result = SEMANTIC_SEARCH.search(source, text, noreviews, min_year, max_year, limit)
    logger.info(f'Return semantic search results in JSON format {len(result)}')
    return json.dumps(result)


@semantic_search_app.route('/semantic_search_embeddings', methods=['POST'])
def semantic_search_embeddings():
    """
    Perform semantic search using pre-computed embeddings
    ---
    tags:
      - Semantic Search
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - source
            - embeddings
            - noreviews
            - limit
          properties:
            source:
              type: string
              description: Data source (e.g., "Pubmed")
              example: "Pubmed"
            embeddings:
              type: string
              description: JSON-encoded array of embedding vectors
              example: "[0.1, 0.2, 0.3, ...]"
            noreviews:
              type: boolean
              description: Exclude review articles
              example: true
            minyear:
              type: integer
              description: Minimum publication year (optional)
              example: 2020
            maxyear:
              type: integer
              description: Maximum publication year (optional)
              example: 2024
            limit:
              type: integer
              description: Maximum number of results to return
              example: 100
    responses:
      200:
        description: List of search results with PMIDs and similarity scores
        schema:
          type: array
          items:
            type: object
            properties:
              pmid:
                type: string
                description: PubMed ID
              similarity:
                type: number
                format: float
                description: Similarity score
      500:
        description: Error processing request
        schema:
          type: object
          properties:
            error:
              type: string
              description: Error message
    """
    # Please ensure that already initialized. Otherwise, model loading may take some time
    try:
        data = request.get_json()
        logger.info('Search embeddings')
        logger.info(f'Request data keys: {list(data.keys())}')
        source = data['source']
        embeddings = json.loads(data['embeddings'])
        noreviews = data['noreviews']
        min_year = data.get('minyear', None)
        max_year = data.get('maxyear', None)
        limit = data['limit']
        logger.info(f'Search parameters: source={source}, noreviews={noreviews}, '
                    f'min_year={min_year}, max_year={max_year}, limit={limit}')
        result = SEMANTIC_SEARCH.search_embeddings(source, embeddings, noreviews, min_year, max_year, limit)
        logger.info(f'Return semantic search results in JSON format {len(result)}')
        return json.dumps(result)
    except Exception as e:
        logger.error(f'Error in semantic_search_embeddings: {e}', exc_info=True)
        return json.dumps({'error': str(e)}), 500


@semantic_search_app.route('/', methods=['GET'])
def index():
    return f'Semantic search with FAISS DB'


# Application
def get_app():
    return semantic_search_app


# With debug=True, the Flask server will auto-reload on changes
if __name__ == '__main__':
    semantic_search_app.run(host='0.0.0.0', debug=True, port=5002)
