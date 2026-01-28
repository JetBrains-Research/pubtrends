import json
import logging
import os

from flask import Flask, request

from pysrc.endpoints.semantic_search.semantic_search import SEMANTIC_SEARCH

semantic_search_app = Flask(__name__)

#####################
# Configure logging #
#####################

# Deployment and development
LOG_PATHS = ['/logs', os.path.expanduser('~/.pubk/logs')]
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


@semantic_search_app.route('/semantic_search', methods=['GET'])
def semantic_search():
    # Please ensure that already initialized. Otherwise, model loading may take some time
    data = request.get_json()
    logger.info('Search')
    source = data['source']
    text = data['text']
    noreviews = data['noreviews']
    min_year = data.get('minyear', None)
    max_year = data.get('maxyear', None)
    limit = data['limit']
    result = SEMANTIC_SEARCH.search(source, text, noreviews, min_year, max_year, limit)
    logger.info(f'Return semantic search results in JSON format {len(result)}')
    return json.dumps(result)


@semantic_search_app.route('/semantic_search_embeddings', methods=['GET'])
def semantic_search_embeddings():
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
