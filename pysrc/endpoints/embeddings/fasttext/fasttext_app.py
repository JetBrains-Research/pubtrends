import json
import logging
import os
import threading

from flask import Flask, request

from pysrc.endpoints.embeddings.fasttext.fasttext import FASTTEXT_MODEL_CACHE, FASTTEXT_MODEL_CACHE_LOCK

fasttext_app = Flask(__name__)

#####################
# Configure logging #
#####################

# Deployment and development
LOG_PATHS = ['/logs', os.path.expanduser('~/.pubtrends/logs')]
for p in LOG_PATHS:
    if os.path.isdir(p):
        logfile = os.path.join(p, 'fasttext_app.log')
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
    fasttext_app.logger.setLevel(gunicorn_logger.level)

logger = fasttext_app.logger


def init_fasttext_model():
    logger.info('Prepare embeddings pretrained model')
    # noinspection PyUnusedLocal,PyStatementEffect
    FASTTEXT_MODEL_CACHE.download_and_load_model
    logger.info('Model is ready')
    fasttext_app.config['LOADED'] = True


@fasttext_app.route('/check', methods=['GET'])
def check():
    if fasttext_app.config.get('LOADED', False):
        return json.dumps(True)
    if fasttext_app.config.get('LOADING', False):
        return json.dumps(False)
    try:
        FASTTEXT_MODEL_CACHE_LOCK.acquire()
        if fasttext_app.config.get('LOADED', False):
            return json.dumps(True)
        if fasttext_app.config.get('LOADING', False):
            return json.dumps(False)
        fasttext_app.config['LOADING'] = True
        threading.Thread(target=init_fasttext_model, daemon=True).start()
        return json.dumps(False)
    finally:
        FASTTEXT_MODEL_CACHE_LOCK.release()


@fasttext_app.route('/embeddings_tokens', methods=['GET'])
def embeddings_tokens():
    # Please ensure that already initialized. Otherwise, model loading may take some time
    corpus_tokens = request.get_json()
    logger.info('Computing embeddings')
    embeddings = FASTTEXT_MODEL_CACHE.tokens_embeddings_fasttext(corpus_tokens).tolist()
    # Even if /check wasn't invoked, mark the model as ready
    fasttext_app.config['LOADED'] = True
    logger.info(f'Return embeddings in JSON format')
    return json.dumps(embeddings)

@fasttext_app.route('/', methods=['GET'])
def index():
    return 'Words embedding with https://github.com/ncbi-nlp/BioSentVec'


# Application
def get_app():
    return fasttext_app


# With debug=True, the Flask server will auto-reload on changes
if __name__ == '__main__':
    fasttext_app.run(host='0.0.0.0', debug=False, port=5001)
