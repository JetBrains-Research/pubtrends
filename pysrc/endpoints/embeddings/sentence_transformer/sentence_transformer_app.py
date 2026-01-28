import json
import logging
import os

from flask import Flask, request

from pysrc.endpoints.embeddings.sentence_transformer.sentence_transformer import SENTENCE_TRANSFORMER_MODEL

sentence_transformer_app = Flask(__name__)

#####################
# Configure logging #
#####################

# Deployment and development
LOG_PATHS = ['/logs', os.path.expanduser('~/.pubtrends/logs')]
for p in LOG_PATHS:
    if os.path.isdir(p):
        logfile = os.path.join(p, 'sentence_transformer_app.log')
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
    sentence_transformer_app.logger.setLevel(gunicorn_logger.level)

logger = sentence_transformer_app.logger


@sentence_transformer_app.route('/embeddings_texts', methods=['GET'])
def embeddings_texts():
    # Please ensure that already initialized. Otherwise, model loading may take some time
    texts = request.get_json()
    logger.info('Computing texts embeddings')
    embeddings = SENTENCE_TRANSFORMER_MODEL.encode(texts).tolist()
    # Even if /check wasn't invoked, mark the model as ready
    sentence_transformer_app.config['LOADED'] = True
    logger.info(f'Return embeddings in JSON format')
    return json.dumps(embeddings)


@sentence_transformer_app.route('/', methods=['GET'])
def index():
    return f'Embedding with Sequence Transformer model {SENTENCE_TRANSFORMER_MODEL.model_name}'


# Application
def get_app():
    return sentence_transformer_app


# With debug=True, the Flask server will auto-reload on changes
if __name__ == '__main__':
    sentence_transformer_app.run(host='0.0.0.0', debug=False, port=5001)
