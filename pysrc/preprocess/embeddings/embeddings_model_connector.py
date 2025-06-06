import re
import logging

from pysrc.config import PubtrendsConfig
from pysrc.endpoints.embeddings.sentence_transformer.sentence_transformer import SentenceTransformerModel

config = PubtrendsConfig(test=False)

logger = logging.getLogger(__name__)


class EmbeddingsModelConnector:
    def __init__(self):
        self.sentence_transformer_model = SentenceTransformerModel()
        # Init lazy model
        # noinspection PyStatementEffect
        self.sentence_transformer_model.download_and_load_model

        self.text_embedding = lambda t: self.sentence_transformer_model.encode(t)
        self.batch_texts_embeddings = lambda t: self.sentence_transformer_model.encode(t)
        self.embeddings_model_name = re.sub('[^a-zA-Z0-9]+', '_', self.sentence_transformer_model.model_name)
        self.device = self.sentence_transformer_model.device
        emb = self.sentence_transformer_model.encode(['This is a test.', 'This is a test2'])
        self.embeddings_dimension = emb.shape[1]
