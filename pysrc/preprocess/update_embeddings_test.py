import sys
import os

sys.path.append(os.path.expanduser('~/work/pubtrends'))


from pysrc.endpoints.embeddings.sentence_transformer.sentence_transformer import SentenceTransformerModel

print('Loading model')
sentence_transformer_model = SentenceTransformerModel()
sentence_transformer_model.download_and_load_model
emb = sentence_transformer_model.encode(['This is a test.', 'This is a test2'])
print(emb.shape)

text_embedding = lambda t: sentence_transformer_model.encode(t)
batch_texts_embeddings = lambda t: sentence_transformer_model.encode(t)
embeddings_model = sentence_transformer_model

from pysrc.preprocess.embeddings.faiss_connector import FaissConnector
from pysrc.preprocess.embeddings.embeddings_db_connector import EmbeddingsDBConnector
from pysrc.preprocess.embeddings.embeddings_model_connector import EmbeddingsModelConnector
from pysrc.preprocess.embeddings.publications_db_connector import PublicationsDBConnector

publications_db_connector = PublicationsDBConnector()
embeddings_model_connector = EmbeddingsModelConnector()

embeddings_db_connector = EmbeddingsDBConnector(
    host='localhost',
    port=5430,
    database='pubtrends',
    user='biolabs',
    password='mysecretpassword',
    embeddings_model_name=embeddings_model_connector.embeddings_model_name,
    embedding_dimension=embeddings_model_connector.embeddings_dimension
)
embeddings_db_connector.init_database()

faiss_connector = FaissConnector(
    embeddings_model_name=embeddings_model_connector.embeddings_model_name,
    embeddings_dimension=embeddings_model_connector.embeddings_dimension
)

print('Create embeddings and faiss with training')
from pysrc.preprocess.update_embeddings import update_embeddings

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

update_embeddings(
    publications_db_connector=publications_db_connector,
    embeddings_model_connector=embeddings_model_connector,
    embeddings_db_connector=embeddings_db_connector,
    faiss_connector=faiss_connector,
    max_year=2025, min_year=2024, test_limit_per_year=10_000
)

print('Update only embeddings DB')
from preprocess.update_embeddings import update_embeddings

update_embeddings(
    publications_db_connector=publications_db_connector,
    embeddings_model_connector=embeddings_model_connector,
    embeddings_db_connector=embeddings_db_connector,
    faiss_connector=None,
    max_year=2024, min_year=2023, test_limit_per_year=10_000
)

print('Update faiss from database')
from preprocess.update_embeddings import update_faiss

update_faiss(
    publications_db_connector=publications_db_connector,
    embeddings_model_connector=embeddings_model_connector,
    embeddings_db_connector=embeddings_db_connector,
    faiss_connector=faiss_connector,
    max_year=2024, min_year=2023, test_limit_per_year=10_000
)

print('Update both embeddings DB and faiss')
from preprocess.update_embeddings import update_embeddings

update_embeddings(
    publications_db_connector=publications_db_connector,
    embeddings_model_connector=embeddings_model_connector,
    embeddings_db_connector=embeddings_db_connector,
    faiss_connector=faiss_connector,
    max_year=2023, min_year=2022, test_limit_per_year=10_000
)