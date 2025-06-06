import os

from pysrc.preprocess.embeddings.embeddings_db_connector import EmbeddingsDBConnector
from pysrc.preprocess.embeddings.embeddings_model_connector import EmbeddingsModelConnector
from pysrc.preprocess.embeddings.faiss_connector import FaissConnector
from pysrc.preprocess.embeddings.publications_db_connector import PublicationsDBConnector
from pysrc.preprocess.update_embeddings import update_embeddings
from pysrc.preprocess.update_embeddings import update_faiss

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Please ensure that pgvector postgres is runnning
# docker run --rm --name pgvector -p 5430:5432 \
#         -m 32G \
#         -e POSTGRES_USER=biolabs -e POSTGRES_PASSWORD=mysecretpassword \
#         -e POSTGRES_DB=pubtrends \
#         -v ~/pgvector/:/var/lib/postgresql/data \
#         -e PGDATA=/var/lib/postgresql/data/pgdata \
#         -d pgvector/pgvector:pg17

if __name__ == '__main__':
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
    update_embeddings(
        publications_db_connector=publications_db_connector,
        embeddings_model_connector=embeddings_model_connector,
        embeddings_db_connector=embeddings_db_connector,
        faiss_connector=faiss_connector,
        max_year=2025, min_year=2024, test_limit_per_year=10_000
    )
    print(faiss_connector.ntotal())
    print(embeddings_db_connector.ntotal())
    assert faiss_connector.ntotal() == embeddings_db_connector.ntotal()

    print('Update only embeddings DB')
    update_embeddings(
        publications_db_connector=publications_db_connector,
        embeddings_model_connector=embeddings_model_connector,
        embeddings_db_connector=embeddings_db_connector,
        faiss_connector=None,
        max_year=2024, min_year=2023, test_limit_per_year=10_000
    )
    print(faiss_connector.ntotal())
    print(embeddings_db_connector.ntotal())
    assert faiss_connector.ntotal() < embeddings_db_connector.ntotal()

    print('Update faiss from database')
    update_faiss(
        publications_db_connector=publications_db_connector,
        embeddings_model_connector=embeddings_model_connector,
        embeddings_db_connector=embeddings_db_connector,
        faiss_connector=faiss_connector,
        max_year=2024, min_year=2023
    )
    print(faiss_connector.ntotal())
    print(embeddings_db_connector.ntotal())
    assert faiss_connector.ntotal() == embeddings_db_connector.ntotal()

    print('Update both embeddings DB and faiss')
    update_embeddings(
        publications_db_connector=publications_db_connector,
        embeddings_model_connector=embeddings_model_connector,
        embeddings_db_connector=embeddings_db_connector,
        faiss_connector=faiss_connector,
        max_year=2023, min_year=2022, test_limit_per_year=10_000
    )
    print(faiss_connector.ntotal())
    print(embeddings_db_connector.ntotal())
    assert faiss_connector.ntotal() == embeddings_db_connector.ntotal()
    print('Done')
