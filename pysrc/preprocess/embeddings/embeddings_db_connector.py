import ast
import logging

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from tqdm.auto import tqdm

from pysrc.config import PubtrendsConfig
from pysrc.papers.db.postgres_utils import ints_to_vals

config = PubtrendsConfig(test=False)

logger = logging.getLogger(__name__)


class EmbeddingsDBConnector:
    def __init__(
            self,
            host=None,
            port=None,
            database=None,
            user=None,
            password=None,
            embeddings_model_name=None,
            embedding_dimension=0
    ):
        self.connection_string = f"""
                    host={host or config.embeddings_postgres_host} \
                    port={port or config.embeddings_postgres_port} \
                    dbname={database or config.embeddings_postgres_database} \
                    user={user or config.embeddings_postgres_username} \
                    password={password or config.embeddings_postgres_password}
                """.strip()
        self.embeddings_model_name = embeddings_model_name or config.embeddings_model_name
        self.embedding_dimension = embedding_dimension or config.embeddings_dimension

    # Embeddings DB initialization
    def init_database(self):
        with psycopg2.connect(self.connection_string) as connection:
            connection.set_session(readonly=False)
            with connection.cursor() as cursor:
                cursor.execute("select * from information_schema.tables where table_name=%s",
                               (self.embeddings_model_name,))
                if cursor.rowcount:
                    return

            query = f'''
                    CREATE EXTENSION IF NOT EXISTS vector;
                    create table {self.embeddings_model_name}(
                        pmid    integer,
                        chunk   integer,
                        embedding vector({self.embedding_dimension})
                    );
                    CREATE INDEX pmid_chunk_idx_{self.embeddings_model_name}
                    ON {self.embeddings_model_name}(pmid, chunk);
                    '''
            with connection.cursor() as cursor:
                cursor.execute(query)
            connection.commit()

    def collect_ids_without_embeddings(self, pids):
        logger.debug('collect_ids_without_embeddings')
        with psycopg2.connect(self.connection_string) as connection:
            connection.set_session(readonly=True)
            vals = ints_to_vals(pids)
            query = f'''
                        SELECT pmid
                        FROM {self.embeddings_model_name} P
                        WHERE P.pmid IN (VALUES {vals});
                        '''
            logger.debug(f'query: {query[:1000]}')
            with connection.cursor() as cursor:
                cursor.execute(query)
                df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)
                pids_with_embeddings = set(df['pmid'])
                return [pid for pid in pids if pid not in pids_with_embeddings]

    def l2norm(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            norm = np.finfo(v.dtype).eps
        v /= norm
        return v

    def store_embeddings_to_postgresql(self, chunk_embeddings, chunk_idx):
        logger.debug('store_embeddings_to_postgresql')
        # Normalize embeddings if using cosine similarity
        data = [(pmid, chunk, self.l2norm(e).tolist())
                for (pmid, chunk), e in zip(chunk_idx, chunk_embeddings)]
        with psycopg2.connect(self.connection_string) as connection:
            with connection.cursor() as cursor:
                execute_values(
                    cursor,
                    f"INSERT INTO {self.embeddings_model_name} (pmid, chunk, embedding) VALUES %s",
                    data
                )
            connection.commit()

    def sample_embeddings(self, n=10_000):
        logger.debug('sample_embeddings')
        with psycopg2.connect(self.connection_string) as connection:
            with connection.cursor() as cursor:
                query = f"""
                    SELECT embedding FROM {self.embeddings_model_name}
                    LIMIT {n};
            """
                logger.debug(f'query: {query[:1000]}')
                cursor.execute(query)
                embeddings = [ast.literal_eval(row[0]) for row in tqdm(cursor.fetchall())]
                return np.array(embeddings).astype(np.float32)

    def load_embeddings_by_ids(self, pids):
        logger.debug('load_embeddings_by_ids')
        vals = ints_to_vals(pids)
        with psycopg2.connect(self.connection_string) as connection:
            with connection.cursor() as cursor:
                query = f"""
                        SELECT pmid, chunk, embedding FROM {self.embeddings_model_name}
                        WHERE pmid IN (VALUES {vals})
                        ORDER BY pmid, chunk;
                """
                logger.debug(f'query: {query[:1000]}')
                cursor.execute(query)
                result = cursor.fetchall()
                index = [(pmid, chunk) for pmid, chunk, _ in result]
                embeddings = [np.array(ast.literal_eval(row[2])) for row in result]
                return index, embeddings

    def is_embeddings_db_available(self):
        try:
            with psycopg2.connect(self.connection_string) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"select * from information_schema.tables where table_name=%s",
                                   (self.embeddings_model_name,))
                    return cursor.rowcount > 0
        except Exception:
            return False
