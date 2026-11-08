import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
from tqdm.auto import tqdm

from pysrc.papers.utils import l2norm
from pysrc.config import PubtrendsConfig

config = PubtrendsConfig(test=False)

logger = logging.getLogger(__name__)


def parse_vector(s):
    """Parse pgvector text representation '[1.0,2.0,...]', way faster than ast.literal_eval"""
    return np.fromstring(s[1:-1], dtype=np.float32, sep=',')


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
        self._pool = None

    @property
    def pool(self):
        if self._pool is None:
            self._pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.connection_string
            )
        return self._pool

    @contextmanager
    def get_connection(self, readonly=False):
        conn = self.pool.getconn()
        try:
            conn.set_session(readonly=readonly)
            yield conn
        finally:
            self.pool.putconn(conn)

    def close(self):
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None

    # Embeddings DB initialization
    def init_database(self):
        table = self.embeddings_model_name
        unique_idx = f'pmid_chunk_uidx_{table}'[:63]  # Postgres truncates identifiers to 63 chars
        with self.get_connection(readonly=False) as connection:
            with connection.cursor() as cursor:
                cursor.execute("select 1 from pg_indexes where indexname=%s", (unique_idx,))
                if cursor.fetchone():
                    return

            with connection.cursor() as cursor:
                cursor.execute(f'''
                    CREATE EXTENSION IF NOT EXISTS vector;
                    CREATE TABLE IF NOT EXISTS {table}(
                        pmid    integer,
                        chunk   integer,
                        embedding vector({self.embedding_dimension})
                    );
                    ''')
                # One-time migration of tables created without a unique index: drop duplicates first
                logger.info(f'Removing duplicate (pmid, chunk) rows from {table}')
                cursor.execute(f'''
                    DELETE FROM {table} a USING {table} b
                    WHERE a.pmid = b.pmid AND a.chunk = b.chunk AND a.ctid < b.ctid;
                    ''')
                logger.info(f'Removed {cursor.rowcount} duplicate rows')
                cursor.execute(f'''
                    CREATE UNIQUE INDEX {unique_idx} ON {table}(pmid, chunk);
                    DROP INDEX IF EXISTS pmid_chunk_idx_{table};
                    ''')
            connection.commit()

    def _query_pmids_batch(self, connection, pids_list):
        try:
            with connection.cursor() as cursor:
                query = f"""
                        SELECT pmid
                        FROM {self.embeddings_model_name} P
                        WHERE P.pmid = ANY (%s);
                        """
                logger.debug(f'query: {query[:1000]}')
                cursor.execute(query, (pids_list,))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            try:
                connection.rollback()
            except Exception:
                pass
            if len(pids_list) > 1:
                mid = len(pids_list) // 2
                left = self._query_pmids_batch(connection, pids_list[:mid])
                right = self._query_pmids_batch(connection, pids_list[mid:])
                return left + right
            else:
                logger.warning(f"Failed to check embedding presence for pmid {pids_list[0]} due to DB error: {e}")
                return []

    def collect_ids_without_embeddings(self, pids):
        logger.debug('collect_ids_without_embeddings')
        pids_list = list(pids)
        if not pids_list:
            return []
        with self.get_connection(readonly=True) as connection:
            result = self._query_pmids_batch(connection, pids_list)
            pids_with_embeddings = set(result)
            return [pid for pid in pids_list if pid not in pids_with_embeddings]

    def store_embeddings_to_postgresql(self, chunk_embeddings, chunk_idx):
        logger.debug('store_embeddings_to_postgresql')
        if len(chunk_idx) == 0:
            return
        # Normalize embeddings if using cosine similarity
        data = [(pmid, chunk, l2norm(e).tolist())
                for (pmid, chunk), e in zip(chunk_idx, chunk_embeddings)]
        with self.get_connection(readonly=False) as connection:
            with connection.cursor() as cursor:
                execute_values(
                    cursor,
                    f"INSERT INTO {self.embeddings_model_name} (pmid, chunk, embedding) VALUES %s "
                    f"ON CONFLICT (pmid, chunk) DO NOTHING",
                    data
                )
            connection.commit()

    def sample_embeddings(self, n=10_000):
        logger.debug('sample_embeddings')
        with self.get_connection(readonly=True) as connection:
            with connection.cursor() as cursor:
                # Random sample instead of the first physical rows, which are biased towards a single year.
                # Oversample slightly as TABLESAMPLE is approximate, LIMIT then cuts only the physical tail;
                # reltuples is -1/0 if table was never analyzed.
                cursor.execute("SELECT reltuples FROM pg_class WHERE relname=%s", (self.embeddings_model_name,))
                row = cursor.fetchone()
                total = row[0] if row else 0
                percent = min(100.0, 105.0 * n / total) if total > 0 else 100.0
                query = f"""
                    SELECT embedding FROM {self.embeddings_model_name}
                    TABLESAMPLE BERNOULLI ({percent})
                    LIMIT {n};
            """
                logger.debug(f'query: {query[:1000]}')
            # Server-side cursor to stream millions of rows, preallocated array instead of lists of floats
            embeddings = np.empty((n, self.embedding_dimension), dtype=np.float32)
            size = 0
            with connection.cursor(name='sample_embeddings') as cursor:
                cursor.itersize = 10_000
                cursor.execute(query)
                for row in tqdm(cursor, total=n):
                    embeddings[size] = parse_vector(row[0])
                    size += 1
            return embeddings[:size]

    def _query_embeddings_batch(self, connection, pids_list):
        try:
            with connection.cursor() as cursor:
                query = f"""
                        SELECT pmid, chunk, embedding FROM {self.embeddings_model_name}
                        WHERE pmid = ANY (%s)
                        ORDER BY pmid, chunk;
                """
                logger.debug(f'query: {query[:1000]}')
                cursor.execute(query, (pids_list,))
                return cursor.fetchall()
        except Exception as e:
            try:
                connection.rollback()
            except Exception:
                pass
            if len(pids_list) > 1:
                mid = len(pids_list) // 2
                left = self._query_embeddings_batch(connection, pids_list[:mid])
                right = self._query_embeddings_batch(connection, pids_list[mid:])
                return left + right
            else:
                logger.warning(f"Failed to load embedding for pmid {pids_list[0]} due to DB error: {e}")
                return []

    def load_embeddings_by_ids(self, pids):
        logger.debug('load_embeddings_by_ids')
        pids_list = list(pids)
        if not pids_list:
            return [], []
        with self.get_connection(readonly=True) as connection:
            result = self._query_embeddings_batch(connection, pids_list)
            index = []
            embeddings = []
            for pmid, chunk, emb_str in result:
                try:
                    vec = parse_vector(emb_str)
                    index.append((pmid, chunk))
                    embeddings.append(vec)
                except Exception as e:
                    logger.warning(f"Failed to parse embedding vector for pmid {pmid}, chunk {chunk}: {e}")
            return index, embeddings

    def is_embeddings_db_available(self):
        try:
            with self.get_connection(readonly=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("select 1 from information_schema.tables where table_name=%s",
                                   (self.embeddings_model_name,))
                    return cursor.fetchone() is not None
        except Exception:
            return False
