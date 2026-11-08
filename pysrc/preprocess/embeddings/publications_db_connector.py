import logging
import os
from contextlib import contextmanager

import pandas as pd
import psycopg2
from psycopg2.pool import ThreadedConnectionPool

from pysrc.config import PubtrendsConfig

config = PubtrendsConfig(test=False)

logger = logging.getLogger(__name__)


class PublicationsDBConnector:
    def __init__(self):
        self.connection_string = f"""
                    host={config.postgres_host} \
                    port={config.postgres_port} \
                    dbname={config.postgres_database} \
                    user={config.postgres_username} \
                    password={config.postgres_password}
                """.strip()
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
    def get_connection(self, readonly=True):
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

    def _query_publications_batch(self, connection, pids_list):
        try:
            with connection.cursor() as cursor:
                query = '''
                        SELECT P.pmid as id, title, abstract, type, year, keywords
                        FROM PMPublications P
                        WHERE P.pmid = ANY (%s);
                        '''
                cursor.execute(query, (pids_list,))
                return cursor.fetchall()
        except Exception as e:
            try:
                connection.rollback()
            except Exception:
                pass
            if len(pids_list) > 1:
                mid = len(pids_list) // 2
                left = self._query_publications_batch(connection, pids_list[:mid])
                right = self._query_publications_batch(connection, pids_list[mid:])
                return left + right
            else:
                logger.warning(f"Failed to load publication for pmid {pids_list[0]} due to DB error: {e}")
                return []

    def load_publications(self, pids):
        pids_list = list(pids)
        if not pids_list:
            return pd.DataFrame(columns=['pmid', 'title', 'abstract', 'type', 'year', 'keywords'], dtype=object)
        with self.get_connection(readonly=True) as connection:
            rows = self._query_publications_batch(connection, pids_list)
            df = pd.DataFrame(rows,
                              columns=['pmid', 'title', 'abstract', 'type', 'year', 'keywords'],
                              dtype=object)
            return df

    def load_publications_year(self, year):
        query = f'''
                SELECT P.pmid as id, title, abstract
                FROM PMPublications P
                WHERE year = {year}
                ORDER BY pmid;
                '''
        with self.get_connection(readonly=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                df = pd.DataFrame(cursor.fetchall(),
                                  columns=['pmid', 'title', 'abstract'],
                                  dtype=object)
                return df

    def collect_pids_types_year(self, year):
        query = f'''SELECT pmid
                FROM PMPublications P
                WHERE year = {year}
                ORDER BY pmid;
                '''
        with self.get_connection(readonly=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                return [v[0] for v in cursor.fetchall()]

    def load_min_year(self):
        with self.get_connection(readonly=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute('SELECT MIN(year) FROM PMPublications;')
                return cursor.fetchone()[0]
