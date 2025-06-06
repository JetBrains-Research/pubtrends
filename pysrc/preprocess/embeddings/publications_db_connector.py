import logging
import os

import pandas as pd
import psycopg2

from pysrc.config import PubtrendsConfig
from pysrc.papers.db.postgres_utils import ints_to_vals

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
        self.cache_dir = os.path.expanduser(f'~/pubtrends_year')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def load_publications(self, pids):
        with psycopg2.connect(self.connection_string) as connection:
            connection.set_session(readonly=True)
        vals = ints_to_vals(pids)
        query = f'''
                    SELECT P.pmid as id, title, abstract, type
                    FROM PMPublications P
                    WHERE P.pmid IN (VALUES {vals});
                    '''
        with connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(),
                              columns=['id', 'title', 'abstract', 'type'],
                              dtype=object)
            return df

    def load_publications_year(self, year):
        with psycopg2.connect(self.connection_string) as connection:
            connection.set_session(readonly=True)
            query = f'''
                    SELECT P.pmid as id, title, abstract
                    FROM PMPublications P
                    WHERE year = {year}
                    ORDER BY pmid;
                    '''
            with connection.cursor() as cursor:
                cursor.execute(query)
                df = pd.DataFrame(cursor.fetchall(),
                                  columns=['id', 'title', 'abstract'],
                                  dtype=object)
                return df

    def collect_pids_types_year(self, year):
        with psycopg2.connect(self.connection_string) as connection:
            connection.set_session(readonly=True)
            query = f'''SELECT pmid
                    FROM PMPublications P
                    WHERE year = {year}
                    ORDER BY pmid;
                    '''
        with connection.cursor() as cursor:
            cursor.execute(query)
            return [v[0] for v in cursor.fetchall()]

    def fetch_year(self, year):
        path = os.path.expanduser(f'{self.cache_dir}/{year}.csv.gz')
        if os.path.exists(path):
            try:
                return pd.read_csv(path, compression='gzip')
            except Exception as e:
                logger.error(f'Error reading {path}: {e}')
                os.path.remove(path)
                return self.fetch_year(year)
        else:
            df = self.load_publications_year(year)
            df.to_csv(path, index=None, compression='gzip')
            return df
