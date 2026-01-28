import logging

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

    def load_publications(self, pids):
        with psycopg2.connect(self.connection_string) as connection:
            connection.set_session(readonly=True)
        vals = ints_to_vals(pids)
        query = f'''
                    SELECT P.pmid as id, title, abstract, type, year
                    FROM PMPublications P
                    WHERE P.pmid = ANY ('{{{vals}}}'::integer[]);
                    '''
        with connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(),
                              columns=['pmid', 'title', 'abstract', 'type', 'year'],
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
                                  columns=['pmid', 'title', 'abstract'],
                                  dtype=object)
                return df

    def collect_pids_year(self, year):
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
