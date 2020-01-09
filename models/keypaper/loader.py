import html
import json
import re

import numpy as np
import psycopg2 as pg_driver

from neo4j import GraphDatabase
from models.keypaper.utils import extract_authors


class Loader:

    def __init__(self, pubtrends_config, connect=True):
        self.pubtrends_config = pubtrends_config
        self.conn = None
        if connect and int(pubtrends_config.port) != 0:
            # TODO[shpynov] Remove this simple check that port is configured after removing postgresql support
            connection_string = f"""
                dbname={pubtrends_config.dbname} user={pubtrends_config.user} password={pubtrends_config.password} \
                host={pubtrends_config.host} port={pubtrends_config.port}
            """.strip()
            self.conn = pg_driver.connect(connection_string)

        self.neo4jdriver = None
        if connect:
            self.neo4jdriver = GraphDatabase.driver(f'bolt://{pubtrends_config.neo4jurl}',
                                                    auth=(pubtrends_config.neo4juser, pubtrends_config.neo4jpassword))

        self.logger = None

        self.max_number_of_articles = pubtrends_config.max_number_of_articles
        self.max_number_of_citations = pubtrends_config.max_number_of_citations
        self.max_number_of_cocitations = pubtrends_config.max_number_of_cocitations

    def close_connection(self):
        if self.conn:
            self.conn.close()
        if self.neo4jdriver:
            self.neo4jdriver.close()

    def set_logger(self, logger):
        self.logger = logger

    @staticmethod
    def process_publications_dataframe(publications_df):
        # Semantic Scholar stores aux in jsonb format, no json parsing required
        publications_df['aux'] = publications_df['aux'].apply(
            lambda aux: json.loads(aux) if type(aux) is str else aux
        )
        publications_df = publications_df.fillna(value={'abstract': ''})
        publications_df['year'] = publications_df['year'].apply(
            lambda year: int(year) if year and np.isfinite(year) else np.nan
        )
        publications_df['authors'] = publications_df['aux'].apply(lambda aux: extract_authors(aux['authors']))
        publications_df['journal'] = publications_df['aux'].apply(lambda aux: html.unescape(aux['journal']['name']))
        publications_df['title'] = publications_df['title'].apply(lambda title: html.unescape(title))

        # Semantic Scholar specific hack
        if 'crc32id' in publications_df:
            publications_df['crc32id'] = publications_df['crc32id'].apply(int)
        return publications_df
