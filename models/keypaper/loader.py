import re

import psycopg2 as pg_driver


class Loader:
    VALUES_REGEX = re.compile(r'\$VALUES\$')

    def __init__(self, pubtrends_config, connect=True):
        self.conn = None
        self.cursor = None

        if connect:
            connection_string = f"""
                dbname={pubtrends_config.dbname} user={pubtrends_config.user} password={pubtrends_config.password} \
                host={pubtrends_config.host} port={pubtrends_config.port}
            """.strip()
            self.conn = pg_driver.connect(connection_string)
            self.cursor = self.conn.cursor()

        self.max_number_of_articles = pubtrends_config.max_number_of_articles
        self.max_number_of_citations = pubtrends_config.max_number_of_citations
        self.max_number_of_cocitations = pubtrends_config.max_number_of_cocitations

        # Logger
        self.logger = None

        # Data containers
        self.ids = None
        self.pub_df = None
        self.cit_df = None
        self.cocit_df = None
        self.cocit_grouped_df = None

    def close_connection(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def set_logger(self, logger):
        self.logger = logger
