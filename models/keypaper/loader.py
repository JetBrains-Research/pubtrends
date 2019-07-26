import psycopg2 as pg_driver
import re


class Loader:
    VALUES_REGEX = re.compile(r'\$VALUES\$')

    def __init__(self, pubtrends_config):
        connection_string = f"""
dbname={pubtrends_config.dbname} user={pubtrends_config.user} password={pubtrends_config.password} host={pubtrends_config.host} port={pubtrends_config.port}
        """
        self.conn = pg_driver.connect(connection_string)
        self.cursor = self.conn.cursor()

        # Logger
        self.logger = None

        # Data containers
        self.ids = None
        self.pub_df = None
        self.cit_df = None
        self.cocit_df = None
        self.cocit_grouped_df = None

    def close_connection(self):
        self.cursor.close()
        self.conn.close()

    def set_logger(self, logger):
        self.logger = logger
