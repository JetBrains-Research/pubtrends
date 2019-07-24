import psycopg2 as pg_driver


class Loader:
    def __init__(self, pubtrends_config):
        connection_string = f"""
dbname={pubtrends_config.dbname} user={pubtrends_config.user} password={pubtrends_config.password} host={pubtrends_config.host} port={pubtrends_config.port}
        """
        self.conn = pg_driver.connect(connection_string)
        self.cursor = self.conn.cursor()

    def set_logger(self, logger):
        self.logger = logger
