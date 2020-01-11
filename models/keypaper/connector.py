import psycopg2 as pg_driver

from abc import ABCMeta
from neo4j import GraphDatabase


class Connector:

    def __init__(self, pubtrends_config, connect=True):
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

    def close_connection(self):
        if self.conn:
            self.conn.close()
        if self.neo4jdriver:
            self.neo4jdriver.close()
