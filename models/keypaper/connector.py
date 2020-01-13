from neo4j import GraphDatabase


class Connector:

    def __init__(self, pubtrends_config, connect=True):
        self.neo4jdriver = None
        if connect:
            self.neo4jdriver = GraphDatabase.driver(f'bolt://{pubtrends_config.host}:{pubtrends_config.port}',
                                                    auth=(pubtrends_config.user, pubtrends_config.password))

    def close_connection(self):
        if self.neo4jdriver:
            self.neo4jdriver.close()
