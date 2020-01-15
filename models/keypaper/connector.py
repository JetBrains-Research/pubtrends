from neo4j import GraphDatabase


class Connector:

    def __init__(self, pubtrends_config, connect=True):
        self.neo4jdriver = None
        if connect:
            self.neo4jdriver = GraphDatabase.driver(
                f'bolt://{pubtrends_config.neo4jhost}:{pubtrends_config.neo4jport}',
                auth=(pubtrends_config.neo4juser, pubtrends_config.neo4jpassword)
            )
            try:
                with self.neo4jdriver.session() as session:
                    session.run('Match () Return 1 Limit 1')
            except Exception as e:
                raise Exception(f'Failed to connect to neo4j database. Check config.')

    def close_connection(self):
        if self.neo4jdriver:
            self.neo4jdriver.close()
