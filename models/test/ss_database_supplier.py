from models.keypaper.config import PubtrendsConfig
from models.keypaper.connector import Connector


class SSTestDatabaseSupplier(Connector):

    def __init__(self):
        config = PubtrendsConfig(test=True)
        super(SSTestDatabaseSupplier, self).__init__(config)
        self.project_dir = __file__.replace('/models/test/ss_database_supplier.py', '')

    def init_semantic_scholar_database(self):
        with self.neo4jdriver.session() as session:
            indexes = session.run('CALL db.indexes()').data()
            if len(list(filter(lambda i: i['description'] == 'INDEX ON :SSPublication(crc32id)', indexes))) > 0:
                session.run('DROP INDEX ON :SSPublication(crc32id)')

            if len(list(filter(lambda i: i['description'] == 'INDEX ON NODE:SSPublication(title, abstract)',
                               indexes))) > 0:
                session.run('CALL db.index.fulltext.drop("ssTitlesAndAbstracts")')

        with self.neo4jdriver.session() as session:
            session.run('MATCH ()-[r:SSReferenced]->() DELETE r;')
            session.run('MATCH (p:SSPublication) DELETE p;')

    def insert_semantic_scholar_publications(self, articles):
        query = '''
UNWIND {articles} AS data
MERGE (n:SSPublication { crc32id: toInteger(data.crc32id), ssid: data.id })
ON CREATE SET
    n.pmid = data.pmid,
    n.title = data.title,
    n.abstract = data.abstract,
    n.date = datetime({year: data.year, month: 1, day: 1}),
    n.aux = data.aux
ON MATCH SET
    n.pmid = data.pmid,
    n.title = data.title,
    n.abstract = data.abstract,
    n.date = datetime({year: data.year, month: 1, day: 1}),
    n.aux = data.aux
RETURN n;
'''
        with self.neo4jdriver.session() as session:
            session.run(query, articles=[a.to_dict() for a in articles])

        # Init index by crc32id
        with self.neo4jdriver.session() as session:
            session.run('CREATE INDEX ON :SSPublication(crc32id)')

        # Init full text search index
        with self.neo4jdriver.session() as session:
            session.run('''
CALL db.index.fulltext.createNodeIndex("ssTitlesAndAbstracts", ["SSPublication"], ["title", "abstract"])
''')

    def insert_semantic_scholar_citations(self, citations):
        query = '''
UNWIND {citations} AS cit
MATCH (n_out:SSPublication { crc32id: toInteger(cit.crc32id_out), ssid: cit.ssid_out })
MERGE (n_in:SSPublication { crc32id: toInteger(cit.crc32id_in), ssid: cit.ssid_in })
MERGE (n_out)-[:SSReferenced]->(n_in);
'''
        with self.neo4jdriver.session() as session:
            session.run(query, citations=[{
                'crc32id_out': c[0].crc32id,
                'crc32id_in': c[1].crc32id,
                "ssid_out": c[0].ssid,
                "ssid_in": c[1].ssid} for c in citations])