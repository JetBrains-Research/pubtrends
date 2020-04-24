from models.keypaper.config import PubtrendsConfig
from models.keypaper.connector import Connector


class PMTestDatabaseSupplier(Connector):
    INDEX_FIELDS = ['pmid', 'doi']

    def __init__(self):
        config = PubtrendsConfig(test=True)
        super(PMTestDatabaseSupplier, self).__init__(config)

    def init_pubmed_database(self):
        with self.neo4jdriver.session() as session:
            indexes = session.run('CALL db.indexes()').data()
            for field in self.INDEX_FIELDS:
                if len(list(filter(lambda i: i['description'] == f'INDEX ON :PMPublication({field})', indexes))) > 0:
                    session.run(f'DROP INDEX ON :PMPublication({field})')

            if len(list(filter(lambda i: i['description'] == 'INDEX ON NODE:PMPublication(title, abstract)',
                               indexes))) > 0:
                session.run('CALL db.index.fulltext.drop("pmTitlesAndAbstracts")')

        with self.neo4jdriver.session() as session:
            session.run('MATCH ()-[r:PMReferenced]->() DELETE r;')
            session.run('MATCH (p:PMPublication) DELETE p;')

    def insert_pubmed_publications(self, articles):
        query = '''
UNWIND {articles} AS data
MERGE (n:PMPublication { pmid: data.pmid })
ON CREATE SET
    n.date = data.date,
    n.title = data.title,
    n.abstract = data.abstract,
    n.type = data.type,
    n.doi = data.doi,
    n.aux = data.aux
ON MATCH SET
    n.date = data.date,
    n.title = data.title,
    n.abstract = data.abstract,
    n.type = data.type,
    n.doi = data.doi,
    n.aux = data.aux
RETURN n;
'''

        with self.neo4jdriver.session() as session:
            session.run(query, articles=[a.to_dict() for a in articles])

        # Init indexes by pmid and doi
        for field in self.INDEX_FIELDS:
            with self.neo4jdriver.session() as session:
                session.run(f'CREATE INDEX ON :PMPublication({field})')

        # Init full text search index
        with self.neo4jdriver.session() as session:
            session.run('CALL db.index.fulltext.createNodeIndex'
                        '("pmTitlesAndAbstracts",["PMPublication"],["title", "abstract"])')

    def insert_pubmed_citations(self, citations):
        query = '''
UNWIND {citations} AS cit
MATCH (n_out:PMPublication { pmid: cit.pmid_out })
MERGE (n_in:PMPublication { pmid: cit.pmid_in })
MERGE (n_out)-[:PMReferenced]->(n_in);
'''
        with self.neo4jdriver.session() as session:
            session.run(query, citations=[{
                'pmid_out': int(c[0]),
                'pmid_in': int(c[1])} for c in citations])
