import csv
import tempfile
import re

from models.keypaper.config import PubtrendsConfig
from models.keypaper.loader import Loader


class PMTestDatabaseLoader(Loader):

    def __init__(self):
        config = PubtrendsConfig(test=True)
        super(PMTestDatabaseLoader, self).__init__(config)
        self.project_dir = __file__.replace('/models/test/pm_test_database_loader.py', '')

    def init_pubmed_database(self):
        with self.neo4jdriver.session() as session:
            indexes = session.run('CALL db.indexes()').data()
            if len(list(filter(lambda i: i['description'] == 'INDEX ON :PMPublication(pmid)', indexes))) > 0:
                session.run('DROP INDEX ON :PMPublication(pmid)')

            if len(list(filter(lambda i: i['description'] == 'INDEX ON NODE:PMPublication(title, abstract)',
                               indexes))) > 0:
                session.run('CALL db.index.fulltext.drop("pmTitlesAndAbstracts")')

        with self.neo4jdriver.session() as session:
            session.run('MATCH ()-[r:PMReferenced]->() DELETE r;')
            session.run('MATCH (p:PMPublication) DELETE p;')

    def insert_pubmed_publications(self, articles):
        with tempfile.NamedTemporaryFile(dir=self.project_dir, prefix='pubs', suffix='.csv', delete=True) as file:
            with open(file.name, 'w', newline='') as csvfile:
                wr = csv.writer(csvfile)
                for row in [a.to_list() for a in articles]:
                    wr.writerow(row)

            query = f'''
                LOAD CSV FROM "file://{re.sub(self.project_dir, '', file.name)}" AS line
                WITH line, [x IN split(line[4], "-") | toInteger(x)] AS parts
                WITH line, CASE WHEN line[4] = '' THEN NULL
                ELSE datetime({{year: parts[0], month: parts[1], day: parts[2]}})
                END AS date_or_null
                CREATE (:PMPublication {{ pmid: line[0], title: line[1], aux: line[2], abstract: line[3],
                    date: date_or_null,
                    type: line[5],
                    authors: line[6], journal: line[7] }})
                '''
            with self.neo4jdriver.session() as session:
                session.run(query)

            # Init index by pmid
            with self.neo4jdriver.session() as session:
                session.run('CREATE INDEX ON :PMPublication(pmid)')

            # Init full text search index
            with self.neo4jdriver.session() as session:
                session.run('CALL db.index.fulltext.createNodeIndex'
                            '("pmTitlesAndAbstracts",["PMPublication"],["title", "abstract"])')

    def insert_pubmed_citations(self, citations):
        with tempfile.NamedTemporaryFile(dir=self.project_dir, prefix='cits', suffix='.csv', delete=True) as file:
            with open(file.name, 'w', newline='') as csvfile:
                wr = csv.writer(csvfile)
                for row in citations:
                    wr.writerow(row)

            query = f'''
                LOAD CSV FROM "file://{re.sub(self.project_dir, '', file.name)}" AS line
                MATCH (out:PMPublication),(in:PMPublication)
                WHERE out.pmid = line[0] AND in.pmid = line[1]
                CREATE (out)-[r:PMReferenced]->(in)
                '''
            with self.neo4jdriver.session() as session:
                session.run(query)
