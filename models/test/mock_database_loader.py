import csv
import tempfile
import re

from models.keypaper.config import PubtrendsConfig
from models.keypaper.loader import Loader


class MockDatabaseLoader(Loader):

    def __init__(self):
        config = PubtrendsConfig(test=True)
        super(MockDatabaseLoader, self).__init__(config)
        self.project_dir = __file__.replace('/models/test/mock_database_loader.py', '')

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

    def init_semantic_scholar_database(self):
        query_citations = '''
                    drop table if exists sscitations;
                    create table sscitations (
                        crc32id_out integer,
                        crc32id_in  integer,
                        id_out      varchar(40) not null,
                        id_in       varchar(40) not null
                    );
                    create index if not exists sscitations_crc32id_out_crc32id_in_index
                    on sscitations (crc32id_out, crc32id_in);
                    '''

        query_publications = '''
                    drop table if exists sspublications;
                    create table sspublications(
                        ssid    varchar(40) not null,
                        crc32id integer     not null,
                        pmid    integer,
                        title   varchar(1023),
                        year    integer,
                        abstract text,
                        aux     jsonb
                    );
                    create index if not exists sspublications_crc32id_index
                    on sspublications (crc32id);
                    '''

        with self.conn.cursor() as cursor:
            cursor.execute(query_citations)
            cursor.execute(query_publications)
            self.conn.commit()

    def insert_semantic_scholar_publications(self, articles):
        articles_str = ', '.join(
            map(lambda article: article.to_db_publication(), articles))

        query = f'''
            insert into sspublications(ssid, crc32id, title, year, abstract, aux) values {articles_str};
            alter table sspublications add column tsv TSVECTOR;
            create index sspublications_tsv on sspublications using gin(tsv);
            update sspublications set tsv = to_tsvector(COALESCE(title, ''));
            '''
        with self.conn.cursor() as cursor:
            cursor.execute(query)
            self.conn.commit()

    def insert_semantic_scholar_citations(self, citations):
        citations_str = ', '.join(
            "('{0}', {1}, '{2}', {3})".format(citation[0].ssid, citation[0].crc32id,
                                              citation[1].ssid, citation[1].crc32id) for citation in citations)

        query = f'insert into sscitations (id_out, crc32id_out, id_in, crc32id_in) values {citations_str};'

        with self.conn.cursor() as cursor:
            cursor.execute(query)
            self.conn.commit()
