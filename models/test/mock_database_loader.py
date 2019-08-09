import re

from models.keypaper.config import PubtrendsConfig
from models.keypaper.loader import Loader


class MockDatabaseLoader(Loader):

    def __init__(self, db):
        config = PubtrendsConfig(test=True)
        super(MockDatabaseLoader, self).__init__(config)

    def init_pubmed_database(self):
        query_citations = '''
            DROP TABLE IF EXISTS PMCitations;
            CREATE TABLE PMCitations (
                pmid_out    INTEGER,
                pmid_in     INTEGER
            );
            '''

        query_publications = '''
            DROP TABLE IF EXISTS PMPublications;
            CREATE TABLE PMPublications (
                pmid        INTEGER PRIMARY KEY,
                date        DATE NULL,
                title       VARCHAR(1023),
                abstract    TEXT NULL,
                aux         JSONB
            );
            '''

        with self.conn:
            self.cursor.execute(query_citations)
            self.cursor.execute(query_publications)

    def insert_pubmed_publications(self, articles):
        articles_str = ', '.join(list(map(str, articles)))

        query = re.sub(self.VALUES_REGEX, articles_str, '''
                INSERT INTO PMPublications(pmid, title, aux, abstract, date) VALUES $VALUES$;
            ''')
        with self.conn:
            self.cursor.execute(query)

    def insert_pubmed_citations(self, citations):
        citations_formatted = [f'({id_out}, {id_in})' for id_out, id_in in citations]

        query = re.sub(self.VALUES_REGEX, ', '.join(citations_formatted), '''
                INSERT INTO PMCitations(pmid_out, pmid_in) VALUES $VALUES$;
            ''')

        with self.conn:
            self.cursor.execute(query)

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
                        ssid        varchar(40) not null,
                        crc32id     integer     not null,
                        title       varchar(1023),
                        abstract    text        null,
                        year        integer     null,
                        aux         jsonb
                    );
                    create index if not exists sspublications_crc32id_index
                    on sspublications (crc32id);
                    '''

        with self.conn:
            self.cursor.execute(query_citations)
            self.cursor.execute(query_publications)

    def insert_semantic_scholar_publications(self, articles):
        articles_str = ', '.join(
            map(lambda article: article.to_db_publication(), articles))

        query = re.sub(self.VALUES_REGEX, articles_str, '''
            insert into sspublications(ssid, crc32id, title, year) values $VALUES$;
            alter table sspublications add column tsv TSVECTOR;
            create index sspublications_tsv on sspublications using gin(tsv);
            update sspublications set tsv = to_tsvector(COALESCE(title, ''));
            ''')
        with self.conn:
            self.cursor.execute(query)

    def insert_semantic_scholar_citations(self, citations):
        citations_str = ', '.join(
            "('{0}', {1}, '{2}', {3})".format(citation[0].ssid, citation[0].crc32id,
                                              citation[1].ssid, citation[1].crc32id) for citation in citations)

        query = re.sub(self.VALUES_REGEX, citations_str, '''
            insert into sscitations (id_out, crc32id_out, id_in, crc32id_in) values $VALUES$;
            ''')

        with self.conn:
            self.cursor.execute(query)
