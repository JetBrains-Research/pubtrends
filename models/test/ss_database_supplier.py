from models.keypaper.config import PubtrendsConfig
from models.keypaper.connector import Connector


class SSTestDatabaseSupplier(Connector):

    def __init__(self):
        config = PubtrendsConfig(test=True)
        super(SSTestDatabaseSupplier, self).__init__(config)
        self.project_dir = __file__.replace('/models/test/ss_test_database_loader.py', '')

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
                        type    varchar(20),
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
            insert into sspublications(ssid, crc32id, title, year, abstract, type, aux) values {articles_str};
            alter table sspublications add column tsv TSVECTOR;
            create index sspublications_tsv on sspublications using gin(tsv);
            update sspublications set tsv = to_tsvector(COALESCE(title, ''));
            '''
        with self.conn.cursor() as cursor:
            cursor.execute(query)
            self.conn.commit()

    def insert_semantic_scholar_citations(self, citations):
        citations_str = ', '.join(
            f"('{citation[0].ssid}', {citation[0].crc32id}, '{citation[1].ssid}', {citation[1].crc32id})"
            for citation in citations)

        query = f'insert into sscitations (id_out, crc32id_out, id_in, crc32id_in) values {citations_str};'

        with self.conn.cursor() as cursor:
            cursor.execute(query)
            self.conn.commit()
