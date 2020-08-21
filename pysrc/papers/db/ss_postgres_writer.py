import json

from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.postgres_connector import PostgresConnector


class SemanticScholarPostgresWriter(PostgresConnector):

    def __init__(self):
        config = PubtrendsConfig(test=True)
        super(SemanticScholarPostgresWriter, self).__init__(config)

    def init_semantic_scholar_database(self):
        query_citations = '''
                    drop table if exists sscitations;
                    create table sscitations (
                        crc32id_out integer,
                        crc32id_in  integer,
                        ssid_out      varchar(40) not null,
                        ssid_in       varchar(40) not null
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
                        doi     varchar(100),
                        aux     jsonb
                    );
                    create index if not exists sspublications_crc32id_index on sspublications (crc32id);
                    
                    ALTER TABLE SSPublications ADD COLUMN IF NOT EXISTS tsv TSVECTOR;
                    create index if not exists SSPublications_tsv on SSPublications using gin(tsv);
                    '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query_citations)
            cursor.execute(query_publications)
            self.postgres_connection.commit()

    def insert_semantic_scholar_publications(self, articles):
        ids_vals = ','.join(f"('{a.ssid}')" for a in articles)
        # For some reason SemanticScholarArticle doesn't have abstract field
        articles_vals = ', '.join(
            str((a.ssid, a.crc32id, a.title, a.year, '', a.type,
                 '', json.dumps({"journal": {"name": ""}, "authors": []})))
            for a in articles
        )
        query = f'''
            insert into sspublications(ssid, crc32id, title, year, abstract, type, doi, aux) values {articles_vals};
            update sspublications set tsv = to_tsvector(COALESCE(title, '') || COALESCE(abstract, ''))
            WHERE ssid IN (VALUES {ids_vals});
            '''
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            self.postgres_connection.commit()

    def insert_semantic_scholar_citations(self, citations):
        citations_vals = ', '.join(
            f"('{c[0].ssid}', {c[0].crc32id}, '{c[1].ssid}', {c[1].crc32id})" for c in citations
        )

        query = f'insert into sscitations (ssid_out, crc32id_out, ssid_in, crc32id_in) values {citations_vals};'

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            self.postgres_connection.commit()
