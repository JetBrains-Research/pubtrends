import json

from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.papers.db.ss_postgres_loader import SemanticScholarPostgresLoader


class SemanticScholarPostgresWriter(PostgresConnector):

    def __init__(self, config):
        super(SemanticScholarPostgresWriter, self).__init__(config, readonly=False)

    def init_semantic_scholar_database(self):
        self.check_connection()
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
        query_drop_matview = '''
                    drop materialized view if exists matview_sscitations;
                    drop index if exists SSCitation_matview_index;
                    '''
        query_create_matview = '''
                    create materialized view matview_sscitations as
                    SELECT ssid, crc32id, COUNT(*) AS count
                    FROM SSPublications P
                    LEFT JOIN SSCitations C
                    ON C.ssid_in = ssid AND C.crc32id_in = crc32id
                    GROUP BY ssid, crc32id;
                    create index if not exists SSCitation_matview_index on matview_sscitations (crc32id);
                    '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query_drop_matview)
            cursor.execute(query_citations)
            cursor.execute(query_publications)
            cursor.execute(query_create_matview)
            self.postgres_connection.commit()

    def insert_semantic_scholar_publications(self, articles):
        # For some reason SemanticScholarArticle doesn't have abstract field
        articles_vals = ', '.join(
            str((a.ssid, a.crc32id, a.title, a.year, '',
                 a.doi or '', json.dumps({"journal": {"name": ""}, "authors": []})))
            for a in articles
        )
        query = f'''
            insert into sspublications(ssid, crc32id, title, year, abstract, doi, aux) values {articles_vals};
            update sspublications
            set tsv = setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
            WHERE (crc32id, ssid) in (VALUES {SemanticScholarPostgresLoader.ids2values(a.ssid for a in articles)});
            '''
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            self.postgres_connection.commit()

    def insert_semantic_scholar_citations(self, citations):
        citations_vals = ', '.join(
            f"('{c[0].ssid}', {c[0].crc32id}, '{c[1].ssid}', {c[1].crc32id})" for c in citations
        )

        query = f'insert into sscitations (ssid_out, crc32id_out, ssid_in, crc32id_in) values {citations_vals};'
        query_update_matview = 'refresh materialized view matview_sscitations;'

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            cursor.execute(query_update_matview)
            self.postgres_connection.commit()

    def delete(self, ids):
        raise Exception('delete is not supported')
