import json

from pysrc.papers.db.lucene_searcher import LuceneIndexManager
from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.papers.db.ss_postgres_loader import SemanticScholarPostgresLoader


class SemanticScholarPostgresWriter(PostgresConnector):

    def __init__(self, config):
        super(SemanticScholarPostgresWriter, self).__init__(config, readonly=False)
        self.lucene_manager = LuceneIndexManager(config.ss_lucene_index_path, config.bm25_k1, config.bm25_b)

    def init_semantic_scholar_database(self):
        self.check_connection()
        query_citations = '''
                    drop index if exists sscitations_crc32id_out_index;
                    drop index if exists sscitations_crc32id_in_index;
                    drop table if exists sscitations;
                    create table SSCitations (
                        crc32id_out integer,
                        crc32id_in  integer,
                        ssid_out    varchar(40) not null,
                        ssid_in     varchar(40) not null
                    );
                    create index if not exists sscitations_crc32id_out_index on SSCitations using hash(crc32id_out);
                    create index if not exists sscitations_crc32id_in_index on SSCitations using hash(crc32id_in);
                    '''

        query_publications = '''
                    drop index if exists sspublications_crc32id_index;
                    drop index if exists sspublications_doi_index;
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
                    ALTER TABLE SSPublications ADD CONSTRAINT crc32id_ssid_key PRIMARY KEY (crc32id, ssid);
                    create index if not exists sspublications_doi_index on sspublications using hash(doi);
                    CREATE INDEX IF NOT EXISTS sspublications_crc32id_year ON sspublications (crc32id, year);
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
                    GROUP BY ssid, crc32id
                    HAVING COUNT(*) >= 3; -- Ignore tail of 0,1,2 cited papers
                    create index if not exists SSCitation_matview_index on matview_sscitations using hash(crc32id);
                    '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query_drop_matview)
            cursor.execute(query_citations)
            cursor.execute(query_publications)
            cursor.execute(query_create_matview)
            self.postgres_connection.commit()
        self.lucene_manager.reset()

    def insert_semantic_scholar_publications(self, articles):
        if not articles:
            return
        # For some reason SemanticScholarArticle doesn't have abstract field
        articles_vals = ', '.join(
            str((a.ssid, a.crc32id, a.title, a.year, '',
                 a.doi or '', json.dumps({"journal": {"name": ""}, "authors": []})))
            for a in articles
        )
        query = f'''
            insert into sspublications(ssid, crc32id, title, year, abstract, doi, aux) values {articles_vals};
            '''
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            self.postgres_connection.commit()
        self.lucene_manager.insert_semantic_scholar_articles(articles)

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
