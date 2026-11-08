import json

from pysrc.papers.db.lucene_searcher import LuceneIndexManager
from pysrc.papers.db.postgres_connector import PostgresConnector


class PubmedPostgresWriter(PostgresConnector):

    def __init__(self, config):
        super(PubmedPostgresWriter, self).__init__(config, readonly=False)
        self.lucene_manager = LuceneIndexManager(config.pm_lucene_index_path, config.bm25_k1, config.bm25_b)

    def init_pubmed_database(self):
        self.check_connection()
        query_citations = '''
                    drop index if exists PMCitations_pmid_out;
                    drop index if exists PMCitations_pmid_in;
                    drop table if exists PMCitations;
                    
                    create table PMCitations (
                        pmid_out integer,
                        pmid_in  integer
                    );
                    
                    create index if not exists PMCitations_pmid_out_index on PMCitations using hash(pmid_out);
                    create index if not exists PMCitations_pmid_in_index on PMCitations using hash(pmid_in);
                    '''

        query_publications = '''
                    drop table if exists PMPublications;
                    
                    create table PMPublications(
                        pmid    integer,
                        title   varchar(1023),
                        year    integer,
                        abstract text,
                        type    varchar(20),
                        keywords varchar(100),
                        mesh varchar(100),
                        doi     varchar(100),
                        aux     jsonb
                    );
                    
                    create index if not exists PMPublications_pmid_index on PMPublications using hash(pmid);
                    create index if not exists PMPublications_doi_index on PMPublications using hash(doi); 
                    
                    CREATE INDEX IF NOT EXISTS pmpublications_pmid_year ON pmpublications (pmid, year);
                    '''

        query_drop_matview = '''
                    drop materialized view if exists matview_pmcitations;
                    drop index if exists PMCitation_matview_index;
                    '''
        query_create_matview = '''
                    create materialized view matview_pmcitations as
                    SELECT pmid, COUNT(*) AS count
                    FROM PMPublications P
                    LEFT JOIN PMCitations C
                    ON C.pmid_in = pmid            
                    GROUP BY pmid
                    HAVING COUNT(*) >= 3; -- Ignore tail of 0,1,2 cited papers
                    create index if not exists PMCitation_matview_index on matview_pmcitations using hash(pmid);
                    '''
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query_drop_matview)
            cursor.execute(query_citations)
            cursor.execute(query_publications)
            cursor.execute(query_create_matview)
            self.postgres_connection.commit()
        self.lucene_manager.reset()

    def insert_pubmed_publications(self, articles):
        if not articles:
            return
        articles_vals = ', '.join(
            str((a.pmid, a.title, a.year, a.abstract or '',
                 a.type, ','.join(a.keywords), ','.join(a.mesh),
                 a.doi or '', json.dumps(a.aux.to_dict())))
            for a in articles
        )

        query = f'''
            insert into PMPublications(pmid, title, year, abstract, type, keywords, mesh, doi, aux)
                values {articles_vals};
            '''
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            self.postgres_connection.commit()
        self.lucene_manager.insert_pubmed_articles(articles)

    def insert_pubmed_citations(self, citations):
        citations_vals = ', '.join(f"({c[0]}, {c[1]})" for c in citations)

        query = f'insert into PMCitations (pmid_out, pmid_in) values {citations_vals};'
        query_update_matview = 'refresh materialized view matview_pmcitations;'

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            cursor.execute(query_update_matview)
            self.postgres_connection.commit()

    def delete(self, ids):
        if not ids:
            return
        ids_vals = ','.join(f'({i})' for i in ids)
        query = f'''
                DELETE FROM PMCitations
                WHERE pmid_in IN (VALUES {ids_vals}) OR pmid_out IN (VALUES {ids_vals});
                DELETE FROM PMPublications
                WHERE pmid IN (VALUES {ids_vals});
                '''
        query_update_matview = 'refresh materialized view matview_pmcitations;'
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            cursor.execute(query_update_matview)
            self.postgres_connection.commit()
        self.lucene_manager.delete(ids)
