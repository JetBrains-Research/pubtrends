import json

from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.postgres_connector import PostgresConnector


class PubmedPostgresWriter(PostgresConnector):

    def __init__(self):
        config = PubtrendsConfig(test=True)
        super(PubmedPostgresWriter, self).__init__(config)

    def init_pubmed_database(self):
        query_citations = '''
                    drop table if exists PMCitations;
                    create table PMCitations (
                        pmid_out integer,
                        pmid_in  integer
                    );
                    create index if not exists PMCitations_pmid_out_pmid_in_index
                    on PMCitations (pmid_out, pmid_in);
                    '''

        query_publications = '''
                    drop table if exists PMPublications;
                    create table PMPublications(
                        pmid    integer,
                        title   varchar(1023),
                        date    date,
                        abstract text,
                        type    varchar(20),
                        doi     varchar(100),
                        aux     jsonb
                    );
                    create index if not exists PMPublications_pmid_index on PMPublications (pmid);
                    ALTER TABLE PMPublications ADD COLUMN IF NOT EXISTS tsv TSVECTOR;
                    create index if not exists PMPublications_tsv on PMPublications using gin(tsv);
                    '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query_citations)
            cursor.execute(query_publications)
            self.postgres_connection.commit()

    def insert_pubmed_publications(self, articles):
        ids_vals = ','.join(f'({a.pmid})' for a in articles)
        articles_vals = ', '.join(
            str((a.pmid, a.title, a.date.strftime('%Y-%m-%d'), a.abstract or '',
                 a.type, a.doi, json.dumps(a.aux.to_dict())))
            for a in articles
        )

        query = f'''
            insert into PMPublications(pmid, title, date, abstract, type, doi, aux) values {articles_vals};
            update PMPublications
                set tsv = setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
                WHERE pmid IN (VALUES {ids_vals});
            '''
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            self.postgres_connection.commit()

    def insert_pubmed_citations(self, citations):
        citations_vals = ', '.join(f"({c[0]}, {c[1]})" for c in citations)

        query = f'insert into PMCitations (pmid_out, pmid_in) values {citations_vals};'

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            self.postgres_connection.commit()
