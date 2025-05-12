import json
import logging
import numpy as np
import requests

from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.papers.analysis.text import stemmed_tokens

logger = logging.getLogger(__name__)


class PubmedPostgresWriter(PostgresConnector):

    def __init__(self, config):
        super(PubmedPostgresWriter, self).__init__(config, readonly=False)
        # Launch with Docker address or locally
        self.FASTTEXT_URL = config.get('FASTTEXT_URL', 'http://localhost:5001')

    def compute_embeddings(self, text):
        """
        Compute embeddings for a given text using FastText API

        Args:
            text (str): Text to compute embeddings for

        Returns:
            numpy.ndarray: Embeddings for the text
        """
        if not text or text.strip() == '':
            return None

        # Tokenize text
        tokens = [token for token, _ in stemmed_tokens(text)]
        if not tokens:
            return None

        try:
            # Get embeddings from FastText API
            r = requests.request(
                url=f'{self.FASTTEXT_URL}/fasttext',
                method='POST',
                json=tokens,
                headers={'Accept': 'application/json'}
            )

            if r.status_code == 200:
                # Get embeddings from response
                embeddings = np.array(r.json())

                # Average embeddings for all tokens to get document embedding
                return np.mean(embeddings.reshape(len(tokens), -1), axis=0).tolist()
            else:
                logger.error(f'Failed to get embeddings from FastText API: {r.status_code}')
                return None
        except Exception as e:
            logger.error(f'Error computing embeddings: {e}')
            return None

    def update_embeddings(self, pmid, title, abstract):
        """
        Update embeddings for a publication

        Args:
            pmid (int): Publication ID
            title (str): Publication title
            abstract (str): Publication abstract
        """
        # Combine title and abstract for embedding
        text = f"{title} {abstract}" if abstract else title

        # Compute embeddings
        embeddings = self.compute_embeddings(text)

        if embeddings:
            # Update embeddings in database
            query = f"""
                UPDATE PMPublications
                SET embeddings = ARRAY{embeddings}::float8[]
                WHERE pmid = {pmid};
            """

            with self.postgres_connection.cursor() as cursor:
                cursor.execute(query)
                self.postgres_connection.commit()

    def init_pubmed_database(self):
        self.check_connection()

        # Install pg_vector extension for vector operations if not already installed
        query_extension = '''
                    CREATE EXTENSION IF NOT EXISTS vector;
                    '''

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

                    ALTER TABLE PMPublications ADD COLUMN IF NOT EXISTS tsv TSVECTOR;
                    create index if not exists PMPublications_tsv on PMPublications using gin(tsv);

                    -- Add column for embeddings (vector of 200 dimensions)
                    ALTER TABLE PMPublications ADD COLUMN IF NOT EXISTS embeddings FLOAT8[] DEFAULT NULL;

                    -- Create index on embeddings column for faster similarity search
                    CREATE INDEX IF NOT EXISTS pmpublications_embeddings_idx ON pmpublications USING ivfflat (embeddings vector_cosine_ops);

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
            # Install vector extension first
            cursor.execute(query_extension)
            cursor.execute(query_drop_matview)
            cursor.execute(query_citations)
            cursor.execute(query_publications)
            cursor.execute(query_create_matview)
            self.postgres_connection.commit()

    def insert_pubmed_publications(self, articles):
        ids_vals = ','.join(f'({a.pmid})' for a in articles)
        articles_vals = ', '.join(
            str((a.pmid, a.title, a.year, a.abstract or '',
                 a.type, ','.join(a.keywords), ','.join(a.mesh),
                 a.doi or '', json.dumps(a.aux.to_dict())))
            for a in articles
        )

        query = f'''
            insert into PMPublications(pmid, title, year, abstract, type, keywords, mesh, doi, aux)
                values {articles_vals};
            update PMPublications
                set tsv = setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
                WHERE pmid IN (VALUES {ids_vals});
            '''
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            self.postgres_connection.commit()

        # Compute and store embeddings for each article
        logger.info(f'Computing embeddings for {len(articles)} articles')
        for article in articles:
            self.update_embeddings(article.pmid, article.title, article.abstract or '')

    def insert_pubmed_citations(self, citations):
        citations_vals = ', '.join(f"({c[0]}, {c[1]})" for c in citations)

        query = f'insert into PMCitations (pmid_out, pmid_in) values {citations_vals};'
        query_update_matview = 'refresh materialized view matview_pmcitations;'

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            cursor.execute(query_update_matview)
            self.postgres_connection.commit()

    def delete(self, ids):
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
