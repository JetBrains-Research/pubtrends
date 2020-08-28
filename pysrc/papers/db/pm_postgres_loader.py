import logging

import numpy as np
import pandas as pd

from pysrc.papers.db.loader import Loader
from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.papers.db.postgres_utils import preprocess_search_query_for_postgres, \
    process_bibliographic_coupling_postgres, process_cocitations_postgres
from pysrc.papers.utils import SORT_MOST_RELEVANT, SORT_MOST_CITED, SORT_MOST_RECENT, preprocess_doi, \
    preprocess_search_title

logger = logging.getLogger(__name__)


class PubmedPostgresLoader(PostgresConnector, Loader):
    def __init__(self, config):
        super(PubmedPostgresLoader, self).__init__(config)

    @staticmethod
    def ids_to_vals(ids):
        return ','.join([f'({i})' for i in ids])

    def find(self, key, value):
        value = value.strip()

        if key == 'id':
            key = 'pmid'

            # We use integer PMIDs in neo4j, if value is not a valid integer -> no match
            try:
                value = int(value)
            except ValueError:
                raise Exception("PMID should be an integer")

        # Preprocess DOI
        if key == 'doi':
            value = preprocess_doi(value)

        # Use dedicated text index to search title.
        if key == 'title':
            value = preprocess_search_title(value)
            query = f'''
                SELECT pmid
                FROM to_tsquery('english', \'''{value}\''') query, PMPublications P
                WHERE tsv @@ query AND LOWER(title) = LOWER('{value}');
            '''
        else:
            query = f'''
                SELECT pmid
                FROM PMPublications P
                WHERE {key} = {repr(value)};
            '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)

        return list(df['pmid'].astype(str))

    def search(self, query, limit=None, sort=None):
        query_str = preprocess_search_query_for_postgres(query, self.config.min_search_words)
        if sort == SORT_MOST_RELEVANT:
            query = f'''
                SELECT pmid
                FROM to_tsquery('{query_str}') query, PMPublications P
                WHERE tsv @@ query
                ORDER BY ts_rank_cd(P.tsv, query) DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_CITED:
            query = f'''
                WITH X as (SELECT P.pmid as pmid
                            FROM PMPublications P
                            WHERE tsv @@ to_tsquery('{query_str}'))
                SELECT X.pmid as pmid FROM X
                    LEFT JOIN matview_pmcitations C
                    ON X.pmid = C.pmid
                ORDER BY count DESC NULLS LAST
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_RECENT:
            query = f'''
                SELECT pmid
                FROM to_tsquery('{query_str}') query, PMPublications P
                WHERE tsv @@ query
                ORDER BY date DESC NULLS LAST
                LIMIT {limit};
                '''
        else:
            raise ValueError(f'Illegal sort method: {sort}')

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)

        return list(df['pmid'].astype(str))

    def load_publications(self, ids):
        vals = self.ids_to_vals(ids)
        query = f'''
                SELECT P.pmid as id, title, abstract, date_part('year', date) as year, type,
                    keywords, mesh, doi, aux
                FROM PMPublications P
                WHERE P.pmid IN (VALUES {vals});
                '''
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(),
                              columns=['id', 'title', 'abstract', 'year', 'type', 'keywords', 'mesh', 'doi', 'aux'],
                              dtype=object)
        if np.any(df[['id', 'title']].isna()):
            raise ValueError('Paper must have ID and title')
        df['id'] = df['id'].apply(str)
        logger.debug(f'Loaded {len(df)} papers')
        return Loader.process_publications_dataframe(df)

    def load_citation_stats(self, ids):
        vals = self.ids_to_vals(ids)
        query = f'''
           SELECT C.pmid_in AS id, date_part('year', P_out.date) as year, COUNT(1) AS count
                FROM PMCitations C
                JOIN PMPublications P_out
                  ON C.pmid_out = P_out.pmid
                JOIN PMPublications P_in
                  ON C.pmid_in = P_in.pmid
                WHERE P_out.date >= P_in.date AND C.pmid_in IN (VALUES {vals})
                GROUP BY C.pmid_in, year
                LIMIT {self.config.max_number_of_citations};
            '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(),
                              columns=['id', 'year', 'count'], dtype=object)

        if np.any(df.isna()):
            raise ValueError('NaN values are not allowed in citation stats DataFrame')

        df['id'] = df['id'].apply(str)
        df['year'] = df['year'].apply(int)
        df['count'] = df['count'].apply(int)


        return df

    def load_citations(self, ids):
        vals = self.ids_to_vals(ids)
        query = f'''SELECT pmid_out as id_out, pmid_in as id_in
                    FROM PMCitations C
                    WHERE pmid_in IN (VALUES {vals}) AND pmid_out IN (VALUES {vals})
                    ORDER BY id_out, id_in
                    LIMIT {self.config.max_number_of_citations};
                    '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(),
                              columns=['id_out', 'id_in'], dtype=object)

        if np.any(df.isna()):
            raise ValueError('Citation must have id_out and id_in')
        df['id_in'] = df['id_in'].apply(str)
        df['id_out'] = df['id_out'].apply(str)

        return df

    def load_cocitations(self, ids):
        vals = self.ids_to_vals(ids)
        query = f'''with X AS (SELECT pmid_out, pmid_in
                        FROM PMCitations
                        WHERE pmid_in IN (VALUES {vals})),

                        Y AS (SELECT pmid_out, ARRAY_AGG(pmid_in) as cited_list
                        FROM X
                        GROUP BY pmid_out
                        HAVING COUNT(*) >= 2)

                        SELECT Y.pmid_out, date_part('year', P.date) as year, Y.cited_list
                        FROM Y
                        JOIN PMPublications P
                        ON pmid_out = P.pmid and pmid_out = P.pmid
                        LIMIT {self.config.max_number_of_cocitations};
                    '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df, lines = process_cocitations_postgres(cursor)

        if np.any(df[['citing', 'cited_1', 'cited_2']].isna()):
            raise ValueError('NaN values are not allowed in ids of co-citation DataFrame')

        logger.debug(f'Loaded {lines} lines of citing info')

        return df

    def expand(self, ids, limit):
        vals = self.ids_to_vals(ids)
        # List of ids sorted by citations
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH X AS (
                SELECT C.pmid_in AS pmid
                FROM PMCitations C
                WHERE C.pmid_out IN (VALUES {vals})
                UNION
                SELECT C.pmid_out AS pmid
                FROM PMCitations C
                WHERE C.pmid_in IN (VALUES {vals}))
            SELECT X.pmid as pmid FROM X
                    LEFT JOIN matview_pmcitations C
                    ON X.pmid = C.pmid
                ORDER BY count DESC NULLS LAST
                LIMIT {limit};
                '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)

        return list(df['pmid'].astype(str))

    def load_bibliographic_coupling(self, ids):
        vals = self.ids_to_vals(ids)
        query = f'''WITH X AS (SELECT pmid_out, pmid_in
                        FROM PMCitations
                        WHERE pmid_out IN (VALUES {vals}))

                        SELECT pmid_in, ARRAY_AGG(pmid_out) as citing_list
                        FROM X
                        GROUP BY pmid_in
                        HAVING COUNT(*) >= 2
                        LIMIT {self.config.max_number_of_bibliographic_coupling};
                    '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df, lines = process_bibliographic_coupling_postgres(cursor)

        logger.debug(f'Loaded {lines} lines of bibliographic coupling info')

        return df
