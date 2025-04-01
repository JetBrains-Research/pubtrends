from datetime import datetime
import logging
import os
import re

import numpy as np
import pandas as pd

from pysrc.papers.db.loader import Loader
from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.papers.db.postgres_utils import preprocess_search_query_for_postgres, no_stemming_filter_for_phrases, \
    process_bibliographic_coupling_postgres, process_cocitations_postgres, preprocess_quotes
from pysrc.papers.utils import SORT_MOST_CITED, SORT_MOST_RECENT, preprocess_doi

logger = logging.getLogger(__name__)


class PubmedPostgresLoader(PostgresConnector, Loader):
    def __init__(self, config):
        super(PubmedPostgresLoader, self).__init__(config)

    @staticmethod
    def ids_to_vals(ids):
        return ','.join([f'({i})' for i in ids])

    UPDATE_LAST_PATH = os.path.expanduser('~/.pubtrends/pubmedpostgreswriter_last.tsv')

    def last_update(self):
        if os.path.exists(self.UPDATE_LAST_PATH):
            return datetime.fromtimestamp(os.path.getmtime(self.UPDATE_LAST_PATH)).strftime('%Y-%m-%d %H:%M:%S')
        return None

    def search_id(self, pid):
        self.check_connection()
        try:
            pid = int(pid)
        except ValueError:
            raise Exception("PMID should be an integer")
        query = f'''
                SELECT pmid
                FROM PMPublications P
                WHERE pmid = {pid};
            '''
        logger.debug(f'find query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)
        return list(df['pmid'].astype(str))


    def search_doi(self, doi):
        self.check_connection()
        doi = preprocess_doi(doi)
        doi = preprocess_quotes(doi)

        query = f'''
                SELECT pmid
                FROM PMPublications P
                WHERE doi = {repr(doi)};
            '''
        logger.debug(f'find query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)
        return list(df['pmid'].astype(str))

    def search_title(self, title):
        self.check_connection()
        title = title.strip()
        title = re.sub('\.$', '', title)
        title = preprocess_quotes(title)
        query = f'''
                SELECT pmid
                FROM to_tsquery(\'''{title}\''') query, PMPublications P
                WHERE tsv @@ query AND TRIM(TRAILING '.' FROM LOWER(title)) = LOWER('{title}');
            '''
        logger.debug(f'find query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)
        return list(df['pmid'].astype(str))

    def search(self, query, limit=None, sort=None, noreviews=True):
        self.check_connection()
        noreviews_filter = "AND type != 'Review'" if noreviews else ''
        query_str = preprocess_search_query_for_postgres(query, self.config.min_search_words)

        # Disable stemming-based lookup for phrases, see: https://github.com/JetBrains-Research/pubtrends/issues/242
        exact_phrase_filter = no_stemming_filter_for_phrases(query_str)

        by_citations = 'count DESC NULLS LAST'
        by_year = 'year DESC NULLS LAST'
        # 2 divides the rank by the document length
        # 4 divides the rank by the mean harmonic distance between extents (this is implemented only by ts_rank_cd)
        # See https://www.postgresql.org/docs/12/textsearch-controls.html#TEXTSEARCH-RANKING
        if sort == SORT_MOST_CITED:
            order = f'{by_citations}, ts_rank_cd(P.tsv, query, 2|4) DESC, {by_year}'
        elif sort == SORT_MOST_RECENT:
            order = f'{by_year}, ts_rank_cd(P.tsv, query, 2|4) DESC, {by_citations}'
        elif sort is None:
            order = 'ts_rank_cd(P.tsv, query, 2|4) DESC'
        else:
            raise ValueError(f'Illegal sort method: {sort}')

        query = f'''
            SELECT P.pmid 
            FROM to_tsquery('{query_str}') query, 
            PMPublications P
            LEFT JOIN matview_pmcitations C 
            ON P.pmid = C.pmid
            WHERE P.tsv @@ query {noreviews_filter} {exact_phrase_filter}
            ORDER BY {order}, P.pmid
            LIMIT {limit};
            '''
        logger.debug(f'search query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)
            # TODO [shpynov] query stays idle in transaction without this commit
            # Further investigation is required
            self.postgres_connection.commit()

        df['pmid'] = df['pmid'].astype(str)
        return list(df['pmid'])

    def load_publications(self, ids):
        self.check_connection()
        vals = self.ids_to_vals(ids)
        query = f'''
                SELECT P.pmid as id, title, abstract, year, type, keywords, mesh, doi, aux
                FROM PMPublications P
                WHERE P.pmid IN (VALUES {vals});
                '''
        logger.debug(f'load_publications query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(),
                              columns=['id', 'title', 'abstract', 'year',
                                       'type', 'keywords', 'mesh', 'doi', 'aux'],
                              dtype=object)
        if np.any(df[['id', 'title']].isna()):
            raise ValueError('Paper must have ID and title')
        logger.debug(f'Loaded {len(df)} papers')
        return Loader.process_publications_dataframe(ids, df)

    def load_citations_by_year(self, ids):
        self.check_connection()
        vals = self.ids_to_vals(ids)
        query = f'''
            WITH X as (SELECT pmid_in, pmid_out
                FROM PMCitations C
                JOIN PMPublications P
                ON C.pmid_in = P.pmid
                WHERE C.pmid_in != C.pmid_out AND P.pmid in (VALUES {vals})) 
            SELECT X.pmid_in AS id, year, COUNT(1) AS count
            FROM X
                JOIN PMPublications P
                ON X.pmid_out = P.pmid
                GROUP BY id, year
                LIMIT {self.config.max_number_of_citations};
            '''
        logger.debug(f'load_citations_by_year query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(),
                              columns=['id', 'year', 'count'],
                              dtype=object)

        if np.any(df.isna()):
            raise ValueError('NaN values are not allowed in citation stats DataFrame')

        df['id'] = df['id'].apply(str)
        df['year'] = df['year'].apply(int)
        df['count'] = df['count'].apply(int)

        return df

    def load_references(self, pid, limit):
        self.check_connection()
        vals = self.ids_to_vals([pid])
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
                SELECT C.pmid_in AS pmid
                FROM PMCitations C 
                JOIN matview_pmcitations MC
                    ON C.pmid_in = MC.pmid
                WHERE C.pmid_in != C.pmid_out AND C.pmid_out IN (VALUES {vals})
                ORDER BY MC.count DESC NULLS LAST
                LIMIT {limit};
                '''
        logger.debug(f'load_references query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['id'], dtype=object)
        return list(df['id'].astype(str))

    def load_citations_counts(self, ids):
        self.check_connection()
        vals = self.ids_to_vals(ids)
        query = f'''
                SELECT count
                FROM PMPublications P
                    LEFT JOIN matview_pmcitations C
                    ON P.pmid = C.pmid
                WHERE P.pmid in (VALUES {vals});
                '''
        logger.debug(f'estimate_citations query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['total'], dtype=object)
            df.fillna(value=1, inplace=True)  # matview_pmcitations ignores < 3 citations

        return list(df['total'])

    def load_citations(self, ids):
        self.check_connection()
        vals = self.ids_to_vals(ids)
        query = f'''SELECT DISTINCT pmid_out as id_out, pmid_in as id_in
                    FROM PMCitations C
                    WHERE pmid_out != pmid_in AND pmid_in IN (VALUES {vals}) AND pmid_out IN (VALUES {vals})
                    ORDER BY id_out, id_in
                    LIMIT {self.config.max_number_of_citations};
                    '''
        logger.debug(f'load_citations query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(),
                              columns=['id_out', 'id_in'],
                              dtype=object)

        if np.any(df.isna()):
            raise ValueError('Citation must have id_out and id_in')
        df['id_in'] = df['id_in'].apply(str)
        df['id_out'] = df['id_out'].apply(str)

        return df

    def load_cocitations(self, ids):
        self.check_connection()
        vals = self.ids_to_vals(ids)
        query = f'''SELECT C.pmid_out as citing, year, ARRAY_AGG(C.pmid_in) as cited_list
                        FROM PMCitations C
                        JOIN PMPublications P
                            ON C.pmid_out = P.pmid
                        WHERE C.pmid_in != C.pmid_out AND C.pmid_in IN (VALUES {vals})
                        GROUP BY citing, year
                        HAVING COUNT(*) >= 2
                        LIMIT {self.config.max_number_of_cocitations};
                    '''

        logger.debug(f'load_cocitations query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = process_cocitations_postgres(cursor)

        if np.any(df[['citing', 'cited_1', 'cited_2']].isna()):
            raise ValueError('NaN values are not allowed in ids of co-citation DataFrame')

        return df

    def expand(self, ids, limit):
        self.check_connection()
        vals = self.ids_to_vals(ids)
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH X AS (
                SELECT C.pmid_in AS pmid
                FROM PMCitations C
                WHERE C.pmid_in != C.pmid_out AND C.pmid_out IN (VALUES {vals}) AND C.pmid_in NOT IN (VALUES {vals})
                UNION
                SELECT C.pmid_out AS pmid
                FROM PMCitations C
                WHERE C.pmid_in != C.pmid_out AND C.pmid_in IN (VALUES {vals}) AND C.pmid_out NOT IN (VALUES {vals})
            )
            SELECT X.pmid as pmid, count 
                FROM X
                    LEFT JOIN matview_pmcitations C
                    ON X.pmid = C.pmid
                ORDER BY count DESC NULLS LAST
                LIMIT {limit};
                '''

        logger.debug(f'expand query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['id', 'total'], dtype=object)
        df['id'] = df['id'].astype(str)
        df.fillna(value=1, inplace=True)
        return df

    def load_bibliographic_coupling(self, ids):
        self.check_connection()
        vals = self.ids_to_vals(ids)
        query = f'''SELECT C.pmid_in as cited, ARRAY_AGG(C.pmid_out) as citing_list
                    FROM PMCitations C
                    WHERE C.pmid_out IN (VALUES {vals}) AND C.pmid_in != C.pmid_out
                    GROUP BY cited
                    HAVING COUNT(*) >= 2
                    LIMIT {self.config.max_number_of_bibliographic_coupling};
                    '''

        logger.debug(f'load_bibliographic_coupling query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = process_bibliographic_coupling_postgres(ids, cursor)
        return df
