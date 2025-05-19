import logging
import os
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2

from pysrc.papers.db.loader import Loader
from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.papers.db.postgres_utils import preprocess_search_query_for_postgres, \
    process_bibliographic_coupling_postgres, process_cocitations_postgres, ints_to_vals, strs_to_vals
from pysrc.papers.utils import SORT_MOST_CITED, SORT_MOST_RECENT

logger = logging.getLogger(__name__)


class PubmedPostgresLoader(PostgresConnector, Loader):
    def __init__(self, config):
        super(PubmedPostgresLoader, self).__init__(config)


    UPDATE_LAST_PATH = os.path.expanduser('~/.pubtrends/pubmedpostgreswriter_last.tsv')

    def last_update(self):
        if os.path.exists(self.UPDATE_LAST_PATH):
            return datetime.fromtimestamp(os.path.getmtime(self.UPDATE_LAST_PATH)).strftime('%Y-%m-%d %H:%M:%S')
        return None

    def search_id(self, pids):
        self.check_connection()
        pids2search = self.pids_to_list(pids)
        vals = ints_to_vals(pids2search)
        query = f'''
                SELECT pmid
                FROM PMPublications P
                WHERE P.pmid IN (VALUES {vals});
            '''
        logger.debug(f'find query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)
        return list(df['pmid'].astype(str))

    @staticmethod
    def pids_to_list(pids):
        pids2search = []
        for p in pids.split(';'):
            if p.strip() != '':
                try:
                    pids2search.append(int(p))
                except ValueError:
                    raise Exception(f"PMID should be an integer: {p}")
        return pids2search

    def search_doi(self, dois):
        self.check_connection()
        dois2search = self.dois_to_list(dois)
        vals = strs_to_vals(dois2search)
        query = f'''
                SELECT pmid
                FROM PMPublications P
                WHERE doi in (VALUES {vals});
            '''
        logger.debug(f'find query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)
        return list(df['pmid'].astype(str))

    def search_title(self, titles):
        self.check_connection()
        titles2search = self.titles_to_list(titles)
        pids = []
        for t in titles2search:
            query = f'''
                SELECT pmid
                FROM to_tsquery(\'''{t}\''') query, PMPublications P
                WHERE tsv @@ query AND TRIM(TRAILING '.' FROM LOWER(title)) = LOWER('{t}');
            '''
            logger.debug(f'find query: {query[:1000]}')
            with self.postgres_connection.cursor() as cursor:
                cursor.execute(query)
                df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)
                if len(df) > 0:
                    pids.extend(df['pmid'].astype(str))
                else:
                    pids.extend(self._search_title_relaxed(t))
        return pids

    def search(self, query, limit=None, sort=None, noreviews=True, min_year=None, max_year=None):
        self.check_connection()
        noreviews_filter = "AND type != 'Review'" if noreviews else ''

        # Add year filters
        min_year = int(min_year) if min_year else 1900
        max_year = int(max_year) if max_year else datetime.now().year
        year_filter = f'year BETWEEN {min_year} AND {max_year}'

        logger.debug(f'Preprocess search string for Postgres full text lookup query: {query}')
        query_str, exact_phrase_filter = preprocess_search_query_for_postgres(query)
        if exact_phrase_filter:
            exact_phrase_filter = f'AND ({exact_phrase_filter})'

        by_citations = 'count DESC NULLS LAST'
        by_year = 'year DESC NULLS LAST'
        # 2 divides the rank by the document length
        # 4 divides the rank by the mean harmonic distance between extents (this is implemented only by ts_rank_cd)
        # See https://www.postgresql.org/docs/12/textsearch-controls.html#TEXTSEARCH-RANKING
        if sort == SORT_MOST_CITED:
            order = f'{by_citations}, ts_rank_cd(tsv, query, 2|4) DESC, {by_year}'
        elif sort == SORT_MOST_RECENT:
            order = f'{by_year}, ts_rank_cd(tsv, query, 2|4) DESC, {by_citations}'
        elif sort is None:
            order = 'ts_rank_cd(tsv, query, 2|4) DESC'
        else:
            raise ValueError(f'Illegal sort method: {sort}')

        df = None
        sampling_fraction = 1
        sampling_filter = ''

        def cancel_query():
            nonlocal df
            if df is None:
                self.postgres_connection.cancel()

        while df is None:
            query = f'''
            WITH X AS
                (SELECT P.pmid as pmid, P.tsv as tsv, query, P.year as year
                FROM to_tsquery('{query_str}') query, 
                PMPublications P {sampling_filter}
                WHERE {year_filter} AND P.tsv @@ query {noreviews_filter} {exact_phrase_filter} 
                ORDER BY random()
                LIMIT {self.config.max_number_of_papers})
            SELECT X.pmid as pmid
            FROM X
            LEFT JOIN matview_pmcitations C 
            ON X.pmid = C.pmid
            ORDER BY {order}, X.pmid
            LIMIT {limit};
            '''
            logger.debug(f'search query: {query[:1000]}')
            with self.postgres_connection.cursor() as cursor:
                try:
                    # Wait for execution
                    timer = threading.Timer(self.config.max_search_time_sec, cancel_query)
                    timer.start()
                    cursor.execute(query)
                    df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)
                    timer.cancel()
                except psycopg2.extensions.QueryCanceledError:
                    sampling_fraction *= 0.5
                    logger.warning(f'search query timeout, apply sampling fraction {sampling_fraction}')
                    sampling_filter = f'TABLESAMPLE SYSTEM({sampling_fraction})'
                    if exact_phrase_filter:
                        logger.warning(f'disabling exact phrase filter')
                        query_str = query_str.replace('<->', '|')
                        exact_phrase_filter=''
                finally:
                    # TODO [shpynov] query stays idle in transaction without this commit
                    # Further investigation is required
                    self.postgres_connection.commit()

        df['pmid'] = df['pmid'].astype(str)
        return df['pmid'].to_list()

    def load_publications(self, ids):
        self.check_connection()
        vals = ints_to_vals(ids)
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
        vals = ints_to_vals(ids)
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
        vals = ints_to_vals([pid])
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
        vals = ints_to_vals(ids)
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
        vals = ints_to_vals(ids)
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
        vals = ints_to_vals(ids)
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

    def expand(self, ids, limit, noreviews):
        self.check_connection()
        vals = ints_to_vals(ids)
        noreviews_filter = "AND P.type != 'Review'" if noreviews else ''
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH X AS (
                SELECT C.pmid_in AS pmid
                FROM PMCitations C
                JOIN PMPublications P
                ON C.pmid_out = P.pmid
                WHERE C.pmid_in != C.pmid_out AND C.pmid_out IN (VALUES {vals}) AND C.pmid_in NOT IN (VALUES {vals}) 
                {noreviews_filter}
                UNION
                SELECT C.pmid_out AS pmid
                FROM PMCitations C 
                JOIN PMPublications P
                ON C.pmid_out = P.pmid
                WHERE C.pmid_in != C.pmid_out AND C.pmid_in IN (VALUES {vals}) AND C.pmid_out NOT IN (VALUES {vals}) 
                {noreviews_filter}
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
        vals = ints_to_vals(ids)
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
