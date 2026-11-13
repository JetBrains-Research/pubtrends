import logging
import os
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2

from pysrc.config import MAX_NUMBER_OF_PAPERS, MAX_NUMBER_OF_CITATIONS, MAX_NUMBER_OF_COCITATIONS, \
    MAX_NUMBER_OF_BIBLIOGRAPHIC_COUPLING, MAX_SEARCH_TIME_SEC
from pysrc.papers.db.loader import Loader
from pysrc.papers.db.lucene_searcher import LuceneIndexManager
from pysrc.papers.db.mesh_terms import MeshTermExpander
from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.papers.db.postgres_utils import process_bibliographic_coupling_postgres, \
    process_cocitations_postgres, ints_to_vals, to_int_list
from pysrc.papers.utils import SORT_MOST_CITED, SORT_MOST_RECENT

logger = logging.getLogger(__name__)


class PubmedPostgresLoader(PostgresConnector, Loader):
    def __init__(self, config):
        super(PubmedPostgresLoader, self).__init__(config)
        self.lucene_searcher = LuceneIndexManager(config.pm_lucene_index_path, config.bm25_k1, config.bm25_b)
        self.mesh_term_expander = MeshTermExpander(
            lambda: self.postgres_connection,
            enabled=config.mesh_query_expansion,
        )


    UPDATE_LAST_PATH = os.path.expanduser('~/.pubtrends/pubmedpostgreswriter_last.tsv')

    def last_update(self):
        if os.path.exists(self.UPDATE_LAST_PATH):
            return datetime.fromtimestamp(os.path.getmtime(self.UPDATE_LAST_PATH)).strftime('%Y-%m-%d %H:%M:%S')
        return None

    def search_id(self, pids):
        self.check_connection()
        pids2search = self.pids_to_list(pids)
        query = '''
                SELECT pmid
                FROM PMPublications P
                WHERE P.pmid = ANY(%s);
            '''
        logger.debug(f'find query: {query}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query, (pids2search,))
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
        query = '''
                SELECT pmid
                FROM PMPublications P
                WHERE doi = ANY(%s);
            '''
        logger.debug(f'find query: {query}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query, (dois2search,))
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)
        return list(df['pmid'].astype(str))

    def search_title(self, titles):
        self.check_connection()
        titles2search = self.titles_to_list(titles)
        pids = []
        for t in titles2search:
            candidate_ids = self.lucene_searcher.search_title(t, limit=100)
            if candidate_ids:
                int_ids = [int(cid) for cid in candidate_ids]
                query = """
                    SELECT pmid
                    FROM PMPublications P
                    WHERE pmid = ANY(%s) AND TRIM(TRAILING '.' FROM LOWER(title)) = LOWER(%s);
                """
                logger.debug(f'find query: {query[:1000]}')
                with self.postgres_connection.cursor() as cursor:
                    cursor.execute(query, (int_ids, t))
                    df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)
                    if len(df) > 0:
                        pids.extend(df['pmid'].astype(str))
                    else:
                        pids.extend(self._search_title_relaxed(t))
            else:
                pids.extend(self._search_title_relaxed(t))
        return pids

    def search(self, query, limit=None, sort=None, noreviews=True, min_year=None, max_year=None):
        self.check_connection()
        min_year = int(min_year) if min_year else None
        max_year = int(max_year) if max_year else None

        if sort not in (None, SORT_MOST_CITED, SORT_MOST_RECENT):
            raise ValueError(f'Illegal sort method: {sort}')

        if sort == SORT_MOST_CITED:
            candidate_ids = self.lucene_searcher.search(
                query,
                limit=MAX_NUMBER_OF_PAPERS,
                sort=None,
                noreviews=noreviews,
                min_year=min_year,
                max_year=max_year,
                query_expander=self.mesh_term_expander,
            )
            if not candidate_ids:
                return []
            int_ids = [int(cid) for cid in candidate_ids]
            query_sql = """
                SELECT P.pmid, COALESCE(C.count, 0) AS cit_count, P.year
                FROM PMPublications P
                LEFT JOIN matview_pmcitations C ON P.pmid = C.pmid
                WHERE P.pmid = ANY(%s);
            """
            with self.postgres_connection.cursor() as cursor:
                cursor.execute(query_sql, (int_ids,))
                rows = cursor.fetchall()
                df = pd.DataFrame(rows, columns=['pmid', 'cit_count', 'year'])
            if len(df) == 0:
                return []
            order_map = {pid: i for i, pid in enumerate(int_ids)}
            df['rank'] = df['pmid'].map(order_map)
            df.sort_values(by=['cit_count', 'rank', 'year'], ascending=[False, True, False], inplace=True)
            if limit is not None:
                df = df.iloc[:limit]
            return list(df['pmid'].astype(str))

        return self.lucene_searcher.search(
            query,
            limit=limit,
            sort=sort,
            noreviews=noreviews,
            min_year=min_year,
            max_year=max_year,
            query_expander=self.mesh_term_expander,
        )

    def load_publications(self, ids):
        self.check_connection()
        query = f'''
                SELECT P.pmid as id, title, abstract, year, type, keywords, mesh, doi, aux
                FROM PMPublications P
                WHERE P.pmid = ANY (%s);
                '''
        logger.debug(f'load_publications query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query, (to_int_list(ids),))
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
        query = f'''
            WITH X as (SELECT pmid_in, pmid_out
                FROM PMCitations C
                JOIN PMPublications P
                ON C.pmid_in = P.pmid
                WHERE C.pmid_in != C.pmid_out AND P.pmid = ANY (%s))
            SELECT X.pmid_in AS id, year, COUNT(1) AS count
            FROM X
                JOIN PMPublications P
                ON X.pmid_out = P.pmid
                GROUP BY id, year
                LIMIT %s;
            '''
        logger.debug(f'load_citations_by_year query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query, (to_int_list(ids), MAX_NUMBER_OF_CITATIONS))
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
        query = f'''
                SELECT C.pmid_in AS pmid
                FROM PMCitations C
                JOIN matview_pmcitations MC
                    ON C.pmid_in = MC.pmid
                WHERE C.pmid_in != C.pmid_out AND
                C.pmid_out = ANY (%s)
                ORDER BY MC.count DESC NULLS LAST
                LIMIT %s;
                '''
        logger.debug(f'load_references query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query, (to_int_list([pid]), limit))
            df = pd.DataFrame(cursor.fetchall(), columns=['id'], dtype=object)
        return list(df['id'].astype(str))

    def load_citations_counts(self, ids):
        self.check_connection()
        query = f'''
                SELECT count
                FROM PMPublications P
                    LEFT JOIN matview_pmcitations C
                    ON P.pmid = C.pmid
                WHERE P.pmid = ANY (%s);
                '''
        logger.debug(f'estimate_citations query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query, (to_int_list(ids),))
            df = pd.DataFrame(cursor.fetchall(), columns=['total'], dtype=object)
            df.fillna(value=1, inplace=True)  # matview_pmcitations ignores < 3 citations

        return list(df['total'])

    def load_citations(self, ids):
        self.check_connection()
        query = f'''SELECT DISTINCT pmid_out as id_out, pmid_in as id_in
                    FROM PMCitations C
                    WHERE pmid_out != pmid_in AND
                    pmid_in = ANY (%s) AND
                    pmid_out = ANY (%s)
                    ORDER BY id_out, id_in
                    LIMIT %s;
                    '''
        logger.debug(f'load_citations query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            vals = to_int_list(ids)
            cursor.execute(query, (vals, vals, MAX_NUMBER_OF_CITATIONS))
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
        query = f'''SELECT C.pmid_out as citing, year, ARRAY_AGG(DISTINCT C.pmid_in) as cited_list
                        FROM PMCitations C
                        JOIN PMPublications P
                            ON C.pmid_out = P.pmid
                        WHERE C.pmid_in != C.pmid_out AND
                        C.pmid_in = ANY (%s)
                        GROUP BY citing, year
                        HAVING COUNT(DISTINCT C.pmid_in) >= 2
                        LIMIT %s;
                    '''

        logger.debug(f'load_cocitations query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query, (to_int_list(ids), MAX_NUMBER_OF_COCITATIONS))
            df = process_cocitations_postgres(cursor)

        if np.any(df[['citing', 'cited_1', 'cited_2']].isna()):
            raise ValueError('NaN values are not allowed in ids of co-citation DataFrame')

        return df

    def expand(self, ids, limit, noreviews):
        self.check_connection()
        noreviews_filter = "AND P.type != 'Review'" if noreviews else ''
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH X AS (
                (SELECT C.pmid_in AS pmid
                FROM PMCitations C
                JOIN PMPublications P
                ON C.pmid_out = P.pmid
                WHERE C.pmid_in != C.pmid_out AND
                C.pmid_out = ANY (%s) AND
                NOT (C.pmid_in = ANY (%s))
                {noreviews_filter}
                LIMIT %s)
                UNION
                (SELECT C.pmid_out AS pmid
                FROM PMCitations C
                JOIN PMPublications P
                ON C.pmid_out = P.pmid
                WHERE C.pmid_in != C.pmid_out AND
                C.pmid_in = ANY (%s) AND
                NOT (C.pmid_out = ANY (%s))
                {noreviews_filter}
                LIMIT %s)
            )
            SELECT X.pmid as pmid, count
                FROM X
                    LEFT JOIN matview_pmcitations C
                    ON X.pmid = C.pmid
                ORDER BY count, pmid DESC NULLS LAST;
                '''

        logger.debug(f'expand query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            vals = to_int_list(ids)
            limit = int(limit * 0.75)
            cursor.execute(query, (vals, vals, limit, vals, vals, limit))
            df = pd.DataFrame(cursor.fetchall(), columns=['id', 'total'], dtype=object)
        df['id'] = df['id'].astype(str)
        df.fillna(value=1, inplace=True)
        return df

    def load_bibliographic_coupling(self, ids):
        self.check_connection()
        query = f'''SELECT C.pmid_in as cited, ARRAY_AGG(DISTINCT C.pmid_out) as citing_list
                    FROM PMCitations C
                    WHERE C.pmid_out = ANY (%s) AND C.pmid_in != C.pmid_out
                    GROUP BY cited
                    HAVING COUNT(DISTINCT C.pmid_out) >= 2
                    LIMIT %s;
                    '''

        logger.debug(f'load_bibliographic_coupling query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query, (to_int_list(ids), MAX_NUMBER_OF_BIBLIOGRAPHIC_COUPLING))
            df = process_bibliographic_coupling_postgres(ids, cursor)
        return df
