import logging
import os
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2

from pysrc.config import MAX_NUMBER_OF_PAPERS, MAX_NUMBER_OF_CITATIONS, MAX_NUMBER_OF_COCITATIONS, \
    MAX_SEARCH_TIME_SEC
from pysrc.papers.db.loader import Loader
from pysrc.papers.db.lucene_searcher import LuceneIndexManager
from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.papers.db.postgres_utils import process_cocitations_postgres, preprocess_quotes, strs_to_vals
from pysrc.papers.utils import crc32, SORT_MOST_CITED, SORT_MOST_RECENT

logger = logging.getLogger(__name__)


class SemanticScholarPostgresLoader(PostgresConnector, Loader):
    def __init__(self, pubtrends_config):
        super(SemanticScholarPostgresLoader, self).__init__(pubtrends_config)
        self.lucene_searcher = LuceneIndexManager(pubtrends_config.ss_lucene_index_path,
                                                  pubtrends_config.bm25_k1,
                                                  pubtrends_config.bm25_b)

    @staticmethod
    def ids2values(ids):
        ids = list(ids)  # In case of generator, avoid problems with zip and map
        return ', '.join(f'({i}, \'{j}\')' for (i, j) in zip(map(crc32, ids), ids))

    UPDATE_STATS_PATH = os.path.expanduser('~/.pubtrends/semantic_scholar_stats.tsv')

    def last_update(self):
        if os.path.exists(self.UPDATE_STATS_PATH):
            return datetime.fromtimestamp(os.path.getmtime(self.UPDATE_STATS_PATH)).strftime('%Y-%m-%d %H:%M:%S')
        return None

    def search_id(self, pids):
        self.check_connection()
        pids = preprocess_quotes(pids)
        pids2search = []
        for p in pids.split(';'):
            if p.strip() != '':
                pids2search.append(p.strip())
        vals = SemanticScholarPostgresLoader.ids2values(pids2search)
        query = f'''
                SELECT ssid
                FROM SSPublications P
                WHERE (P.crc32id, P.ssid) in (VALUES {vals});
            '''
        logger.debug(f'find query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)
        return list(df['pmid'].astype(str))


    def search_doi(self, dois):
        self.check_connection()
        dois2search = self.dois_to_list(dois)
        vals = strs_to_vals(dois2search)
        query = f'''
                SELECT ssid
                FROM SSPublications P
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
            candidate_ids = self.lucene_searcher.search_title(t, limit=100)
            if candidate_ids:
                vals = strs_to_vals(candidate_ids)
                query = f'''
                    SELECT ssid
                    FROM SSPublications P
                    WHERE ssid in (VALUES {vals}) AND TRIM(TRAILING '.' FROM LOWER(title)) = LOWER(%s);
                '''
                logger.debug(f'find query: {query[:1000]}')
                with self.postgres_connection.cursor() as cursor:
                    cursor.execute(query, (t,))
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
        if noreviews:
            logger.debug('Type is not supported for Semantic Scholar')

        min_year = int(min_year) if min_year else None
        max_year = int(max_year) if max_year else None

        if sort not in (None, SORT_MOST_CITED, SORT_MOST_RECENT):
            raise ValueError(f'Illegal sort method: {sort}')

        if sort == SORT_MOST_CITED:
            candidate_ids = self.lucene_searcher.search(
                query,
                limit=MAX_NUMBER_OF_PAPERS,
                sort=None,
                noreviews=False,
                min_year=min_year,
                max_year=max_year,
            )
            if not candidate_ids:
                return []
            vals = strs_to_vals(candidate_ids)
            query_sql = f'''
                SELECT P.ssid, COALESCE(C.count, 0) as cit_count, P.year
                FROM SSPublications P
                LEFT JOIN matview_sscitations C
                ON C.ssid = P.ssid AND C.crc32id = P.crc32id
                WHERE P.ssid IN (VALUES {vals});
            '''
            with self.postgres_connection.cursor() as cursor:
                cursor.execute(query_sql)
                rows = cursor.fetchall()
                df = pd.DataFrame(rows, columns=['ssid', 'cit_count', 'year'])
            if len(df) == 0:
                return []
            df.drop_duplicates(subset='ssid', inplace=True)
            order_map = {ssid: i for i, ssid in enumerate(candidate_ids)}
            df['rank'] = df['ssid'].map(order_map)
            df.sort_values(by=['cit_count', 'rank', 'year'], ascending=[False, True, False], inplace=True)
            if limit is not None:
                df = df.iloc[:limit]
            return list(df['ssid'].astype(str))

        result_ids = self.lucene_searcher.search(
            query,
            limit=limit,
            sort=sort,
            noreviews=False,
            min_year=min_year,
            max_year=max_year,
        )
        seen = set()
        unique_ids = []
        for rid in result_ids:
            if rid not in seen:
                seen.add(rid)
                unique_ids.append(rid)
        return unique_ids

    def load_publications(self, ids):
        self.check_connection()
        # We can possibly have multiple records for a single paper!!!
        # PostgreSQLUtil.kt#batchInsertOnDuplicateKeyUpdate is not used for Semantic Scholar
        vals = SemanticScholarPostgresLoader.ids2values(ids)
        query = f'''
                SELECT distinct on (P.ssid) P.ssid, P.crc32id, P.pmid, P.title, P.abstract, P.year, P.doi, P.aux
                FROM SSPublications P
                WHERE (P.crc32id, P.ssid) in (VALUES {vals});
                '''
        logger.debug(f'load_publications query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            pub_df = pd.DataFrame(cursor.fetchall(),
                                  columns=['id', 'crc32id', 'pmid', 'title', 'abstract',
                                           'year', 'doi', 'aux'],
                                  dtype=object)

        if np.any(pub_df[['id', 'crc32id', 'title']].isna()):
            raise ValueError('Paper must have ID and title')
        pub_df['pmid'] = pub_df['pmid'].astype(str)

        # Hack for missing type in SS, see https://github.com/JetBrains-Research/pubtrends/issues/200
        pub_df['type'] = 'Article'
        pub_df['mesh'] = ''
        pub_df['keywords'] = ''
        # Hack for missing year in SS, see https://github.com/JetBrains-Research/pubtrends/issues/258
        pub_df['year'] = pub_df['year'].fillna(1970)
        return Loader.process_publications_dataframe(ids, pub_df)

    def load_citations_by_year(self, ids):
        self.check_connection()
        vals = SemanticScholarPostgresLoader.ids2values(ids)
        query = f'''
           SELECT C.ssid_in AS ssid, P.year, COUNT(1) AS count
                FROM SSCitations C
                JOIN SSPublications P
                  ON C.crc32id_out = P.crc32id AND C.ssid_out = P.ssid
                WHERE (C.crc32id_in, C.ssid_in) IN (VALUES {vals})
                GROUP BY C.ssid_in, P.year
                LIMIT {MAX_NUMBER_OF_CITATIONS};
            '''

        logger.debug(f'load_citations_by_year query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            cit_stats_df_from_query = pd.DataFrame(cursor.fetchall(),
                                                   columns=['id', 'year', 'count'],
                                                   dtype=object)

        # Hack for missing year in SS
        cit_stats_df_from_query.dropna(subset=['year'], inplace=True)
        if np.any(cit_stats_df_from_query.isna()):
            raise ValueError('NaN values are not allowed in citation stats DataFrame')

        cit_stats_df_from_query['year'] = cit_stats_df_from_query['year'].apply(int)
        cit_stats_df_from_query['count'] = cit_stats_df_from_query['count'].apply(int)

        return cit_stats_df_from_query

    def load_references(self, pid, limit):
        self.check_connection()
        vals = SemanticScholarPostgresLoader.ids2values([pid])
        query = f'''
                SELECT C.ssid_in AS ssid
                FROM SSCitations C JOIN matview_sscitations MC
                    ON C.ssid_in = MC.ssid
                WHERE (C.crc32id_out, C.ssid_out) in (VALUES {vals})
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
        vals = SemanticScholarPostgresLoader.ids2values(ids)
        query = f'''
                SELECT count
                FROM SSPublications P
                    LEFT JOIN matview_sscitations C
                    ON C.crc32id = P.crc32id and C.ssid = P.ssid
                WHERE (P.crc32id, P.ssid) in (VALUES {vals});
                '''

        logger.debug(f'estimate_citations query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['total'], dtype=object)
            df.fillna(value=1, inplace=True)  # matview_sscitations ignores < 3 citations

        return list(df['total'])

    def load_citations(self, ids):
        self.check_connection()
        vals = SemanticScholarPostgresLoader.ids2values(ids)
        query = f'''SELECT C.ssid_out, C.ssid_in
                    FROM SSCitations C
                    WHERE (C.crc32id_in, C.ssid_in) in (VALUES {vals})
                        AND (C.crc32id_out, C.ssid_out) in (VALUES {vals})
                    ORDER BY C.ssid_out, C.ssid_in
                    LIMIT {MAX_NUMBER_OF_CITATIONS};
                    '''

        logger.debug(f'load_citations query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            citations = pd.DataFrame(cursor.fetchall(),
                                     columns=['id_out', 'id_in'],
                                     dtype=object)

        # TODO[shpynov] we can make it on DB side
        citations = citations[citations['id_out'].isin(ids)]

        if np.any(citations.isna()):
            raise ValueError('Citation must have ssid_out and ssid_in')

        return citations

    def load_cocitations(self, ids):
        self.check_connection()
        vals = SemanticScholarPostgresLoader.ids2values(ids)
        query = f'''
                with Z as (select ssid_out, ssid_in, crc32id_out, crc32id_in
                    from SSCitations
                    where (crc32id_in, ssid_in) IN (VALUES {vals})),
                    X as (select ssid_out, array_agg(ssid_in) as cited_list,
                                 min(crc32id_out) as crc32id_out
                          from Z
                          group by ssid_out
                          having count(*) >= 2)
                select X.ssid_out, P.year, X.cited_list
                from X
                    join SSPublications P
                        on crc32id_out = P.crc32id and ssid_out = P.ssid
                LIMIT {MAX_NUMBER_OF_COCITATIONS};
                '''

        logger.debug(f'load_cocitations query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            cocit_df = process_cocitations_postgres(cursor)

        if np.any(cocit_df[['citing', 'cited_1', 'cited_2']].isna()):
            raise ValueError('NaN values are not allowed in ids of co-citation DataFrame')
        return cocit_df

    def expand(self, ids, limit, noreviews):
        self.check_connection()
        if noreviews:
            logger.debug('Type is not supported for Semantic Scholar')
        vals = SemanticScholarPostgresLoader.ids2values(ids)
        query = f'''
            WITH X AS (
                SELECT C.ssid_in as ssid, C.crc32id_in as crc32id
                FROM sscitations C
                WHERE (C.crc32id_out, C.ssid_out) IN (VALUES {vals})
                    AND (C.crc32id_in, C.ssid_in) NOT IN (VALUES {vals})
                UNION
                SELECT C.ssid_out as ssid, C.crc32id_out as crc32id
                FROM sscitations C
                WHERE (C.crc32id_in, C.ssid_in) IN (VALUES {vals})
                    AND (C.crc32id_out, C.ssid_out) NOT IN (VALUES {vals})
            )
            SELECT X.ssid, count
                FROM X
                    LEFT JOIN matview_sscitations C
                    ON X.ssid = C.ssid AND X.crc32id = C.crc32id
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
        # Workaround for Loading bibliographic coupling takes too long for 1k papers in Semantic Scholar #273
        return pd.DataFrame(columns=['citing_1', 'citing_2', 'total'], dtype=object)
        # self.check_connection()
        # vals = SemanticScholarPostgresLoader.ids2values(ids)
        # query = f'''WITH X AS (SELECT ssid_out, ssid_in, crc32id_in
        #                 FROM sscitations C
        #                 WHERE (crc32id_out, ssid_out) IN (VALUES  {vals}))
        #                 SELECT ssid_in, ARRAY_AGG(ssid_out) as citing_list
        #                 FROM X
        #                 GROUP BY crc32id_in, ssid_in
        #                 HAVING COUNT(*) >= 2
        #                 LIMIT {self.config.MAX_NUMBER_OF_BIBLIOGRAPHIC_COUPLING};
        #             '''
        #
        # logger.debug(f'load_bibliographic_coupling query: {query[:1000]}')
        # with self.postgres_connection.cursor() as cursor:
        #     cursor.execute(query)
        #     df = process_bibliographic_coupling_postgres(cursor)
        #
        # return df
