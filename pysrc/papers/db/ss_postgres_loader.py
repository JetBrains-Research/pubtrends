import logging

import numpy as np
import pandas as pd

from pysrc.papers.db.loader import Loader
from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.papers.db.postgres_utils import preprocess_search_query_for_postgres, \
    process_bibliographic_coupling_postgres, process_cocitations_postgres, no_stemming_filter
from pysrc.papers.utils import crc32, SORT_MOST_CITED, SORT_MOST_RECENT, preprocess_doi, \
    preprocess_search_title

logger = logging.getLogger(__name__)


class SemanticScholarPostgresLoader(PostgresConnector, Loader):
    def __init__(self, pubtrends_config):
        super(SemanticScholarPostgresLoader, self).__init__(pubtrends_config)

    @staticmethod
    def ids2values(ids):
        ids = list(ids)  # In case of generator, avoid problems with zip and map
        return ', '.join(f'({i}, \'{j}\')' for (i, j) in zip(map(crc32, ids), ids))

    def find(self, key, value):
        self.check_connection()
        value = value.strip()

        # Preprocess DOI
        if key == 'doi':
            value = preprocess_doi(value)

        if key == 'id':
            key = 'ssid'

        # Use dedicated text index to search title.
        if key == 'title':
            value = preprocess_search_title(value)
            query = f'''
                SELECT ssid
                FROM to_tsquery('english', \'''{value}\''') query, SSPublications P
                WHERE tsv @@ query AND LOWER(title) = LOWER('{value}');
            '''
        else:
            query = f'''
                SELECT ssid
                FROM SSPublications P
                WHERE {key} = {repr(value)};
            '''

        logger.debug(f'find query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['id'], dtype=object)

        return list(df['id'].astype(str))

    def search(self, query, limit=None, sort=None, noreviews=True):
        self.check_connection()
        if noreviews:
            logger.debug('Type is not supported for Semantic Scholar')
        query_str = preprocess_search_query_for_postgres(query, self.config.min_search_words)
        # Disable stemming-based lookup for now, see: https://github.com/JetBrains-Research/pubtrends/issues/242
        exact_filter = no_stemming_filter(query_str)

        by_citations = 'count DESC NULLS LAST'
        by_year = 'year DESC NULLS LAST'
        # 2 divides the rank by the document length
        # 4 divides the rank by the mean harmonic distance between extents (this is implemented only by ts_rank_cd)
        # See https://www.postgresql.org/docs/12/textsearch-controls.html#TEXTSEARCH-RANKING
        if sort == SORT_MOST_CITED:
            order = f'{by_citations}, ts_rank_cd(P.tsv, query, 2|4) DESC, {by_year}'
        elif sort == SORT_MOST_RECENT:
            order = f'{by_year}, ts_rank_cd(P.tsv, query, 2|4) DESC, {by_citations}'
        else:
            raise ValueError(f'Illegal sort method: {sort}')

        query = f'''
            SELECT P.ssid 
            FROM to_tsquery('{query_str}') query, 
            SSPublications P
            LEFT JOIN matview_sscitations C 
            ON C.ssid = P.ssid AND C.crc32id = P.crc32id
            WHERE P.tsv @@ query {exact_filter}
            ORDER BY {order}, P.crc32id
            LIMIT {limit};
            '''

        logger.debug(f'search query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            pub_df = pd.DataFrame(cursor.fetchall(), columns=['id'], dtype=object)

        # Duplicate rows may occur if crawler was stopped while parsing Semantic Scholar archive
        pub_df.drop_duplicates(subset='id', inplace=True)

        return list(pub_df['id'].values)

    def load_publications(self, ids):
        self.check_connection()
        # We can possible have multiple records for a single paper!!!
        # PostgreSQLUtil.kt#batchInsertOnDuplicateKeyUpdate is not used for Semantic Scholar
        query = f'''
                SELECT distinct on (P.ssid) P.ssid, P.crc32id, P.pmid, P.title, P.abstract, P.year, P.doi, P.aux
                FROM SSPublications P
                WHERE (P.crc32id, P.ssid) in (VALUES {SemanticScholarPostgresLoader.ids2values(ids)});
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
        pub_df['year'].fillna(1970, inplace=True)
        return Loader.process_publications_dataframe(ids, pub_df)

    def load_citations_by_year(self, ids):
        self.check_connection()
        query = f'''
           SELECT C.ssid_in AS ssid, P.year, COUNT(1) AS count
                FROM SSCitations C
                JOIN SSPublications P
                  ON C.crc32id_out = P.crc32id AND C.ssid_out = P.ssid
                WHERE (C.crc32id_in, C.ssid_in) IN (VALUES {SemanticScholarPostgresLoader.ids2values(ids)})
                GROUP BY C.ssid_in, P.year
                LIMIT {self.config.max_number_of_citations};
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
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
                SELECT C.ssid_in AS ssid
                FROM SSCitations C JOIN matview_sscitations MC
                    ON C.ssid_in = MC.ssid
                WHERE (C.crc32id_out, C.ssid_out) in (VALUES {SemanticScholarPostgresLoader.ids2values([pid])})
                ORDER BY MC.count DESC NULLS LAST
                LIMIT {limit};
                '''

        logger.debug(f'load_references query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['id'], dtype=object)
        return list(df['id'].astype(str))

    def estimate_citations(self, ids):
        self.check_connection()
        query = f'''
                SELECT count
                FROM SSPublications P
                    LEFT JOIN matview_sscitations C
                    ON C.crc32id = P.crc32id and C.ssid = P.ssid 
                WHERE (P.crc32id, P.ssid) in (VALUES {SemanticScholarPostgresLoader.ids2values(ids)});
                '''

        logger.debug(f'estimate_citations query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['total'], dtype=object)
            df.fillna(value=1, inplace=True)  # matview_sscitations ignores < 3 citations

        return df['total']

    def load_citations(self, ids):
        self.check_connection()
        query = f'''SELECT C.ssid_out, C.ssid_in
                    FROM SSCitations C
                    WHERE (C.crc32id_in, C.ssid_in) in (VALUES {SemanticScholarPostgresLoader.ids2values(ids)})
                        AND (C.crc32id_out, C.ssid_out) in (VALUES {SemanticScholarPostgresLoader.ids2values(ids)})
                    ORDER BY C.ssid_out, C.ssid_in
                    LIMIT {self.config.max_number_of_citations};
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
        query = f'''
                with Z as (select ssid_out, ssid_in, crc32id_out, crc32id_in
                    from SSCitations
                    where (crc32id_in, ssid_in) IN (VALUES  {SemanticScholarPostgresLoader.ids2values(ids)})),
                    X as (select ssid_out, array_agg(ssid_in) as cited_list,
                                 min(crc32id_out) as crc32id_out
                          from Z
                          group by ssid_out
                          having count(*) >= 2)
                select X.ssid_out, P.year, X.cited_list
                from X
                    join SSPublications P
                        on crc32id_out = P.crc32id and ssid_out = P.ssid
                LIMIT {self.config.max_number_of_cocitations};
                '''

        logger.debug(f'load_cocitations query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            cocit_df, lines = process_cocitations_postgres(cursor)

        if np.any(cocit_df[['citing', 'cited_1', 'cited_2']].isna()):
            raise ValueError('NaN values are not allowed in ids of co-citation DataFrame')
        return cocit_df

    def expand(self, ids, limit):
        self.check_connection()
        query = f'''
            WITH X AS (
                SELECT C.ssid_in as ssid, C.crc32id_in as crc32id
                FROM sscitations C
                WHERE (C.crc32id_out, C.ssid_out) IN (VALUES {SemanticScholarPostgresLoader.ids2values(ids)})
                UNION
                SELECT C.ssid_out as ssid, C.crc32id_out as crc32id
                FROM sscitations C
                WHERE (C.crc32id_in, C.ssid_in) IN (VALUES {SemanticScholarPostgresLoader.ids2values(ids)}))
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
        self.check_connection()
        query = f'''WITH X AS (SELECT ssid_out, ssid_in, crc32id_in
                        FROM sscitations C
                        WHERE (crc32id_out, ssid_out) IN (VALUES  {SemanticScholarPostgresLoader.ids2values(ids)}))
                        SELECT ssid_in, ARRAY_AGG(ssid_out) as citing_list
                        FROM X
                        GROUP BY crc32id_in, ssid_in
                        HAVING COUNT(*) >= 2
                        LIMIT {self.config.max_number_of_bibliographic_coupling};
                    '''

        logger.debug(f'load_bibliographic_coupling query: {query[:1000]}')
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df, lines = process_bibliographic_coupling_postgres(cursor)

        logger.debug(f'Loaded {lines} lines of bibliographic coupling info')
        return df
