import html
import logging
from collections import Iterable

import numpy as np
import pandas as pd

from pysrc.papers.db.loader import Loader
from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.papers.utils import crc32, SORT_MOST_RELEVANT, SORT_MOST_CITED, SORT_MOST_RECENT, preprocess_doi, \
    preprocess_pubmed_search_title

logger = logging.getLogger(__name__)


class SemanticScholarPostgresLoader(PostgresConnector, Loader):
    def __init__(self, pubtrends_config):
        super(SemanticScholarPostgresLoader, self).__init__(pubtrends_config)

    @staticmethod
    def ids2values(ids):
        return ', '.join([f'({i}, \'{j}\')' for (i, j) in zip(map(crc32, ids), ids)])

    def find(self, key, value, current=1, task=None):
        value = value.strip()
        self.progress.info(f"Searching for a publication with {key} '{value}'", current=current, task=task)

        # Preprocess DOI
        if key == 'doi':
            value = preprocess_doi(value)

        if key == 'id':
            key = 'ssid'

        # Use dedicated text index to search title.
        if key == 'title':
            value = preprocess_pubmed_search_title(value)
            query = f'''
                SELECT ssid
                FROM websearch_to_tsquery('english', '{value}') query, SSPublications P 
                WHERE tsv @@ query AND title = '{value}';
            '''
        else:
            query = f'''
                SELECT ssid
                FROM SSPublications P 
                WHERE {key} = {repr(value)};
            '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['id'], dtype=object)

        self.progress.info(f'Found {len(df)} publications in the local database', current=current,
                           task=task)

        return list(df['id'].astype(str))

    def search(self, query, limit=None, sort=None, current=0, task=None):
        self.progress.info(html.escape(f'Searching publications matching <{query}>'), current=current, task=task)
        query_str = '\'' + query + '\''
        if not limit:
            limit_message = ''
            limit = self.config.max_number_of_articles
        else:
            limit_message = f'{limit} '

        self.progress.info(html.escape(f'Searching {limit_message}{sort.lower()} publications matching <{query}>'),
                           current=current, task=task)

        if sort == SORT_MOST_RELEVANT:
            query = f'''
                SELECT ssid
                FROM websearch_to_tsquery('english', {query_str}) query, SSPublications P 
                WHERE tsv @@ query
                ORDER BY ts_rank_cd(P.tsv, query) DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_CITED:
            query = f'''
                SELECT P.ssid
                FROM websearch_to_tsquery('english', {query_str}) query, SSPublications P
                    LEFT JOIN SSCitations C
                        ON C.ssid_in = P.ssid
                WHERE tsv @@ query
                GROUP BY ssid
                ORDER BY COUNT(*) DESC NULLS LAST
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_RECENT:
            query = f'''
                SELECT ssid
                FROM websearch_to_tsquery('english', {query_str}) query, SSPublications P
                WHERE tsv @@ query
                ORDER BY year DESC NULLS LAST
                LIMIT {limit};
                '''
        else:
            raise ValueError(f'Illegal sort method: {sort}')

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            pub_df = pd.DataFrame(cursor.fetchall(), columns=['id'], dtype=object)

        # Duplicate rows may occur if crawler was stopped while parsing Semantic Scholar archive
        pub_df.drop_duplicates(subset='id', inplace=True)

        self.progress.info(f'Found {len(pub_df)} publications in the local database', current=current,
                           task=task)

        return (list(pub_df['id'].values)), False

    def load_publications(self, ids, temp_table_created=False, current=0, task=None):
        if not temp_table_created:
            self.progress.info('Loading publication data', current=current, task=task)

        query = f'''
                SELECT P.ssid, P.crc32id, P.title, P.abstract, P.year, P.type, P.aux
                FROM SSPublications P
                WHERE (P.crc32id, P.ssid) in (VALUES {SemanticScholarPostgresLoader.ids2values(ids)});
                '''
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            pub_df = pd.DataFrame(cursor.fetchall(),
                                  columns=['id', 'crc32id', 'title', 'abstract', 'year', 'type', 'aux'], dtype=object)

        if np.any(pub_df[['id', 'crc32id', 'title']].isna()):
            raise ValueError('Paper must have ID and title')

        return Loader.process_publications_dataframe(pub_df)

    def load_citation_stats(self, ids, current=0, task=None):
        self.progress.info('Loading citations statistics among millions of citations',
                           current=current, task=task)
        query = f'''
           SELECT C.ssid_in AS ssid, P.year, COUNT(1) AS count
                FROM SSCitations C
                JOIN SSPublications P
                  ON C.crc32id_out = P.crc32id AND C.ssid_out = P.ssid
                WHERE (C.crc32id_in, C.ssid_in) IN (VALUES {SemanticScholarPostgresLoader.ids2values(ids)}) 
                AND P.year > 0
                GROUP BY C.ssid_in, P.year;
            '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            cit_stats_df_from_query = pd.DataFrame(cursor.fetchall(),
                                                   columns=['id', 'year', 'count'], dtype=object)

        if np.any(cit_stats_df_from_query.isna()):
            raise ValueError('NaN values are not allowed in citation stats DataFrame')

        cit_stats_df_from_query['year'] = cit_stats_df_from_query['year'].apply(int)
        cit_stats_df_from_query['count'] = cit_stats_df_from_query['count'].apply(int)

        self.progress.info(f'Found {cit_stats_df_from_query.shape[0]} records of citations by year',
                           current=current, task=task)

        return cit_stats_df_from_query

    def load_citations(self, ids, current=0, task=None):
        self.progress.info('Loading citations data', current=current, task=task)
        logger.debug('Started loading raw information about citations', current=current, task=task)

        query = f'''SELECT C.ssid_out, C.ssid_in
                    FROM SSCitations C
                    WHERE (C.crc32id_in, C.ssid_in) in (VALUES {SemanticScholarPostgresLoader.ids2values(ids)}) 
                        AND (C.crc32id_out, C.ssid_out) in (VALUES {SemanticScholarPostgresLoader.ids2values(ids)});
                    '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            citations = pd.DataFrame(cursor.fetchall(),
                                     columns=['id_out', 'id_in'], dtype=object)

        # TODO[shpynov] we can make it on DB side
        citations = citations[citations['id_out'].isin(ids)]

        if np.any(citations.isna()):
            raise ValueError('Citation must have ssid_out and ssid_in')

        self.progress.info(f'Found {len(citations)} citations', current=current, task=task)

        return citations

    def load_cocitations(self, ids, current=0, task=None):
        self.progress.info('Calculating co-citations for selected papers', current=current, task=task)

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
                        on crc32id_out = P.crc32id and ssid_out = P.ssid;
                '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)

            cocit_data = []
            lines = 0
            for row in cursor:
                lines += 1
                citing, year, cited = row
                for i in range(len(cited)):
                    for j in range(i + 1, len(cited)):
                        cocit_data.append((citing, cited[i], cited[j], year))
            cocit_df = pd.DataFrame(cocit_data, columns=['citing', 'cited_1', 'cited_2', 'year'], dtype=object)

        if np.any(cocit_df[['citing', 'cited_1', 'cited_2']].isna()):
            raise ValueError('NaN values are not allowed in ids of co-citation DataFrame')
        cocit_df['year'] = cocit_df['year'].apply(lambda x: int(x) if x else np.nan)

        logger.debug(f'Loaded {lines} lines of citing info', current=current, task=task)
        self.progress.info(f'Found {len(cocit_df)} co-cited pairs of papers', current=current, task=task)

        return cocit_df

    def expand(self, ids, limit, current=1, task=None):
        max_to_expand = limit - len(ids)
        if isinstance(ids, Iterable):
            self.progress.info('Expanding current topic', current=current, task=task)

            query = f'''
                SELECT C.ssid_out as id_inner, ARRAY_AGG(C.ssid_in) as ids_outer
                FROM sscitations C
                WHERE (C.crc32id_out, C.ssid_out) IN (VALUES  {SemanticScholarPostgresLoader.ids2values(ids)})
                GROUP BY C.ssid_out
                UNION
                SELECT C.ssid_in as id_inner, ARRAY_AGG(C.ssid_out) as ids_outer
                FROM sscitations C
                WHERE (C.crc32id_in, C.ssid_in) IN (VALUES  {SemanticScholarPostgresLoader.ids2values(ids)})
                GROUP BY C.ssid_in
                LIMIT {max_to_expand};
                '''
        elif isinstance(ids, int):
            crc32id = crc32(ids)
            query = f'''
                SELECT C.ssid_out as id_inner, ARRAY_AGG(C.ssid_in) as ids_outer
                FROM sscitations C
                WHERE (C.crc32id_out = {crc32id} AND C.ssid_out = '{ids}')
                GROUP BY C.ssid_out
                UNION
                SELECT C.ssid_in as id_inner, ARRAY_AGG(C.ssid_out) as ids_outer
                FROM sscitations C
                WHERE (C.crc32id_in = {crc32id} AND C.ssid_in = '{ids}')
                GROUP BY C.ssid_in
                LIMIT {max_to_expand};
                '''
        else:
            raise TypeError('ids should be either int or Iterable')

        expanded = set()
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)

            for row in cursor.fetchall():
                citing, cited = row
                expanded.add(citing)
                expanded |= set(cited)

        self.progress.info(f'Found {len(expanded)} papers', current=current, task=task)
        return expanded

    def load_bibliographic_coupling(self, ids, current=1, task=None):
        self.progress.info('Processing bibliographic coupling for selected papers', current=current, task=task)
        query = f'''WITH X AS (SELECT ssid_out, ssid_in
                        FROM sscitations C
                        WHERE (crc32id_out, ssid_out) IN (VALUES  {SemanticScholarPostgresLoader.ids2values(ids)}))
                        
                        SELECT ssid_in, ARRAY_AGG(ssid_out) as citing_list
                        FROM X
                        GROUP BY ssid_in
                        HAVING COUNT(*) >= 2;
                    '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)

            data = []
            lines = 0
            for row in cursor:
                lines += 1
                _, citing = row
                for i in range(len(citing)):
                    for j in range(i + 1, len(citing)):
                        data.append((str(citing[i]), str(citing[j]), 1))
            df = pd.DataFrame(data, columns=['citing_1', 'citing_2', 'total'], dtype=object)
            df = df.groupby(['citing_1', 'citing_2']).sum().reset_index()

        logger.debug(f'Loaded {lines} lines of bibliographic coupling info')
        self.progress.info(f'Found {len(df)} bibliographic coupling pairs of papers',
                           current=current, task=task)

        return df