import html
import logging
from collections import Iterable

import numpy as np
import pandas as pd

from .loader import Loader
from ..utils import crc32, SORT_MOST_RELEVANT, SORT_MOST_CITED, SORT_MOST_RECENT, preprocess_doi

logger = logging.getLogger(__name__)


class SemanticScholarLoader(Loader):
    def __init__(self, pubtrends_config):
        super(SemanticScholarLoader, self).__init__(pubtrends_config)

    def find(self, key, value, current=1, task=None):
        value = value.strip()
        self.progress.info(f"Searching for a publication with {key} '{value}'", current=current, task=task)

        if key == 'id':
            key = 'ssid'

        # Preprocess DOI
        if key == 'doi':
            value = preprocess_doi(value)
        pass

    def load_bibliographic_coupling(self, ids, current=1, task=None):
        pass

    @staticmethod
    def ids2values(ids):
        return ', '.join([f'({i}, \'{j}\')' for (i, j) in zip(map(crc32, ids), ids)])

    def search(self, query, limit=None, sort=None, current=0, task=None):
        self.progress.info(html.escape(f'Searching publications matching <{query}>'), current=current, task=task)
        query_str = '\'' + query + '\''
        if not limit:
            limit_message = ''
            limit = self.config.max_number_of_articles
        else:
            limit_message = f'{limit} '

        columns = ['id', 'crc32id', 'pmid']
        self.progress.info(html.escape(f'Searching {limit_message}{sort.lower()} publications matching <{query}>'),
                           current=current, task=task)

        if sort == SORT_MOST_RELEVANT:
            query = f'''
                SELECT P.ssid, P.crc32id, P.pmid, ts_rank_cd(P.tsv, query) AS rank
                FROM SSPublications P, websearch_to_tsquery('english', {query_str}) query
                WHERE tsv @@ query
                ORDER BY rank DESC
                LIMIT {limit};
                '''
            columns.append('ts_rank')
        elif sort == SORT_MOST_CITED:
            query = f'''
                SELECT P.ssid, P.crc32id, P.pmid, COUNT(1) AS count
                FROM SSPublications P
                LEFT JOIN SSCitations C
                ON C.crc32id_in = P.crc32id AND C.id_in = P.ssid
                WHERE tsv @@ websearch_to_tsquery('english', {query_str})
                GROUP BY P.ssid, P.crc32id, P.pmid
                ORDER BY count DESC NULLS LAST
                LIMIT {limit};
                '''
            columns.append('citations')
        elif sort == SORT_MOST_RECENT:
            query = f'''
                SELECT ssid, crc32id, pmid
                FROM SSPublications P
                WHERE tsv @@ websearch_to_tsquery('english', {query_str})
                ORDER BY year DESC NULLS LAST
                LIMIT {limit};
                '''
        else:
            raise ValueError(f'Illegal sort method: {sort}')

        with self.pg_connection.cursor() as cursor:
            cursor.execute(query)
            pub_df = pd.DataFrame(cursor.fetchall(), columns=columns, dtype=object)

        # Duplicate rows may occur if crawler was stopped while parsing Semantic Scholar archive
        pub_df.drop_duplicates(subset='id', inplace=True)
        # IMPORTANT Remove Pubmed ids from account
        pub_df = pub_df[pd.isnull(pub_df.pmid)]

        self.progress.info(f'Found {len(pub_df)} publications in the local database', current=current,
                           task=task)

        return (list(pub_df['id'].values)), False

    def load_publications(self, ids, temp_table_created=False, current=0, task=None):
        if not temp_table_created:
            self.progress.info('Loading publication data', current=current, task=task)
            query = f'''
                    DROP TABLE IF EXISTS temp_ssids;
                    WITH vals(crc32id, ssid) AS (VALUES {SemanticScholarLoader.ids2values(ids)})
                    SELECT crc32id, ssid INTO table temp_ssids FROM vals;
                    DROP INDEX IF EXISTS temp_ssids_index;
                    CREATE INDEX temp_ssids_index ON temp_ssids USING btree (crc32id);
                    '''

            with self.pg_connection.cursor() as cursor:
                cursor.execute(query)

            logger.debug('Created table for request with index.', current=current, task=task)

        columns = ['id', 'crc32id', 'title', 'abstract', 'year', 'type', 'aux']
        query = f'''
                SELECT P.ssid, P.crc32id, P.title, P.abstract, P.year, P.type, P.aux
                FROM SSPublications P
                LEFT JOIN temp_ssids TS
                ON P.crc32id = TS.crc32id AND P.ssid = TS.ssid;
                '''
        with self.pg_connection.cursor() as cursor:
            cursor.execute(query)
            pub_df = pd.DataFrame(cursor.fetchall(), columns=columns, dtype=object)

        if np.any(pub_df[['id', 'crc32id', 'title']].isna()):
            raise ValueError('Paper must have ID and title')

        return Loader.process_publications_dataframe(pub_df)

    def load_citation_stats(self, ids, current=0, task=None):
        self.progress.info('Loading citations statistics among millions of citations',
                           current=current, task=task)
        query = f'''
           SELECT C.id_in AS ssid, P.year, COUNT(1) AS count
                FROM SSCitations C
                JOIN (VALUES {SemanticScholarLoader.ids2values(ids)}) AS CT(crc32id, ssid)
                  ON (C.crc32id_in = CT.crc32id AND C.id_in = CT.ssid)
                JOIN SSPublications P
                  ON C.crc32id_out = P.crc32id AND C.id_out = P.ssid
                WHERE C.crc32id_in
                between (SELECT MIN(crc32id) FROM temp_ssids)
                  AND (select max(crc32id) FROM temp_ssids)
                AND P.year > 0
                GROUP BY C.id_in, P.year;
            '''

        with self.pg_connection.cursor() as cursor:
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

        query = f'''SELECT C.id_out, C.id_in
                    FROM SSCitations C
                    JOIN (VALUES {SemanticScholarLoader.ids2values(ids)}) AS CT(crc32id, ssid)
                    ON (C.crc32id_in = CT.crc32id AND C.id_in = CT.ssid)
                    WHERE C.crc32id_in between (SELECT MIN(crc32id) FROM temp_ssids)
                    AND (select max(crc32id) FROM temp_ssids);
                    '''

        with self.pg_connection.cursor() as cursor:
            cursor.execute(query)
            citations = pd.DataFrame(cursor.fetchall(),
                                     columns=['id_out', 'id_in'], dtype=object)

        # TODO[shpynov] we can make it on DB side
        citations = citations[citations['id_out'].isin(ids)]

        if np.any(citations.isna()):
            raise ValueError('Citation must have id_out and id_in')

        self.progress.info(f'Found {len(citations)} citations', current=current, task=task)

        return citations

    def load_cocitations(self, ids, current=0, task=None):
        self.progress.info('Calculating co-citations for selected papers', current=current, task=task)

        query = f'''
                with Z as (select id_out, id_in, crc32id_out, crc32id_in
                    from SSCitations
                    where crc32id_in between (select min(crc32id) from temp_ssids)
                        and (select max(crc32id) from temp_ssids)
                        and (crc32id_in, id_in) in (select crc32id, ssid
                                                    from temp_ssids)),

                    X as (select id_out, array_agg(id_in) as cited_list,
                                 min(crc32id_out) as crc32id_out
                          from Z
                          group by id_out
                          having count(*) >= 2)

                select X.id_out, P.year, X.cited_list
                from X
                    join SSPublications P
                        on crc32id_out = P.crc32id and id_out = P.ssid;
                '''

        with self.pg_connection.cursor() as cursor:
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
            values = SemanticScholarLoader.ids2values(ids)

            query = f'''
                DROP TABLE IF EXISTS TEMP_SSIDS;
                WITH vals(crc32id, ssid) AS (VALUES {values})
                SELECT crc32id, ssid INTO table TEMP_SSIDS FROM vals;
                DROP INDEX IF EXISTS temp_ssids_unique_index;
                CREATE UNIQUE INDEX temp_ssids_unique_index ON TEMP_SSIDS USING btree (crc32id);

                SELECT C.id_out, ARRAY_AGG(C.id_in)
                FROM sscitations C
                JOIN temp_ssids T
                ON (C.crc32id_out = T.crc32id OR C.crc32id_in = T.crc32id)
                AND (C.id_out = T.ssid OR C.id_in = T.ssid)
                GROUP BY C.id_out
                LIMIT {max_to_expand};
                '''
        elif isinstance(ids, int):
            crc32id = crc32(ids)
            query = f'''
                SELECT C.id_out, ARRAY_AGG(C.id_in)
                FROM sscitations C
                WHERE (C.crc32id_out = {crc32id} OR C.crc32id_in = {crc32id})
                AND (C.id_out = {ids} OR C.id_in = {ids})
                GROUP BY C.id_out
                LIMIT {max_to_expand};
                '''
        else:
            raise TypeError('ids should be either int or Iterable')

        expanded = set()
        with self.pg_connection.cursor() as cursor:
            cursor.execute(query)

            for row in cursor.fetchall():
                citing, cited = row
                expanded.add(citing)
                expanded |= set(cited)

        self.progress.info(f'Found {len(expanded)} papers', current=current, task=task)
        return expanded
