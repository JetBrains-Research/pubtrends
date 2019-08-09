import html

import numpy as np
import pandas as pd

from .loader import Loader


class SemanticScholarLoader(Loader):
    def __init__(self, pubtrends_config):
        super(SemanticScholarLoader, self).__init__(pubtrends_config)

    def search(self, *terms, limit=None, sort=None, current=0, task=None):
        self.terms = [t.lower() for t in terms]
        self.logger.info('Searching publication data', current=current, task=task)
        terms_str = '\'' + ' '.join(self.terms) + '\''

        columns = ['id', 'crc32id', 'title', 'abstract', 'year', 'aux']
        if sort == 'relevance':
            query = f'''
            SELECT P.ssid, P.crc32id, P.title, P.abstract, P.year, P.aux, ts_rank_cd(P.tsv, query) AS rank
            FROM SSPublications P, websearch_to_tsquery('english', {terms_str}) query
            WHERE tsv @@ query
            ORDER BY rank DESC
            LIMIT {limit};
            '''
            columns.append('ts_rank')
        elif sort == 'citations':
            query = f'''
            SELECT P.ssid, P.crc32id, P.title, P.abstract, P.year, P.aux, COUNT(1) AS count
            FROM SSPublications P
            LEFT JOIN SSCitations C
            ON C.crc32id_in = P.crc32id AND C.id_in = P.ssid
            WHERE tsv @@ websearch_to_tsquery('english', {terms_str})
            GROUP BY P.ssid, P.crc32id, P.title, P.abstract, P.year, P.aux
            ORDER BY count DESC
            LIMIT {limit};
            '''
            columns.append('citations')
        elif sort == 'year':
            query = f'''
            SELECT ssid, crc32id, title, abstract, year, aux
            FROM SSPublications P
            WHERE tsv @@ websearch_to_tsquery('english', {terms_str})
            ORDER BY year DESC NULLS LAST
            LIMIT {limit};
            '''
        else:
            raise ValueError('sort can be either citations, relevance or year')

        with self.conn:
            self.cursor.execute(query)
        self.pub_df = pd.DataFrame(self.cursor.fetchall(),
                                   columns=columns,
                                   dtype=object)

        # Duplicate rows may occur if crawler was stopped while parsing Semantic Scholar archive
        self.pub_df.drop_duplicates(subset='id', inplace=True)

        self.logger.info(f'Found {len(self.pub_df)} publications in the local database', current=current,
                         task=task)

        self.ids = list(self.pub_df['id'].values)
        crc32ids = list(self.pub_df['crc32id'].values)
        self.values = ', '.join(['({0}, \'{1}\')'.format(i, j) for (i, j) in zip(crc32ids, self.ids)])
        return self.ids, False

    def load_publications(self, temp_table_created=False, current=0, task=None):
        if not temp_table_created:
            self.logger.info('Loading publication data', current=current, task=task)
            query = f'''
                    DROP TABLE IF EXISTS temp_ssids;
                    WITH vals(crc32id, ssid) AS (VALUES {self.values})
                    SELECT crc32id, ssid INTO table temp_ssids FROM vals;
                    DROP INDEX IF EXISTS temp_ssids_index;
                    CREATE INDEX temp_ssids_index ON temp_ssids USING btree (crc32id);
                    '''

            with self.conn:
                self.cursor.execute(query)

            self.logger.debug('Created table for request with index.', current=current, task=task)

        if np.any(self.pub_df[['id', 'crc32id', 'title']].isna()):
            raise ValueError('Article must have ID and title')
        self.pub_df = self.pub_df.fillna(value={'abstract': ''})

        self.pub_df['year'] = self.pub_df['year'].apply(lambda year: int(year) if year else np.nan)
        self.pub_df['authors'] = self.pub_df['aux'].apply(
            lambda aux: ', '.join(map(lambda authors: html.unescape(authors['name']), aux['authors'])))
        self.pub_df['journal'] = self.pub_df['aux'].apply(lambda aux: html.unescape(aux['journal']['name']))
        self.pub_df['title'] = self.pub_df['title'].apply(lambda title: html.unescape(title))

        return self.pub_df

    def load_citation_stats(self, current=0, task=None):
        self.logger.info('Loading citations statistics: searching for correct citations over 150 million of citations',
                         current=current, task=task)

        query = f'''
           SELECT C.id_in AS ssid, P.year, COUNT(1) AS count
                FROM SSCitations C
                JOIN (VALUES {self.values}) AS CT(crc32id, ssid)
                  ON (C.crc32id_in = CT.crc32id AND C.id_in = CT.ssid)
                JOIN SSPublications P
                  ON C.crc32id_out = P.crc32id AND C.id_out = P.ssid
                WHERE C.crc32id_in
                between (SELECT MIN(crc32id) FROM temp_ssids)
                  AND (select max(crc32id) FROM temp_ssids)
                AND P.year > 0
                GROUP BY C.id_in, P.year;
            '''

        with self.conn:
            self.cursor.execute(query)
        self.logger.debug('Done loading citation stats', current=current, task=task)
        cit_stats_df_from_query = pd.DataFrame(self.cursor.fetchall(),
                                               columns=['id', 'year', 'count'], dtype=object)

        if np.any(cit_stats_df_from_query.isna()):
            raise ValueError('NaN values are not allowed in citation stats DataFrame')

        cit_stats_df_from_query['year'] = cit_stats_df_from_query['year'].apply(int)
        cit_stats_df_from_query['count'] = cit_stats_df_from_query['count'].apply(int)

        return cit_stats_df_from_query

    def load_citations(self, current=0, task=None):
        self.logger.info('Loading citations data', current=current, task=task)
        self.logger.debug('Started loading raw information about citations', current=current, task=task)

        query = f'''SELECT C.id_out, C.id_in
                    FROM SSCitations C
                    JOIN (VALUES {self.values}) AS CT(crc32id, ssid)
                    ON (C.crc32id_in = CT.crc32id AND C.id_in = CT.ssid)
                    WHERE C.crc32id_in
                    between (SELECT MIN(crc32id) FROM temp_ssids)
                    AND (select max(crc32id) FROM temp_ssids);
                    '''

        with self.conn:
            self.cursor.execute(query)
        citations = pd.DataFrame(self.cursor.fetchall(),
                                 columns=['id_out', 'id_in'], dtype=object)

        citations = citations[citations['id_out'].isin(self.ids)]

        if np.any(citations.isna()):
            raise ValueError('Citation must have id_out and id_in')

        self.logger.debug(f'Found {len(citations)} citations', current=current, task=task)

        return citations

    def load_cocitations(self, current=0, task=None):
        self.logger.info('Calculating co-citations for selected articles', current=current, task=task)

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

        with self.conn:
            self.cursor.execute(query)

        cocit_data = []
        lines = 0
        for row in self.cursor:
            lines += 1
            citing, year, cited = row
            for i in range(len(cited)):
                for j in range(i + 1, len(cited)):
                    cocit_data.append((citing, cited[i], cited[j], year))
        cocit_df = pd.DataFrame(cocit_data, columns=['citing', 'cited_1', 'cited_2', 'year'], dtype=object)

        if np.any(cocit_df[['citing', 'cited_1', 'cited_2']].isna()):
            raise ValueError('NaN values are not allowed in ids of co-citation DataFrame')
        cocit_df['year'] = cocit_df['year'].apply(lambda x: int(x) if x else np.nan)

        self.logger.debug(f'Loaded {lines} lines of citing info', current=current, task=task)
        self.logger.debug(f'Found {len(cocit_df)} co-cited pairs of articles', current=current, task=task)

        return cocit_df
