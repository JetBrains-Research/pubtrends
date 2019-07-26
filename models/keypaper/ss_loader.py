import html

import networkx as nx
import numpy as np
import pandas as pd

from .loader import Loader


class SemanticScholarLoader(Loader):
    def __init__(self,
                 pubtrends_config,
                 publications_table='sspublications',
                 citations_table='sscitations',
                 temp_ids_table='temp_ssids',
                 index='ssid'):
        super(SemanticScholarLoader, self).__init__(pubtrends_config)

        self.publications_table = publications_table
        self.citations_table = citations_table
        self.temp_ids_table = temp_ids_table

    def search(self, *terms, current=0, task=None):
        self.terms = [t.lower() for t in terms]
        self.logger.info('Searching publication data', current=current, task=task)
        terms_str = '\'' + ' '.join(self.terms) + '\''
        query = f'''
        SELECT DISTINCT ON(ssid) ssid, crc32id, title, abstract, year, aux FROM {self.publications_table} P
        WHERE tsv @@ websearch_to_tsquery('english', {terms_str}) limit 100000;
        '''

        with self.conn:
            self.cursor.execute(query)
        self.pub_df = pd.DataFrame(self.cursor.fetchall(),
                                   columns=['id', 'crc32id', 'title', 'abstract', 'year', 'aux'],
                                   dtype=object)

        self.pub_df['authors'] = self.pub_df['aux'].apply(
            lambda aux: ', '.join(map(lambda authors: html.unescape(authors['name']), aux['authors'])))

        self.logger.info(f'Found {len(self.pub_df)} publications in the local database', current=current,
                         task=task)

        self.pub_df['title'] = self.pub_df['title'].apply(lambda title: html.unescape(title))

        self.ids = self.pub_df['id']
        self.crc32ids = self.pub_df['crc32id']
        self.values = ', '.join(['({0}, \'{1}\')'.format(i, j) for (i, j) in zip(self.crc32ids, self.ids)])
        self.articles_found = len(self.ids)

    def load_publications(self, current=0, task=None):
        self.logger.info('Loading publication data', current=current, task=task)
        query = f'''
                DROP TABLE IF EXISTS {self.temp_ids_table};
                WITH vals(crc32id, ssid) AS (VALUES {self.values})
                SELECT crc32id, ssid INTO table {self.temp_ids_table} FROM vals;
                DROP INDEX IF EXISTS temp_ssids_index;
                CREATE INDEX temp_ssids_index ON {self.temp_ids_table} USING btree (crc32id);
                '''

        with self.conn:
            self.cursor.execute(query)

        self.logger.debug('Created table for request with index.', current=current, task=task)

    def load_citation_stats(self, filter_citations=True, current=0, task=None):
        self.logger.info('Loading citations statistics: searching for correct citations over 150 million of citations',
                         current=current, task=task)

        query = f'''
           SELECT C.id_in AS ssid, P.year, COUNT(1) AS count
                FROM {self.citations_table} C
                JOIN (VALUES {self.values}) AS CT(crc32id, ssid)
                  ON (C.crc32id_in = CT.crc32id AND C.id_in = CT.ssid)
                JOIN {self.publications_table} P
                  ON C.crc32id_out = P.crc32id AND C.id_out = P.ssid
                WHERE C.crc32id_in
                between (SELECT MIN(crc32id) FROM {self.temp_ids_table})
                  AND (select max(crc32id) FROM {self.temp_ids_table})
                AND P.year > 0
                GROUP BY C.id_in, P.year;
            '''

        with self.conn:
            self.cursor.execute(query)
        self.logger.debug('Done loading citation stats', current=current, task=task)
        self.cit_stats_df_from_query = pd.DataFrame(self.cursor.fetchall(),
                                                    columns=['id', 'year', 'count'], dtype=object)

        self.cit_df = self.cit_stats_df_from_query.pivot(index='id',
                                                         columns='year',
                                                         values='count') \
            .reset_index().replace(np.nan, 0)
        self.cit_df['total'] = self.cit_df.iloc[:, 1:].sum(axis=1)
        self.cit_df = self.cit_df.sort_values(by='total', ascending=False)
        self.logger.debug(f"Loaded citation stats for {len(self.cit_df)} of {len(self.ids)} articles.\n" +
                          "Others may either have zero citations or be absent in the local database.", current=current,
                          task=task)

        if filter_citations:
            self.logger.debug('Filtering top 100000 or 80% of all the citations', current=current, task=task)
            self.cit_df = self.cit_df.iloc[:min(100000, round(0.8 * len(self.cit_df))), :]

    def load_citations(self, current=0, task=None):
        self.logger.info('Loading citations data', current=current, task=task)
        self.logger.debug('Started loading raw information about citations', current=current, task=task)

        query = f'''SELECT C.id_out, C.id_in
                    FROM {self.citations_table} C
                    JOIN (VALUES {self.values}) AS CT(crc32id, ssid)
                    ON (C.crc32id_in = CT.crc32id AND C.id_in = CT.ssid)
                    JOIN (VALUES {self.values}) AS CT2(crc32id, ssid)
                    ON (C.crc32id_out = CT2.crc32id AND C.id_out = CT2.ssid);
                    '''

        with self.conn:
            self.cursor.execute(query)
        self.logger.info('Building citation graph', current=current, task=task)

        self.G = nx.DiGraph()
        for row in self.cursor:
            v, u = row
            self.G.add_edge(v, u)

        self.logger.debug(f'Built citation graph - nodes {len(self.G.nodes())} edges {len(self.G.edges())}',
                          current=current, task=task)

    def load_cocitations(self, current=0, task=None):
        self.logger.info('Calculating co-citations for selected articles', current=current, task=task)

        query = f'''
                with Z as (select id_out, id_in, crc32id_out, crc32id_in
                    from {self.citations_table}
                    where crc32id_in between (select min(crc32id) from {self.temp_ids_table})
                        and (select max(crc32id) from {self.temp_ids_table})
                        and (crc32id_in, id_in) in (select crc32id, ssid
                                                    from {self.temp_ids_table})),

                    X as (select id_out, array_agg(id_in) as cited_list,
                                 min(crc32id_out) as crc32id_out
                          from Z
                          group by id_out
                          having count(*) >= 2)

                select X.id_out, P.year, X.cited_list
                from X
                    join {self.publications_table} P
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
        self.cocit_df = pd.DataFrame(cocit_data, columns=['citing', 'cited_1', 'cited_2', 'year'], dtype=object)
        self.logger.debug(f'Loaded {lines} lines of citing info', current=current, task=task)
        self.logger.debug(f'Found {len(self.cocit_df)} co-cited pairs of articles', current=current, task=task)

        self.logger.debug(f'Aggregating co-citations', current=current, task=task)
        self.cocit_grouped_df = self.cocit_df.groupby(['cited_1', 'cited_2', 'year']).count().reset_index()
        self.cocit_grouped_df = self.cocit_grouped_df.pivot_table(index=['cited_1', 'cited_2'],
                                                                  columns=['year'],
                                                                  values=['citing']).reset_index()
        self.cocit_grouped_df = self.cocit_grouped_df.replace(np.nan, 0)
        self.cocit_grouped_df['total'] = self.cocit_grouped_df.iloc[:, 2:].sum(axis=1)
        self.cocit_grouped_df = self.cocit_grouped_df.sort_values(by='total', ascending=False)
        self.logger.debug('Filtering top 100000 of all the co-citations', current=current, task=task)
        self.cocit_grouped_df = self.cocit_grouped_df.iloc[:min(100000, len(self.cocit_grouped_df)), :]

        for col in self.cocit_grouped_df:
            self.cocit_grouped_df[col] = self.cocit_grouped_df[col].astype(object)
