import re

import networkx as nx
import numpy as np
import pandas as pd
from Bio import Entrez

from .loader import Loader


class PubmedLoader(Loader):
    def __init__(self, index='pmid'):
        super(PubmedLoader, self).__init__()
        self.index = index

    def search(self, *terms):
        self.logger.info('TODO: handle queries which return more than 10000 items')
        self.terms = [t.lower() for t in terms]
        query = ' '.join(terms)
        handle = Entrez.esearch(db='pubmed', retmax='100000',
                                retmode='xml', term=query)
        self.pmids = Entrez.read(handle)['IdList']
        self.logger.info(f'Found {len(self.pmids)} articles about {terms}')

    def load_publications(self):
        self.logger.info('Loading publication data')

        values = ', '.join(['({})'.format(i) for i in sorted(self.pmids)])
        query = re.sub('\$VALUES\$', values, '''
        DROP TABLE IF EXISTS TEMP_PMIDS;
        WITH vals(pmid) AS (VALUES $VALUES$)
        SELECT pmid INTO temporary table TEMP_PMIDS FROM vals;
        DROP INDEX IF EXISTS temp_pmids_unique_index;
        CREATE UNIQUE INDEX temp_pmids_unique_index ON TEMP_PMIDS USING btree (pmid);

        SELECT P.pmid, P.title, P.abstract, date_part('year', P.date) AS year
        FROM PMPublications P
        JOIN TEMP_PMIDS AS T ON (P.pmid = T.pmid);
        ''')
        self.logger.info('Creating pmids table for request with index.')

        with self.conn:
            self.cursor.execute(query)
        self.pub_df = pd.DataFrame(self.cursor.fetchall(),
                                   columns=['pmid', 'title', 'abstract', 'year'], dtype=object)
        self.logger.info(f'Found {len(self.pub_df)} publications in the local database\n')

    def load_citation_stats(self):
        self.logger.info('Started loading citation stats')

        values = ', '.join(['({})'.format(i) for i in sorted(self.pmids)])
        query = re.sub('\$VALUES\$', values, '''
        SELECT C.pmid_in AS pmid, date_part('year', P.date) AS year, COUNT(1) AS count
        FROM PMCitations C
        JOIN (VALUES $VALUES$) AS CT(pmid) ON (C.pmid_in = CT.pmid)
        JOIN PMPublications P
        ON C.pmid_out = P.pmid
        WHERE date_part('year', P.date) > 0
        GROUP BY C.pmid_in, date_part('year', P.date);
        ''')

        with self.conn:
            self.cursor.execute(query)
        self.logger.info('Done loading citation stats')
        self.cit_df = pd.DataFrame(self.cursor.fetchall(),
                                   columns=['pmid', 'year', 'count'])

        self.cit_df = self.cit_df.pivot(index='pmid', columns='year', values='count').reset_index().replace(np.nan, 0)
        self.cit_df['total'] = self.cit_df.iloc[:, 1:].sum(axis=1)
        self.cit_df = self.cit_df.sort_values(by='total', ascending=False)
        self.logger.info(f"Loaded citation stats for {len(self.cit_df)} of {len(self.pmids)} articles.\n" +
                         "Others may either have zero citations or be absent in the local database.")

        self.logger.info('Filtering top 100000 or 80% of all the citations')
        self.cit_df = self.cit_df.iloc[:min(100000, round(0.8 * len(self.cit_df))), :]

        self.df = pd.merge(self.pub_df, self.cit_df, on='pmid')
        self.pmids = sorted(list(self.df['pmid']))
        self.logger.info(f'{len(self.df)} articles to process.\n')

    def load_citations(self):
        self.logger.info('Started loading raw information about citations')

        values = ', '.join(['({})'.format(i) for i in sorted(self.pmids)])
        query = re.sub('\$VALUES\$', values, '''
        SELECT C.pmid_in, C.pmid_out
        FROM PMCitations C
        JOIN (VALUES $VALUES$) AS CT(pmid) ON (C.pmid_in = CT.pmid)
        JOIN (VALUES $VALUES$) AS CT2(pmid) ON (C.pmid_out = CT2.pmid);
        ''')

        with self.conn:
            self.cursor.execute(query)
        self.logger.info('Done loading citations, building citation graph')

        self.G = nx.DiGraph()
        for row in self.cursor:
            v, u = row
            self.G.add_edge(v, u)

        self.logger.info(f'Built citation graph - nodes {len(self.G.nodes())} edges {len(self.G.edges())}')

    def load_cocitations(self):
        self.logger.info('Calculating co-citations for selected articles')

        # Use unfolding to pairs on the client side instead of DataBase
        query = '''
        with Z as (select pmid_out, pmid_in
            from PMCitations
            -- Hack to make Postgres use index!
            where pmid_in between (select min(pmid) from TEMP_PMIDS) and (select max(pmid) from TEMP_PMIDS)
            and pmid_in in (select pmid from TEMP_PMIDS)),
        X as (select pmid_out, array_agg(pmid_in) as cited_list
            from Z
            group by pmid_out
            having count(*) >= 2)
        select X.pmid_out, date_part('year', P.date) AS year, X.cited_list from
            X join PMPublications P
            on pmid_out = P.pmid;
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
        self.logger.info(f'Loaded {lines} lines of citing info')
        self.logger.info(f'Found {len(self.cocit_df)} co-cited pairs of articles')

        self.logger.info(f'Aggregating co-citations')
        self.cocit_grouped_df = self.cocit_df.groupby(['cited_1', 'cited_2', 'year']).count().reset_index()
        self.cocit_grouped_df = self.cocit_grouped_df.pivot_table(index=['cited_1', 'cited_2'],
                                                                  columns=['year'], values=['citing']).reset_index()
        self.cocit_grouped_df = self.cocit_grouped_df.replace(np.nan, 0)
        self.cocit_grouped_df['total'] = self.cocit_grouped_df.iloc[:, 2:].sum(axis=1)
        self.cocit_grouped_df = self.cocit_grouped_df.sort_values(by='total', ascending=False)
        self.logger.info('Filtering top 100000 of all the co-citations')
        self.cocit_grouped_df = self.cocit_grouped_df.iloc[:min(100000, len(self.cocit_grouped_df)), :]

        for col in self.cocit_grouped_df:
            self.cocit_grouped_df[col] = self.cocit_grouped_df[col].astype(object)

        self.logger.info(f'Building co-citations graph')
        self.CG = nx.Graph()
        # NOTE: we use nodes id as String to avoid problems str keys in jsonify during graph visualization
        for el in self.cocit_grouped_df[['cited_1', 'cited_2', 'total']].values:
            start, end, weight = el
            if start in self.pmids and end in self.pmids:
                self.CG.add_edge(str(start), str(end), weight=int(weight))
        self.logger.info(f'Co-citations graph nodes {len(self.CG.nodes())} edges {len(self.CG.edges())}\n')
