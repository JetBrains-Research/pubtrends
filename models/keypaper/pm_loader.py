import html
import re

import networkx as nx
import numpy as np
import pandas as pd
from Bio import Entrez

from .loader import Loader


class PubmedLoader(Loader):
    def __init__(self,
                 pubtrends_config,
                 index='pmid'):
        super(PubmedLoader, self).__init__(pubtrends_config)
        Entrez.email = pubtrends_config.pm_entrez_email
        self.index = index

    def search(self, *terms, current=0, task=None):
        self.logger.debug('TODO: handle queries which return more than 100000 items', current=current, task=task)
        self.terms = [t.lower() for t in terms]
        query = ' '.join(terms).replace("\"", "")
        handle = Entrez.esearch(db='pubmed', retmax='100000',
                                retmode='xml', term=query)
        self.pmids = Entrez.read(handle)['IdList']
        self.articles_found = len(self.pmids)
        self.logger.info(f'Found {len(self.pmids)} articles about {terms}', current=current, task=task)

    def load_publications(self, current=0, task=None):
        self.logger.info('Loading publication data', current=current, task=task)

        values = ', '.join(['({})'.format(i) for i in sorted(self.pmids)])
        query = re.sub('\$VALUES\$', values, '''
        DROP TABLE IF EXISTS TEMP_PMIDS;
        WITH vals(pmid) AS (VALUES $VALUES$)
        SELECT pmid INTO temporary table TEMP_PMIDS FROM vals;
        DROP INDEX IF EXISTS temp_pmids_unique_index;
        CREATE UNIQUE INDEX temp_pmids_unique_index ON TEMP_PMIDS USING btree (pmid);

        SELECT P.pmid, P.title, P.aux, P.abstract, date_part('year', P.date) AS year
        FROM PMPublications P
        JOIN TEMP_PMIDS AS T ON (P.pmid = T.pmid);
        ''')
        self.logger.debug('Creating pmids table for request with index.', current=current, task=task)

        with self.conn:
            self.cursor.execute(query)
        self.pub_df = pd.DataFrame(self.cursor.fetchall(),
                                   columns=['pmid', 'title', 'aux', 'abstract', 'year'], dtype=object)

        self.pub_df['authors'] = self.pub_df['aux'].apply(
            lambda aux: ', '.join(map(lambda authors: html.unescape(authors['name']), aux['authors'])))

        self.logger.debug(f'Found {len(self.pub_df)} publications in the local database\n', current=current, task=task)

    def load_citation_stats(self, current=0, task=None):
        self.logger.info('Loading citations statistics: searching for correct citations over 168 million of citations',
                         current=current, task=task)

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
        self.logger.debug('Done loading citation stats', current=current, task=task)
        self.cit_df = pd.DataFrame(self.cursor.fetchall(),
                                   columns=['pmid', 'year', 'count'])

        self.cit_df = self.cit_df.pivot(index='pmid', columns='year', values='count').reset_index().replace(np.nan, 0)
        self.cit_df['total'] = self.cit_df.iloc[:, 1:].sum(axis=1)
        self.cit_df = self.cit_df.sort_values(by='total', ascending=False)
        self.logger.debug(f"Loaded citation stats for {len(self.cit_df)} of {len(self.pmids)} articles.\n" +
                          "Others may either have zero citations or be absent in the local database.", current=current,
                          task=task)

        self.logger.debug('Filtering top 100000 or 80% of all the citations', current=current, task=task)
        self.cit_df = self.cit_df.iloc[:min(100000, round(0.8 * len(self.cit_df))), :]

        self.df = pd.merge(self.pub_df, self.cit_df, on='pmid')
        self.pmids = sorted(list(self.df['pmid']))
        self.logger.debug(f'{len(self.df)} articles to process.\n', current=current, task=task)

    def load_citations(self, current=0, task=None):
        self.logger.info('Started loading raw information about citations', current=current, task=task)

        values = ', '.join(['({})'.format(i) for i in sorted(self.pmids)])
        query = re.sub('\$VALUES\$', values, '''
        SELECT C.pmid_in, C.pmid_out
        FROM PMCitations C
        JOIN (VALUES $VALUES$) AS CT(pmid) ON (C.pmid_in = CT.pmid)
        JOIN (VALUES $VALUES$) AS CT2(pmid) ON (C.pmid_out = CT2.pmid);
        ''')

        with self.conn:
            self.cursor.execute(query)
        self.logger.debug('Done loading citations, building citation graph', current=current, task=task)

        self.G = nx.DiGraph()
        for row in self.cursor:
            v, u = row
            self.G.add_edge(v, u)

        self.logger.debug(f'Built citation graph - nodes {len(self.G.nodes())} edges {len(self.G.edges())}',
                          current=current, task=task)

    def load_cocitations(self, current=0, task=None):
        self.logger.info('Calculating co-citations for selected articles', current=current, task=task)

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
        self.logger.debug(f'Loaded {lines} lines of citing info', current=current, task=task)
        self.logger.debug(f'Found {len(self.cocit_df)} co-cited pairs of articles', current=current, task=task)

        self.logger.debug(f'Aggregating co-citations', current=current, task=task)
        self.cocit_grouped_df = self.cocit_df.groupby(['cited_1', 'cited_2', 'year']).count().reset_index()
        self.cocit_grouped_df = self.cocit_grouped_df.pivot_table(index=['cited_1', 'cited_2'],
                                                                  columns=['year'], values=['citing']).reset_index()
        self.cocit_grouped_df = self.cocit_grouped_df.replace(np.nan, 0)
        self.cocit_grouped_df['total'] = self.cocit_grouped_df.iloc[:, 2:].sum(axis=1)
        self.cocit_grouped_df = self.cocit_grouped_df.sort_values(by='total', ascending=False)
        self.logger.debug('Filtering top 100000 of all the co-citations', current=current, task=task)
        self.cocit_grouped_df = self.cocit_grouped_df.iloc[:min(100000, len(self.cocit_grouped_df)), :]

        for col in self.cocit_grouped_df:
            self.cocit_grouped_df[col] = self.cocit_grouped_df[col].astype(object)

        self.logger.info(f'Building co-citations graph', current=current, task=task)
        self.CG = nx.Graph()
        # NOTE: we use nodes id as String to avoid problems str keys in jsonify during graph visualization
        for el in self.cocit_grouped_df[['cited_1', 'cited_2', 'total']].values:
            start, end, weight = el
            if start in self.pmids and end in self.pmids:
                self.CG.add_edge(str(start), str(end), weight=int(weight))
        self.logger.debug(f'Co-citations graph nodes {len(self.CG.nodes())} edges {len(self.CG.edges())}\n',
                          current=current, task=task)
