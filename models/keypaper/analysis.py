import community
import logging
import networkx as nx
import numpy as np
import pandas as pd
import psycopg2 as pg_driver
import re

from Bio import Entrez

from .utils import get_subtopic_descriptions


class KeyPaperAnalyzer:
    def __init__(self,
                 host='localhost', port='5432', dbname='pubmed',
                 user='biolabs', password='pubtrends',
                 email='nikolay.kapralov@gmail.com'):
        Entrez.email = email
        connection_string = f"""
        dbname={dbname} user={user} password={password} host={host} port={port}
        """

        self.conn = pg_driver.connect(connection_string)
        self.cursor = self.conn.cursor()

    def launch(self, *terms, task=None):
        # Search articles relevant to the terms
        self.search(*terms)
        if task:
            task.update_state(state='PROGRESS', meta={'current': 1, 'total': 10})

        # Load data about publications, citations and co-citations
        self.load_publications()
        if task:
            task.update_state(state='PROGRESS', meta={'current': 2, 'total': 10})
        self.load_citation_stats()
        if task:
            task.update_state(state='PROGRESS', meta={'current': 3, 'total': 10})
        self.load_cocitations()
        if task:
            task.update_state(state='PROGRESS', meta={'current': 4, 'total': 10})

        # Calculate min and max year of publications
        self.update_years()
        if task:
            task.update_state(state='PROGRESS', meta={'current': 5, 'total': 10})
        # Perform basic analysis
        self.subtopic_analysis()
        if task:
            task.update_state(state='PROGRESS', meta={'current': 6, 'total': 10})
        self.find_top_cited_papers()  # run after subtopic analysis to color components
        if task:
            task.update_state(state='PROGRESS', meta={'current': 7, 'total': 10})
        self.find_max_gain_papers()
        if task:
            task.update_state(state='PROGRESS', meta={'current': 8, 'total': 10})
        self.find_max_relative_gain_papers()
        if task:
            task.update_state(state='PROGRESS', meta={'current': 9, 'total': 10})
        self.subtopic_evolution_analysis()
        if task:
            task.update_state(state='PROGRESS', meta={'current': 10, 'total': 10})

    def search(self, *terms):
        print('TODO: handle queries which return more than 1000000 items')
        print('TODO: use local database instead of PubMed API')
        self.terms = [t.lower() for t in terms]
        query = ' '.join(terms)
        handle = Entrez.esearch(db='pubmed', retmax='1000000',
                                retmode='xml', term=query)
        self.pmids = [int(pmid) for pmid in Entrez.read(handle)['IdList']]
        logging.info(f'Found {len(self.pmids)} articles about {terms}')

    def load_publications(self):
        logging.info('Loading publication data')

        values = ', '.join(['({})'.format(i) for i in sorted(self.pmids)])
        query = re.sub('\$VALUES\$', values, '''
        DROP TABLE IF EXISTS TEMP_PMIDS;
        WITH vals(pmid) AS (VALUES $VALUES$)
        SELECT pmid INTO temporary table TEMP_PMIDS FROM vals;
        DROP INDEX IF EXISTS temp_pmids_unique_index;
        CREATE UNIQUE INDEX temp_pmids_unique_index ON TEMP_PMIDS USING btree (pmid);

        SELECT P.pmid, P.title, P.year
        FROM Publications P
        JOIN TEMP_PMIDS AS T ON (P.pmid = T.pmid);
        ''')
        logging.info('Creating pmids table for request with index.')

        with self.conn:
            self.cursor.execute(query)
        self.pub_df = pd.DataFrame(self.cursor.fetchall(),
                                   columns=['pmid', 'title', 'year'])
        logging.info(f'Found {len(self.pub_df)} publications in the local database\n')

    def load_citation_stats(self):
        logging.info('Started loading citation stats')

        values = ', '.join(['({})'.format(i) for i in sorted(self.pmids)])
        query = re.sub('\$VALUES\$', values, '''
        SELECT C.pmid_cited AS pmid, P.year, COUNT(1) AS count
        FROM Citations C
        JOIN (VALUES $VALUES$) AS CT(pmid) ON (C.pmid_cited = CT.pmid)
        JOIN Publications P
        ON C.pmid_citing = P.pmid
        WHERE P.year > 0
        GROUP BY C.pmid_cited, P.year;
        ''')

        with self.conn:
            self.cursor.execute(query)
        logging.info('Done loading citation stats')
        self.cit_df = pd.DataFrame(self.cursor.fetchall(),
                                   columns=['pmid', 'year', 'count'])

        self.cit_df = self.cit_df.pivot(index='pmid', columns='year', values='count').reset_index().replace(np.nan, 0)
        self.cit_df['total'] = self.cit_df.iloc[:, 1:].sum(axis=1)
        self.cit_df = self.cit_df.sort_values(by='total', ascending=False)
        logging.info(f"Loaded citation stats for {len(self.cit_df)} of {len(self.pmids)} articles. " +
                     "Others may either have zero citations or be absent in the local database.")

        #        logging.info('Filtering top 1000 or 50% of all the papers')
        #        self.cit_df = self.cit_df.iloc[:min(1000, round(0.5 * len(self.cit_df))), :]
        #        logging.info('Done aggregation')

        self.df = pd.merge(self.pub_df, self.cit_df, on='pmid')
        self.pmids = sorted(list(self.df['pmid']))
        logging.info(f'{len(self.df)} articles are further analyzed\n')

    def update_years(self):
        years = self.df.columns.values[3:-2].astype(int)
        self.min_year, self.max_year = np.min(years), np.max(years)

    def load_cocitations(self):
        logging.info('Calculating co-citations for selected articles')

        # Use unfolding to pairs on the client side instead of DataBase
        query = '''
        with Z as (select pmid_citing, pmid_cited
            from citations
            -- Hack to make Postgres use index!
            where pmid_cited between %s and %s
            and pmid_cited in (select pmid from TEMP_PMIDS)),
        X as (select pmid_citing, array_agg(pmid_cited) as cited_list
            from Z
            group by pmid_citing
            having count(*) >= 2)
        select X.pmid_citing, P.year, X.cited_list from
            X join publications P
            on pmid_citing = P.pmid;
        '''

        with self.conn:
            self.cursor.execute(query, (min(self.pmids), max(self.pmids),))

        cocit_data = []
        lines = 0
        for row in self.cursor:
            lines += 1
            citing, year, cited = row
            for i in range(len(cited)):
                for j in range(i + 1, len(cited)):
                    cocit_data.append((citing, cited[i], cited[j], year))
        self.cocit_df = pd.DataFrame(cocit_data, columns=['citing', 'cited_1', 'cited_2', 'year'])
        logging.info(f'Loaded {lines} lines of citing info')
        logging.info(f'Found {len(self.cocit_df)} co-cited pairs of articles')

        logging.info(f'Aggregating co-citations')
        self.cocit_grouped_df = self.cocit_df.groupby(['cited_1', 'cited_2', 'year']).count().reset_index()
        self.cocit_grouped_df = self.cocit_grouped_df.pivot_table(index=['cited_1', 'cited_2'],
                                                                  columns=['year'], values=['citing']).reset_index()
        self.cocit_grouped_df = self.cocit_grouped_df.replace(np.nan, 0)
        self.cocit_grouped_df['total'] = self.cocit_grouped_df.iloc[:, 2:].sum(axis=1)
        self.cocit_grouped_df = self.cocit_grouped_df.sort_values(by='total', ascending=False)
        logging.info('Filtering top 10000 or 80% of all the co-citations')
        self.cocit_grouped_df = self.cocit_grouped_df.iloc[:min(10000, round(0.8 * len(self.cocit_grouped_df))), :]

        logging.info(f'Building co-citations graph')
        self.CG = nx.Graph()
        # NOTE: we use nodes id as String to avoid problems str keys in jsonify during graph visualization
        for el in self.cocit_grouped_df[['cited_1', 'cited_2', 'total']].values.astype(int):
            start, end, weight = el
            if start in self.pmids and end in self.pmids:
                self.CG.add_edge(str(start), str(end), weight=weight)
        logging.info(f'Co-citations graph nodes {len(self.CG.nodes())} edges {len(self.CG.edges())}\n')

    def find_top_cited_papers(self, max_papers=20, threshold=0.1):
        logging.info(f'Identifying top cited papers overall')
        papers_to_show = min(max_papers, round(len(self.df) * threshold))
        self.top_cited_df = self.df.sort_values(by='total', ascending=False).iloc[:papers_to_show, :]
        self.top_cited_papers = set(self.top_cited_df['pmid'].astype(int).values)

    def find_max_gain_papers(self):
        logging.info('Identifying papers with max citation gain for each year')
        max_gain_data = []
        cols = self.df.columns[3:-2]
        for i in range(len(cols)):
            max_gain = self.df[cols[i]].astype(int).max()
            if max_gain > 0:
                sel = self.df[self.df[cols[i]] == max_gain]
                max_gain_data.append([cols[i], str(sel['pmid'].values[0]),
                                      sel['title'].values[0],
                                      sel['year'].values[0], max_gain])

        self.max_gain_df = pd.DataFrame(max_gain_data,
                                        columns=['year', 'pmid', 'title',
                                                 'paper_year', 'count'])
        self.max_gain_papers = set(self.max_gain_df['pmid'].astype(int).values)

    def find_max_relative_gain_papers(self):
        logging.info('Identifying papers with max relative citation gain for each year\n')
        current_sum = pd.Series(np.zeros(len(self.df), ))
        cols = self.df.columns[3:-2]
        df_rel = self.df.loc[:, ['pmid', 'title', 'year']]
        for col in cols:
            df_rel[col] = self.df[col] / (current_sum + (current_sum == 0))
            current_sum += self.df[col]

        max_rel_gain_data = []
        cols = self.df.columns[3:-2]
        for col in cols:
            max_rel_gain = df_rel[col].max()
            if max_rel_gain > 1e-6:
                sel = df_rel[df_rel[col] == max_rel_gain]
                max_rel_gain_data.append([col, str(sel['pmid'].values[0]),
                                          sel['title'].values[0],
                                          sel['year'].values[0], max_rel_gain])

        self.max_rel_gain_df = pd.DataFrame(max_rel_gain_data,
                                            columns=['year', 'pmid', 'title',
                                                     'paper_year', 'rel_gain'])
        self.max_rel_gain_papers = set(self.max_rel_gain_df['pmid'].astype(int).values)

    def subtopic_analysis(self, sort_components_key='size'):
        # Graph clustering via Louvain algorithm
        logging.info(f'Louvain community clustering of co-citation graph')
        p = community.best_partition(self.CG)
        self.components = set(p.values())
        logging.info(f'Found {len(self.components)} components')
        logging.info(f'Graph modularity: {community.modularity(p, self.CG):.3f}')

        # Merge small components to 'Other'
        GRANULARITY = 0.01
        logging.info(f'Merging components smaller than {GRANULARITY} to "Other" component')
        threshold = int(GRANULARITY * len(p))
        comp_sizes = {com: sum([p[node] == com for node in p.keys()]) for com in self.components}
        comp_to_merge = {com: comp_sizes[com] <= threshold for com in self.components}
        self.components_merged = sum(comp_to_merge.values()) > 0
        if self.components_merged > 0:
            logging.info(f'Reassigning components')
            pm = {}
            newcomps = {}
            ci = 1  # Other component is 0.
            for k, v in p.items():
                if comp_sizes[v] <= threshold:
                    pm[k] = 0  # Other
                    continue
                if v not in newcomps:
                    newcomps[v] = ci
                    ci += 1
                pm[k] = newcomps[v]
            logging.info(f'Processed {len(set(pm.values()))} components')
        else:
            logging.info(f'All components are bigger than {GRANULARITY}, no need to reassign')
            pm = p
        self.components = set(pm.values())
        self.pm = pm
        pmcomp_sizes = {com: sum([pm[node] == com for node in pm.keys()]) for com in self.components}
        for k, v in pmcomp_sizes.items():
            logging.info(f'Cluster {k}: {v} ({int(100 * v / len(pm))}%)')

        # Added 'comp' column containing the ID of component
        pm_ints = {int(k): v for k, v in pm.items()}
        df_comp = pd.Series(pm_ints).reset_index().rename(columns={'index': 'pmid', 0: 'comp'})
        self.df = pd.merge(self.df, df_comp, on='pmid')

        # Get n-gram descriptions for subtopics
        logging.info('Getting n-gram descriptions for subtopics')
        df_kwd = pd.Series(get_subtopic_descriptions(self.df)).reset_index()
        df_kwd = df_kwd.rename(columns={'index': 'comp', 0: 'kwd'})
        self.df = pd.merge(self.df, df_kwd, on='comp')
        logging.info('Done\n')

        # TODO: Fix components reordering

    #        KEY = 'citations' # 'size' or 'citations'
    #
    #        start = int(components_merged) # Sort from component #1 if merged, Other should be the lowest priority
    #        if KEY == 'size':
    #            order = df_all.groupby('comp')['pmid'].count().sort_values(ascending=False).index.values
    #        elif KEY == 'citations':
    #            order = df_all.groupby('comp')['total'].sum().sort_values(ascending=False).index.values
    #        df_all['comp'] = df_all['comp'].map(dict(enumerate(order)))

    def subtopic_evolution_analysis(self, step=2):
        min_year = self.cocit_df['year'].min().astype(int)
        max_year = self.cocit_df['year'].max().astype(int)
        logging.info(f'Studying evolution of subtopic clusters in {min_year} - {max_year} with step of {step} years')

        evolution_series = []
        year_range = range(max_year, min_year - 1, -step)
        logging.info('Filtering top 10000 or 80% of all the co-citations')
        for year in year_range:
            cocit_grouped_df = self.cocit_df[self.cocit_df['year'] <= year].groupby(
                ['cited_1', 'cited_2', 'year']).count().reset_index()
            cocit_grouped_df = cocit_grouped_df.pivot_table(index=['cited_1', 'cited_2'],
                                                            columns=['year'], values=['citing']).reset_index()
            cocit_grouped_df = cocit_grouped_df.replace(np.nan, 0)
            cocit_grouped_df['total'] = cocit_grouped_df.iloc[:, 2:].sum(axis=1)
            cocit_grouped_df = cocit_grouped_df.sort_values(by='total', ascending=False)
            cocit_grouped_df = cocit_grouped_df.iloc[:min(10000, round(0.8 * len(cocit_grouped_df))), :]

            CG = nx.Graph()
            # NOTE: we use nodes id as String to avoid problems str keys in jsonify during graph visualization
            for el in cocit_grouped_df[['cited_1', 'cited_2', 'total']].values.astype(int):
                CG.add_edge(str(el[0]), str(el[1]), weight=el[2])
            logging.info(f'{year}: graph contains {len(CG.nodes)} nodes, {len(CG.edges)} edges')

            p = {int(vertex): int(comp) for vertex, comp in community.best_partition(CG).items()}
            evolution_series.append(pd.Series(p))

        SHIFT = True  # use random shift to see trace of separate articles
        FILLNA = True  # NaN values sometimes cause KeyError while plotting, but sometimes not (?!)

        self.evolution_df = pd.concat(evolution_series, axis=1).rename(columns=dict(enumerate(year_range)))
        self.evolution_df['current'] = self.evolution_df[max_year]
        self.evolution_df = self.evolution_df[list(reversed(list(self.evolution_df.columns)))]

        if SHIFT:
            shift = np.random.uniform(0.25, 0.75, size=(len(self.evolution_df),))
            for year in year_range:
                self.evolution_df[year] += shift

        if FILLNA:
            self.evolution_df = self.evolution_df.fillna(-1.0)
