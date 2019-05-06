import community
import logging
import networkx as nx
import numpy as np
import pandas as pd
import psycopg2 as pg_driver
import re

from Bio import Entrez

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)

class KeyPaperAnalyzer:
    def __init__(self, email):
        Entrez.email = email
        self.conn = pg_driver.connect(dbname='pubmed', user='biolabs',
                                      password='pubtrends', host='localhost')
        self.cursor = self.conn.cursor()
        
    def search(self, *terms):
        print('TODO: handle queries which return more than 1000000 items')
        print('TODO: use local database instead of PubMed API')
        self.terms = [t.lower() for t in terms]
        query=' '.join(terms)
        handle = Entrez.esearch(db='pubmed', retmax='1000000', retmode='xml', term=query)
        self.pmids = [int(pmid) for pmid in Entrez.read(handle)['IdList']]
        logging.info(f'Found {len(self.pmids)} articles about {terms}')
        

    def load_publications(self):
        logging.info('Loading publication data')

        query = '''
        SELECT pmid, title, year
        FROM Publications
        WHERE pmid = ANY(%s);
        '''

        with self.conn:
            self.cursor.execute(query, (self.pmids,))
        pub_data = []
        for row in self.cursor:
            pub_data.append(list(row))
        self.pub_df = pd.DataFrame(pub_data, columns=['pmid', 'title', 'year'])
        logging.info(f'Found {len(self.pub_df)} publications in the local database')

    def load_cocitations(self):
        logging.info('Calculating co-citations for selected articles')

        # Optimize WHERE with JOIN (VALUES ... ) AS
        # instead of WHERE C1.pmid_cited = ANY(%s) AND C2.pmid_cited = ANY(%s)
        # See https://pgday.ru/files/pgmaster14/max.boguk.query.optimization.pdf
        values = ', '.join(['({})'.format(i) for i in sorted(self.pmids)])
        query = re.sub('\$VALUES\$', values, '''
        SELECT C1.pmid_citing, C1.pmid_cited, C2.pmid_cited, P.year
        FROM Citations C1
        JOIN (VALUES $VALUES$) AS C1T(pmid_cited) ON (C1.pmid_cited = C1T.pmid_cited)
        JOIN Citations C2
        JOIN (VALUES $VALUES$) AS C2T(pmid_cited) ON (C2.pmid_cited = C2T.pmid_cited)
        ON C1.pmid_citing = C2.pmid_citing AND C1.pmid_cited < C2.pmid_cited
        JOIN Publications P
        ON C1.pmid_citing = P.pmid;
        ''')

        with self.conn:
            self.cursor.execute(query)

        cocit_data = []
        for row in self.cursor:
            cocit_data.append(list(row))
        self.cocit_df = pd.DataFrame(cocit_data, columns=['citing', 'cited_1', 'cited_2', 'year'])
        logging.info(f'Found {len(self.cocit_df)} co-cited pairs of articles')

        logging.info(f'Building co-citations graph')
        self.cocit_grouped_df = self.cocit_df.groupby(['cited_1', 'cited_2', 'year']).count().reset_index()
        self.cocit_grouped_df = self.cocit_grouped_df.pivot_table(index=['cited_1', 'cited_2'],
                                                          columns=['year'], values=['citing']).reset_index()
        self.cocit_grouped_df = self.cocit_grouped_df.replace(np.nan, 0)
        self.cocit_grouped_df['total'] = self.cocit_grouped_df.iloc[:, 2:].sum(axis=1)
        self.cocit_grouped_df = self.cocit_grouped_df.sort_values(by='total', ascending=False)

        self.CG = nx.Graph()
        # NOTE: we use nodes id as String to avoid problems str keys in jsonify during graph visualization
        for el in self.cocit_grouped_df[['cited_1', 'cited_2', 'total']].values.astype(int):
            self.CG.add_edge(str(el[0]), str(el[1]), weight=el[2])
        logging.info(f'Co-citations graph nodes {len(self.CG.nodes())} edges {len(self.CG.edges())}')

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

        pub_data = []
        for row in self.cursor:
            pub_data.append(list(row))
        self.cit_df = pd.DataFrame(pub_data, columns=['pmid', 'year', 'count'])

        self.cit_df = self.cit_df.pivot(index='pmid', columns='year', values='count').reset_index().replace(np.nan, 0)
        self.cit_df['total'] = self.cit_df.iloc[:, 1:].sum(axis = 1)
        self.cit_df = self.cit_df.sort_values(by='total', ascending=False)

        logging.info('Filtering top 1000 or 50% of all the papers')
        self.cit_df = self.cit_df.iloc[:min(1000, round(0.5 * len(self.cit_df))), :]
        logging.info('Done aggregation')

        logging.info(f"Loaded citation stats for {len(self.cit_df)} of {len(self.pmids)} articles. " +
                    "Others may either have zero citations or be absent in the local database.")
        
    def subtopic_analysis(self):
        logging.info(f'Louvain community clustering of co-citation graph')
        p = community.best_partition(self.CG)
        components = set(p.values())
        logging.info(f'Found {len(components)} components')
        
        # This will limit total number of components
        GRANULARITY = 0.01 
        logging.info(f'Merging components smaller than {GRANULARITY} to "Other" component')
        threshold = int(GRANULARITY * len(p))
        comp_sizes = {com: sum([p[node] == com for node in p.keys()]) for com in components}
        comp_to_merge = {com: comp_sizes[com] <= threshold for com in components}
        if sum(comp_to_merge.values()) > 0:
            logging.info(f'Reassigning components')
            pm = {}
            newcomps = {}
            ci = 1 # Other component is 0.
            for k, v in p.items():
                if comp_sizes[v] <= threshold:
                    pm[k] = 0 # Other
                    continue
                if v not in newcomps:
                    newcomps[v] = ci
                    ci += 1
                pm[k] = newcomps[v]
            logging.info(f'Processed {len(set(pm.values()))} components')
        else:
            pm = p
        components = set(pm.values())    
        pmcomp_sizes = {com: sum([pm[node] == com for node in pm.keys()]) for com in components}
        logging.info('\n'.join([f'{k}: {v} ({int(100 * v / len(pm))}%)' for k,v in pmcomp_sizes.items()]))
        
        pm_ints = {int(k): v for k,v in pm.items()}
        df_comp = pd.Series(pm_ints).reset_index().rename(columns={'index': 'pmid', 0: 'comp'})
        self.pubcit_df = pd.merge(self.pub_df, self.cit_df, on='pmid')
        self.pubcit_df = pd.merge(self.pubcit_df, df_comp, on='pmid')