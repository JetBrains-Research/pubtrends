import json

import networkx as nx
import pandas as pd

CG_GRAPHML = 'models/test/test_data/cg.graphml'
DF_CSV = 'models/test/test_data/df.csv'
EVOLUTION_DF_CSV = 'models/test/test_data/evolution_df.csv'
EVOLUTION_KEYWORDS_JSON = 'models/test/test_data/evolution_keywords.json'


class MockAnalyzer:
    def __init__(self):
        # Load DataFrame and convert id to str
        self.df = pd.read_csv(DF_CSV)
        self.df['id'] = self.df['id'].astype(str)
        self.pub_types = ['Article', 'Review']

        self.min_year = 2005
        self.max_year = 2019

        # Load co-citation graph and convert nodes to str
        self.CG = nx.read_graphml(CG_GRAPHML)
        mapping = {node: str(node) for node in self.CG.nodes()}
        self.CG = nx.relabel_nodes(self.CG, mapping, copy=False)

        self.components = [0, 1, 2]
        self.comp_other = 2
        self.pm = {'15790588': 1, '18483325': 0, '19296127': 1, '22080730': 1, '24138928': 0, '27346616': 1,
                   '28259012': 0, '28364215': 0, '28423572': 0, '29235933': 0, '31179760': 2}
        self.pmcomp_sizes = {0: 6, 1: 4, 2: 1}

        # Load evolution DataFrame and keywords
        self.evolution_df = pd.read_csv(EVOLUTION_DF_CSV)
        self.n_steps = 2

        with open(EVOLUTION_KEYWORDS_JSON, 'r') as f:
            self.evolution_kwds = json.load(f)

        # Convert years and comp keys to int
        for year in self.evolution_kwds.keys():
            self.evolution_kwds[int(year)] = self.evolution_kwds.pop(year)
        for year in self.evolution_kwds.keys():
            for c in self.evolution_kwds[year].keys():
                self.evolution_kwds[year][int(c)] = self.evolution_kwds[year].pop(c)
