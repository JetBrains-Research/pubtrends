import json

import networkx as nx
import pandas as pd

RELATIONS_GRAPH_GRAPHML = 'pysrc/test/test_data/similarity_graph.graphml'
DF_CSV = 'pysrc/test/test_data/df.csv'
EVOLUTION_DF_CSV = 'pysrc/test/test_data/evolution_df.csv'
EVOLUTION_KEYWORDS_JSON = 'pysrc/test/test_data/evolution_keywords.json'


class MockAnalyzer:
    def __init__(self):
        # Load DataFrame and convert id to str
        self.df = pd.read_csv(DF_CSV)
        self.df['id'] = self.df['id'].astype(str)
        self.pub_types = ['Article', 'Review']

        self.min_year = 2005
        self.max_year = 2019

        # Load co-citation graph and convert nodes to str
        self.similarity_graph = nx.read_graphml(RELATIONS_GRAPH_GRAPHML)
        mapping = {node: str(node) for node in self.similarity_graph.nodes()}
        self.similarity_graph = nx.relabel_nodes(self.similarity_graph, mapping, copy=False)

        # Components are already in df
        self.components = [0, 1, 2]
        self.partition = pd.Series(self.df['comp']).set_axis(self.df['id']).to_dict()
        self.comp_sizes = {0: 6, 1: 4, 2: 1}

        # Load evolution DataFrame and keywords
        self.evolution_df = pd.read_csv(EVOLUTION_DF_CSV)
        self.n_steps = 2

        with open(EVOLUTION_KEYWORDS_JSON, 'r') as f:
            self.evolution_kwds = json.load(f)

        # Convert years and comp keys to int
        for year in list(self.evolution_kwds.keys()):
            self.evolution_kwds[int(year)] = self.evolution_kwds.pop(year)
        for year, evolution_year in self.evolution_kwds.items():
            for c in list(evolution_year.keys()):
                evolution_year[int(c)] = evolution_year.pop(c)
