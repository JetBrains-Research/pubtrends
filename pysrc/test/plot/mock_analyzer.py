import networkx as nx
import pandas as pd

from pysrc.config import VECTOR_WORDS, VECTOR_MIN_DF, VECTOR_MAX_DF
from pysrc.papers.analysis.text import vectorize_corpus

RELATIONS_GRAPH_GRAPHML = 'pysrc/test/test_data/papers_graph.graphml'
DF_CSV = 'pysrc/test/test_data/df.csv'


class MockAnalyzer:
    def __init__(self):
        # Load DataFrame and convert id to str
        self.df = pd.read_csv(DF_CSV)
        self.df['id'] = self.df['id'].astype(str)
        self.df['mesh'] = ''
        self.df['keywords'] = ''

        self.corpus, self.corpus_tokens, self.corpus_counts = vectorize_corpus(
            self.df,
            max_features=VECTOR_WORDS,
            min_df=VECTOR_MIN_DF,
            max_df=VECTOR_MAX_DF,
            test=True
        )

        self.min_year = 2005
        self.max_year = 2019

        # Load co-citation graph and convert nodes to str
        self.papers_graph = nx.read_graphml(RELATIONS_GRAPH_GRAPHML)
        mapping = {node: str(node) for node in self.papers_graph.nodes()}
        self.papers_graph = nx.relabel_nodes(self.papers_graph, mapping, copy=False)

        # Components are already in df
        self.components = [0, 1, 2]
        self.partition = pd.Series(self.df['comp']).set_axis(self.df['id']).to_dict()
        self.comp_sizes = {0: 6, 1: 4, 2: 1}
