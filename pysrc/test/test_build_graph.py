import unittest

import networkx as nx
import pandas as pd

from pysrc.papers.analyzer import KeyPaperAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.test.db.ss_test_articles import citations_graph, cocitations_df, bibliographic_coupling_df, \
    expected_cocit_and_cit_graph
from pysrc.test.mock_loaders import MockLoader


class TestBuildGraph(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        loader = MockLoader()
        cls.analyzer = KeyPaperAnalyzer(loader, TestBuildGraph.PUBTRENDS_CONFIG, test=True)
        cls.analyzer.citations_graph = citations_graph
        # Turn off bibliographic coupling for tests purposes
        cls.analyzer.SIMILARITY_BIBLIOGRAPHIC_COUPLING = 0
        cls.analyzer.SIMILARITY_COCITATION = 1
        cls.analyzer.SIMILARITY_CITATION = 0.01
        cls.analyzer.similarity_graph = cls.analyzer.build_similarity_graph(
            pd.DataFrame(columns=['id', 'title', 'abstract']),
            citations_graph, cocitations_df, bibliographic_coupling_df
        )

    def test_cocitations_graph_nodes(self):
        expected_nodes = expected_cocit_and_cit_graph.nodes()
        actual_nodes = self.analyzer.similarity_graph.nodes()
        self.assertEqual(expected_nodes, actual_nodes,
                         "Nodes in co-citation graph are incorrect")

    def test_build_cocitation_graph_graph(self):
        print(self.analyzer.similarity_graph)
        self.assertTrue(nx.is_isomorphic(expected_cocit_and_cit_graph, self.analyzer.similarity_graph),
                        "Graph of co-citation is incorrect")


if __name__ == "__main__":
    unittest.main()
