import unittest

import networkx as nx

from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.test.mock_loaders import MockLoader
from models.test.ss_database_articles import expected_graph, cocitations_df, expected_cocit_and_cit_graph


class TestBuildGraph(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = KeyPaperAnalyzer(MockLoader(), TestBuildGraph.PUBTRENDS_CONFIG, test=True)
        cls.analyzer.G = expected_graph
        cls.CG = cls.analyzer.build_cocitation_graph(cocitations_df, add_citation_edges=True)

    def test_cocitation_nodes_amount(self):
        expected_number_of_nodes = len(expected_cocit_and_cit_graph.nodes())
        actual_number_of_nodes = len(self.CG.nodes())
        self.assertEqual(expected_number_of_nodes, actual_number_of_nodes,
                         "Amount of nodes in co-citation graph is incorrect")

    def test_cocitations_nodes(self):
        expected_nodes = expected_cocit_and_cit_graph.nodes()
        actual_nodes = self.CG.nodes()
        self.assertEqual(expected_nodes, actual_nodes, "Nodes in co-citation graph are incorrect")

    def test_build_cocitations_graph(self):
        self.assertTrue(nx.is_isomorphic(expected_cocit_and_cit_graph, self.CG), "Graph of co-citations is incorrect")


if __name__ == "__main__":
    unittest.main()
