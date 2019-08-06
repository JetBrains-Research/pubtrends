import unittest

import networkx as nx

from models.keypaper.analysis import KeyPaperAnalyzer
from models.test.articles import expected_graph, cocitations_df, expected_cocit_and_cit_graph
from models.test.test_loader import TestLoader


class TestBuildGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.analyzer = KeyPaperAnalyzer(TestLoader())
        cls.analyzer.G = expected_graph
        cls.CG = cls.analyzer.build_cocitation_graph(cocitations_df, add_citation_edges=True)

    def test_cocitation_nodes_amount(self):
        expected_number_of_nodes = len(expected_cocit_and_cit_graph.nodes())
        actual_number_of_nodes = len(self.CG.nodes())
        self.assertEqual(expected_number_of_nodes, actual_number_of_nodes,
                         "Amount of nodes in citation graph is incorrect")

    def test_cocitations_nodes(self):
        expected_nodes = expected_cocit_and_cit_graph.nodes()
        actual_nodes = self.CG.nodes()
        self.assertEqual(expected_nodes, actual_nodes, "Nodes in citation graph are incorrect")

    def test_load_citations(self):
        self.assertTrue(nx.is_isomorphic(expected_cocit_and_cit_graph, self.CG), "Graph of cocitations is incorrect")


if __name__ == "__main__":
    unittest.main()
