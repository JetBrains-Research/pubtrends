import unittest

import networkx as nx

from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.test.mock_loaders import MockLoader
from models.test.ss_database_articles import expected_graph, cocitations_df, expected_cocit_and_cit_graph, \
    bibliographic_coupling_df


class TestBuildGraph(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        loader = MockLoader()
        cls.analyzer = KeyPaperAnalyzer(loader, TestBuildGraph.PUBTRENDS_CONFIG, test=True)
        cls.analyzer.citations_graph = expected_graph
        # Turn off bibliographic coupling for tests purposes
        cls.analyzer.RELATIONS_GRAPH_BIBLIOGRAPHIC_COUPLING = 0
        cls.analyzer.RELATIONS_GRAPH_COCITATION = 1
        cls.analyzer.RELATIONS_GRAPH_CITATION = 0.01
        cls.analyzer.paper_relations_graph = cls.analyzer.build_papers_relation_graph(
            expected_graph, cocitations_df, bibliographic_coupling_df
        )

    def test_cocitations_graph_nodes(self):
        expected_nodes = expected_cocit_and_cit_graph.nodes()
        actual_nodes = self.analyzer.paper_relations_graph.nodes()
        self.assertEqual(expected_nodes, actual_nodes,
                         "Nodes in co-citation graph are incorrect")

    def test_build_cocitation_graph_graph(self):
        print(self.analyzer.paper_relations_graph)
        self.assertTrue(nx.is_isomorphic(expected_cocit_and_cit_graph, self.analyzer.paper_relations_graph),
                        "Graph of co-citation is incorrect")


if __name__ == "__main__":
    unittest.main()
