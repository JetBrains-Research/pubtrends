import unittest

import networkx as nx

from pysrc.papers.analysis.graph import _local_sparse
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.test.mock_loaders import MockLoader, PAPERS_GRAPH_EDGES, MockLoaderSingle


class TestBuildGraph(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        loader = MockLoader()
        cls.analyzer = PapersAnalyzer(loader, TestBuildGraph.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.TOPIC_MIN_SIZE = 0  # Disable merging for tests
        cls.analyzer.analyze_papers(ids, 'query')

    def test_build_similarity_graph_edges(self):
        edges = list(self.analyzer.papers_graph.edges(data=True))
        # print(edges)
        self.assertEquals(len(PAPERS_GRAPH_EDGES), len(edges), msg='size')
        for expected, actual in zip(PAPERS_GRAPH_EDGES, edges):
            self.assertEquals(expected[0], actual[0], msg='source')
            self.assertEquals(expected[1], actual[1], msg='target')
            for m in 'cocitation', 'citations', 'bibcoupling':
                self.assertEquals(m in expected[2], m in actual[2], msg=f'{m} presence')
                if m in expected[2]:
                    self.assertAlmostEqual(expected[2][m], actual[2][m], msg=f'{m} value', delta=1e-3)

    def test_local_sparse(self):
        # Full graph on 4 nodes
        graph = nx.DiGraph()
        for i in range(5):
            for j in range(i + 1, 5):
                graph.add_edge(i, j, similarity=1)

        self.assertEqual([(0, 3), (0, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2)],
                         list(_local_sparse(graph, 0.1).edges))
        self.assertEqual([(0, 3), (0, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2)],
                         list(_local_sparse(graph, 0.5).edges))
        self.assertEqual([(0, 1), (0, 2), (0, 3), (0, 4),
                          (1, 2), (1, 3), (1, 4),
                          (2, 3), (2, 4),
                          (3, 4)],
                         list(_local_sparse(graph, 1).edges))

        # Add not connected edge
        graph.add_edge(10, 11, similarity=10)
        self.assertEqual([(0, 4), (4, 1), (4, 2), (4, 3), (10, 11)], list(_local_sparse(graph, 0).edges))

    def test_isolated_edges_sparse(self):
        # Full graph on 4 nodes
        graph = nx.DiGraph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_node(3)

        self.assertEqual([1, 2, 3], list(_local_sparse(graph, 0).nodes))


class TestBuildGraphSingle(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = PapersAnalyzer(MockLoaderSingle(), TestBuildGraphSingle.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.analyze_papers(ids, 'query')
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.df['id'])


if __name__ == "__main__":
    unittest.main()
