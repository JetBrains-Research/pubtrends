import unittest

import networkx as nx
import pandas as pd

from pysrc.papers.analysis.graph import build_similarity_graph, build_citation_graph, local_sparse
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.test.mock_loaders import MockLoader, CITATION_GRAPH_NODES, CITATION_GRAPH_EDGES, SIMILARITY_GRAPH, \
    MockLoaderSingle


class TestBuildGraph(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        loader = MockLoader()
        cls.analyzer = PapersAnalyzer(loader, TestBuildGraph.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.TOPIC_MIN_SIZE = 0  # Disable merging for tests
        cls.analyzer.analyze_papers(ids, 'query')
        # cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.ids)
        # cls.analyzer.citations_graph = build_citation_graph(cls.analyzer.cit_df)
        # cls.analyzer.bibliographic_coupling_df = loader.load_bibliographic_coupling(cls.analyzer.ids)
        # cls.analyzer.similarity_graph = build_similarity_graph(
        #     cls.analyzer.df, cls.analyzer.texts_similarity, cls.analyzer.citations_graph, cls.analyzer.cocit_grouped_df,
        #     cls.analyzer.bibliographic_coupling_df
        # )

    def test_build_citation_graph_nodes_count(self):
        self.assertEqual(self.analyzer.citations_graph.number_of_nodes(), len(CITATION_GRAPH_NODES))

    def test_build_citation_graph_edges_count(self):
        self.assertEqual(self.analyzer.citations_graph.number_of_edges(), len(CITATION_GRAPH_EDGES))

    def test_build_citation_graph_nodes(self):
        self.assertCountEqual(list(self.analyzer.citations_graph.nodes()), CITATION_GRAPH_NODES)

    def test_build_citation_graph_edges(self):
        self.assertCountEqual(list(self.analyzer.citations_graph.edges()), CITATION_GRAPH_EDGES)

    def test_build_similarity_graph_edges(self):
        edges = list(self.analyzer.similarity_graph.edges(data=True))
        self.assertCountEqual(edges, SIMILARITY_GRAPH)

    def test_local_sparse(self):
        # Full graph on 4 nodes
        graph = nx.DiGraph()
        for i in range(5):
            for j in range(i + 1, 5):
                graph.add_edge(i, j, similarity=1)

        empty = local_sparse(graph, 0)
        self.assertEquals(4, len(empty.edges))
        self.assertEquals([(0, 4), (4, 1), (4, 2), (4, 3)], list(empty.edges))
        self.assertEquals([(0, 3), (0, 4), (3, 4), (4, 1), (4, 2)], list(local_sparse(graph, 0.5).edges))
        self.assertEquals([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
                          list(local_sparse(graph, 1).edges))

        # Add not connected edge
        graph.add_edge(10, 11, similarity=10)
        self.assertEquals([(0, 4), (4, 1), (4, 2), (4, 3), (10, 11)], list(local_sparse(graph, 0).edges))
        self.assertEquals([(0, 3), (0, 4), (3, 4), (4, 1), (4, 2), (10, 11)], list(local_sparse(graph, 0.5).edges))


class TestBuildGraphSingle(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = PapersAnalyzer(MockLoaderSingle(), TestBuildGraphSingle.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.analyze_papers(ids, 'query')
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.ids)
        cls.analyzer.citations_graph = build_citation_graph(cls.analyzer.cit_df)

    def test_build_citation_graph_nodes_count(self):
        self.assertEqual(self.analyzer.citations_graph.number_of_nodes(), 0)

    def test_build_citation_graph_edges_count(self):
        self.assertEqual(self.analyzer.citations_graph.number_of_edges(), 0)

    def test_build_citation_graph_nodes(self):
        self.assertCountEqual(list(self.analyzer.citations_graph.nodes()), [])

    def test_build_citation_graph_edges(self):
        self.assertCountEqual(list(self.analyzer.citations_graph.edges()), [])


if __name__ == "__main__":
    unittest.main()
