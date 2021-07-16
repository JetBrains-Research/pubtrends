import unittest

import networkx as nx
import numpy as np
from parameterized import parameterized

from pysrc.papers.analysis.graph import build_citation_graph
from pysrc.papers.analysis.citations import find_top_cited_papers
from pysrc.papers.analysis.node2vec import _precompute, _random_walks
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.test.mock_loaders import MockLoader, \
    EXPECTED_MAX_GAIN, EXPECTED_MAX_RELATIVE_GAIN, MockLoaderSingle


class TestNode2Vec(unittest.TestCase):

    def test_precompute_nodes_graph(self):
        graph = nx.Graph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_node(3)
        am, psfs, ps = _precompute(graph, weight_key='none')
        self.assertEqual({1: [], 2: [], 3: []}, am)
        self.assertEqual({1: [], 2: [], 3: []}, {k: v.tolist() for k, v in psfs.items()})
        self.assertEqual([[1], [2], [3], [1], [2], [3]], _random_walks(list(graph.nodes), am, psfs, ps, 2, 10))

    def test_walk_triangle(self):
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(2, 3, weight=1)
        graph.add_edge(3, 1, weight=1)
        graph.add_node(4)
        am, psfs, ps = _precompute(graph, weight_key='weight')
        walks = _random_walks(list(graph.nodes), am, psfs, ps, walks_per_node=1, walk_length=3)
        self.assertEqual([[1, 2, 3], [2, 3, 2], [3, 2, 1], [4]], walks)

    def test_walk_triangle_weighted(self):
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=100)
        graph.add_edge(2, 3, weight=10)
        graph.add_edge(3, 1, weight=1)
        am, psfs, ps = _precompute(graph, weight_key='weight')
        walks = _random_walks(list(graph.nodes), am, psfs, ps, walks_per_node=5, walk_length=3)
        self.assertEqual([[1, 2, 1], [2, 1, 2], [3, 2, 1],
                          [1, 2, 1], [2, 1, 2], [3, 2, 3],
                          [1, 2, 1], [2, 1, 2], [3, 2, 1],
                          [1, 2, 1], [2, 1, 2], [3, 2, 1],
                          [1, 2, 1], [2, 1, 2], [3, 2, 1]], walks)


if __name__ == '__main__':
    unittest.main()
