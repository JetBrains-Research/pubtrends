import networkx as nx
import unittest

from pysrc.papers.analysis.node2vec import _precompute, _random_walks, node2vec, NODE2VEC_P, NODE2VEC_Q


class TestNode2Vec(unittest.TestCase):

    def test_precompute_nodes_graph(self):
        graph = nx.Graph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_node(3)
        ps1, ps2 = _precompute(graph, 'weight', NODE2VEC_P, NODE2VEC_Q)
        self.assertEqual({1: [], 2: [], 3: []}, {k: v.tolist() for k, v in ps1.items()})
        self.assertEqual([[1], [2], [3], [1], [2], [3]], _random_walks(graph, ps1, ps2, 2, 10))

    def test_walk_triangle(self):
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(2, 3, weight=1)
        graph.add_edge(3, 1, weight=1)
        graph.add_node(4)
        ps1, ps2 = _precompute(graph, 'weight', NODE2VEC_P, NODE2VEC_Q)
        walks = _random_walks(graph, ps1, ps2, walks_per_node=1, walk_length=3, seed=42)
        self.assertEqual([[1, 2, 3], [2, 3, 1], [3, 2, 1], [4]], walks)

    def test_walk_triangle_weighted(self):
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=100)
        graph.add_edge(2, 3, weight=10)
        graph.add_edge(3, 1, weight=1)
        ps1, ps2 = _precompute(graph, 'weight', NODE2VEC_P, NODE2VEC_Q)
        walks = _random_walks(graph, ps1, ps2, walks_per_node=5, walk_length=3, seed=42)
        self.assertEqual([[1, 2, 3],
                          [2, 1, 2],
                          [3, 2, 1],
                          [1, 2, 3],
                          [2, 1, 2],
                          [3, 2, 1],
                          [1, 2, 1],
                          [2, 1, 2],
                          [3, 2, 1],
                          [1, 2, 1],
                          [2, 1, 2],
                          [3, 2, 1],
                          [1, 2, 3],
                          [2, 1, 2],
                          [3, 2, 1]],
                         walks)

    def test_node2vec_triangle_weighted(self):
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=100)
        graph.add_edge(2, 3, weight=10)
        graph.add_edge(3, 1, weight=1)
        vec = node2vec([1, 2, 3], graph, walk_length=3, walks_per_node=5, vector_size=8, seed=42)
        self.assertEqual((3, 8), vec.shape)

    def test_node2vec_weighted(self):
        graph = nx.Graph()
        for i in range(1, 11):
            j = 2
            while i * j < 11:
                graph.add_edge(i, i * j, weight=i + j)
                j += 1
            graph.add_edge(0, i, weight=1)
        vec = node2vec(range(11), graph, walk_length=3, walks_per_node=5, vector_size=128, seed=42)
        self.assertEqual((11, 128), vec.shape)


if __name__ == '__main__':
    unittest.main()
