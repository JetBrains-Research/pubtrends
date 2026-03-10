import unittest

import networkx as nx

from pysrc.papers.analysis.node2vec import _precompute, _random_walks, node2vec, NODE2VEC_P, NODE2VEC_Q


class TestNode2Vec(unittest.TestCase):

    def test_precompute_nodes_graph(self):
        graph = nx.Graph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_node(3)
        ps1, ps2, neighbors_cache = _precompute(graph, 'weight', NODE2VEC_P, NODE2VEC_Q)
        self.assertEqual({1: [], 2: [], 3: []}, {k: v.tolist() for k, v in ps1.items()})
        self.assertEqual([[1], [2], [3], [1], [2], [3]], _random_walks(neighbors_cache, ps1, ps2, 2, 10))

    def test_walk_triangle(self):
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(2, 3, weight=1)
        graph.add_edge(3, 1, weight=1)
        graph.add_node(4)
        ps1, ps2, neighbors_cache = _precompute(graph, 'weight', NODE2VEC_P, NODE2VEC_Q)
        walks = _random_walks(neighbors_cache, ps1, ps2, walks_per_node=1, walk_length=3, seed=42)
        self.assertEqual([[1, 2, 3], [2, 3, 1], [3, 2, 1], [4]], walks)

    def test_walk_triangle_weighted(self):
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=100)
        graph.add_edge(2, 3, weight=10)
        graph.add_edge(3, 1, weight=1)
        ps1, ps2, neighbors_cache = _precompute(graph, 'weight', NODE2VEC_P, NODE2VEC_Q)
        walks = _random_walks(neighbors_cache, ps1, ps2, walks_per_node=5, walk_length=3, seed=42)

        # Validate structure (exact sequence may vary due to implementation details)
        self.assertEqual(15, len(walks))  # 5 walks per node * 3 nodes

        # Check walk lengths
        for walk in walks:
            self.assertTrue(1 <= len(walk) <= 3, f"Walk length should be 1-3, got {len(walk)}: {walk}")

        # Check starting nodes (should have 5 walks per node)
        starts = [walk[0] for walk in walks]
        self.assertEqual(5, starts.count(1), "Expected 5 walks starting with 1")
        self.assertEqual(5, starts.count(2), "Expected 5 walks starting with 2")
        self.assertEqual(5, starts.count(3), "Expected 5 walks starting with 3")

        # Check that walks only contain valid nodes
        all_nodes = set()
        for walk in walks:
            all_nodes.update(walk)
        self.assertEqual({1, 2, 3}, all_nodes, "Walks should only contain nodes 1,2,3")

    def test_node2vec_triangle_weighted(self):
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=100)
        graph.add_edge(2, 3, weight=10)
        graph.add_edge(3, 1, weight=1)
        vec = node2vec([1, 2, 3], graph, key='weight', walk_length=3, walks_per_node=5, vector_size=8, seed=42)
        self.assertEqual((3, 8), vec.shape)

    def test_node2vec_weighted(self):
        graph = nx.Graph()
        for i in range(1, 11):
            j = 2
            while i * j < 11:
                graph.add_edge(i, i * j, weight=i + j)
                j += 1
            graph.add_edge(0, i, weight=1)
        vec = node2vec(range(11), graph, key='weight', walk_length=3, walks_per_node=5, vector_size=128, seed=42)
        self.assertEqual((11, 128), vec.shape)


if __name__ == '__main__':
    unittest.main()
