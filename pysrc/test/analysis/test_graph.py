import networkx as nx
import unittest

from pysrc.papers.analysis.graph import sparse_graph
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.utils import SORT_MOST_CITED
from pysrc.test.mock_loaders import MockLoader, PAPERS_GRAPH_EDGES, MockLoaderSingle

PUBTRENDS_CONFIG = PubtrendsConfig(test=True)


class TestBuildGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        analyzer = PapersAnalyzer(MockLoader(), PUBTRENDS_CONFIG, test=True)
        ids = analyzer.search_terms(query='query')
        analyzer.analyze_papers(
            ids, 'query', 'Pubmed', SORT_MOST_CITED, 10, PUBTRENDS_CONFIG.show_topics_default_value, test=True
        )
        cls.data = analyzer.save()

    def test_build_papers_graph(self):
        edges = list(self.data.papers_graph.edges(data=True))
        edges.sort(key=lambda x: (x[0], x[1]))
        self.assertEqual(len(PAPERS_GRAPH_EDGES), len(edges), msg='size')
        for expected, actual in zip(PAPERS_GRAPH_EDGES, edges):
            self.assertEqual(expected[0], actual[0], msg='source')
            self.assertEqual(expected[1], actual[1], msg='target')
            for m in 'cocitation', 'citations', 'bibcoupling':
                self.assertEqual(m in expected[2], m in actual[2], msg=f'{m} presence')
                if m in expected[2]:
                    self.assertAlmostEqual(expected[2][m], actual[2][m], msg=f'{m} value', delta=1e-3)


class TestBuildGraphSingle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.analyzer = PapersAnalyzer(MockLoaderSingle(), PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.analyze_papers(ids, 'query', PUBTRENDS_CONFIG.show_topics_default_value, test=True)
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.df['id'])


class TestSparseGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        g = nx.Graph()
        for i in range(10):
            for j in range(10):
                if i >= j:
                    continue
                g.add_edge(i, j, weight=i + j)
        assert len(g.edges) == 45
        cls.g = g

    SPARSE = [(0, 9, {'weight': 9}), (0, 8, {'weight': 8}), (0, 7, {'weight': 7}), (9, 1, {'weight': 10}),
              (9, 2, {'weight': 11}), (9, 6, {'weight': 15}), (8, 1, {'weight': 9}), (8, 2, {'weight': 10}),
              (8, 6, {'weight': 14}), (7, 1, {'weight': 8}), (7, 2, {'weight': 9}), (7, 6, {'weight': 13})]

    def test_sparse_graph(self):
        sg = sparse_graph(self.g, 3, 'weight', False)
        edges = list(sg.edges(data=True))
        print(edges)
        self.assertEqual(len(sg.edges), 12)
        self.assertEqual(self.SPARSE, edges)


if __name__ == "__main__":
    unittest.main()
