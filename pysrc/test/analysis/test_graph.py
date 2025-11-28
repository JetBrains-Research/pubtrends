import unittest

import networkx as nx

from pysrc.config import PubtrendsConfig
from pysrc.papers.analysis.graph import sparse_graph
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.utils import SORT_MOST_CITED, IDS_ANALYSIS_TYPE
from pysrc.test.mock_loaders import MockLoader, PAPERS_GRAPH_EDGES, MockLoaderSingle

PUBTRENDS_CONFIG = PubtrendsConfig(test=True)


class TestBuildGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        analyzer = PapersAnalyzer(MockLoader(), PUBTRENDS_CONFIG, test=True)
        ids = analyzer.search_terms(query='query')
        analyzer.analyze_papers(ids, PUBTRENDS_CONFIG.show_topics_default_value, test=True)
        cls.data = analyzer.save(IDS_ANALYSIS_TYPE, None, 'query', 'Pubmed', SORT_MOST_CITED, 10, False, None, None)

    def test_build_papers_graph(self):
        edges = list(self.data.papers_graph.edges(data=True))
        edges.sort(key=lambda x: (x[0], x[1]))
        print(edges)
        self.assertEqual(len(PAPERS_GRAPH_EDGES), len(edges), msg='size')
        expected = [(min(a, b), max(a, b)) for a, b in PAPERS_GRAPH_EDGES]
        expected.sort(key=lambda x: (x[0], x[1]))
        actual = [(min(a, b), max(a, b)) for a, b, _ in edges]
        actual.sort(key=lambda x: (x[0], x[1]))
        assert expected == actual


class TestBuildGraphSingle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.analyzer = PapersAnalyzer(MockLoaderSingle(), PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.analyze_papers(ids, PUBTRENDS_CONFIG.show_topics_default_value, test=True)
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

    SPARSE = [(9, 8, {'weight': 17}), (9, 7, {'weight': 16}), (9, 6, {'weight': 15}), (9, 0, {'weight': 9}),
              (9, 1, {'weight': 10}), (9, 2, {'weight': 11}), (9, 3, {'weight': 12}), (9, 4, {'weight': 13}),
              (9, 5, {'weight': 14}), (8, 7, {'weight': 15}), (8, 6, {'weight': 14}), (7, 6, {'weight': 13})]

    def test_sparse_graph(self):
        sg = sparse_graph(self.g, 3, 'weight')
        edges = list(sg.edges(data=True))
        print(edges)
        self.assertEqual(len(sg.edges), 12)
        self.assertEqual(self.SPARSE, edges)


if __name__ == "__main__":
    unittest.main()
