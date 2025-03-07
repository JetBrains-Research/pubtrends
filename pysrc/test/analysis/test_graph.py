import unittest

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
        cls.analyzer.analyze_papers(ids, 'query', test=True)

    def test_build_papers_graph(self):
        edges = list(self.analyzer.papers_graph.edges(data=True))
        # print(edges)
        self.assertEqual(len(PAPERS_GRAPH_EDGES), len(edges), msg='size')
        for expected, actual in zip(PAPERS_GRAPH_EDGES, edges):
            self.assertEqual(expected[0], actual[0], msg='source')
            self.assertEqual(expected[1], actual[1], msg='target')
            for m in 'cocitation', 'citations', 'bibcoupling':
                self.assertEqual(m in expected[2], m in actual[2], msg=f'{m} presence')
                if m in expected[2]:
                    self.assertAlmostEqual(expected[2][m], actual[2][m], msg=f'{m} value', delta=1e-3)


class TestBuildGraphSingle(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = PapersAnalyzer(MockLoaderSingle(), TestBuildGraphSingle.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.analyze_papers(ids, 'query', test=True)
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.df['id'])


if __name__ == "__main__":
    unittest.main()
