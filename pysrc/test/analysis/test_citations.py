import unittest

from parameterized import parameterized

from pysrc.papers.analysis.graph import build_citation_graph
from pysrc.papers.analysis.citations import find_top_cited_papers
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.test.mock_loaders import MockLoader, \
    EXPECTED_MAX_GAIN, EXPECTED_MAX_RELATIVE_GAIN, MockLoaderSingle


class TestPopularPapers(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        loader = MockLoader()
        cls.analyzer = PapersAnalyzer(loader, TestPopularPapers.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.TOPIC_MIN_SIZE = 0  # Disable merging for tests
        cls.analyzer.analyze_papers(ids, 'query')
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.df['id'])
        cls.analyzer.citations_graph = build_citation_graph(cls.analyzer.df, cls.analyzer.cit_df)
        cls.analyzer.bibliographic_coupling_df = loader.load_bibliographic_coupling(cls.analyzer.df['id'])

    def test_find_max_gain_papers_count(self):
        max_gain_count = len(list(self.analyzer.max_gain_df['year'].values))
        self.assertEqual(max_gain_count, len(EXPECTED_MAX_GAIN.keys()))

    def test_find_max_gain_papers_years(self):
        max_gain_years = list(self.analyzer.max_gain_df['year'].values)
        self.assertCountEqual(max_gain_years, EXPECTED_MAX_GAIN.keys())

    def test_find_max_gain_papers_ids(self):
        max_gain = dict(self.analyzer.max_gain_df[['year', 'id']].values)
        self.assertDictEqual(max_gain, EXPECTED_MAX_GAIN)

    def test_find_max_relative_gain_papers_count(self):
        max_rel_gain_count = len(list(self.analyzer.max_rel_gain_df['year'].values))
        self.assertEqual(max_rel_gain_count, len(EXPECTED_MAX_RELATIVE_GAIN.keys()))

    def test_find_max_relative_gain_papers_years(self):
        max_rel_gain_years = list(self.analyzer.max_rel_gain_df['year'].values)
        self.assertCountEqual(max_rel_gain_years, EXPECTED_MAX_RELATIVE_GAIN.keys())

    def test_find_max_relative_gain_papers_ids(self):
        max_rel_gain = dict(self.analyzer.max_rel_gain_df[['year', 'id']].values)
        self.assertDictEqual(max_rel_gain, EXPECTED_MAX_RELATIVE_GAIN)

    @parameterized.expand([
        ('threshold 5', 5, ['3', '1', '4', '2', '5']),
        ('threshold 10', 10, ['3', '1', '4', '2', '5']),
        ('limit-2', 2, ['3', '1']),
        ('limit-4', 4, ['3', '1', '4', '2'])
    ])
    def test_find_top_cited_papers(self, name, max_papers, expected):
        _, top_cited_df = find_top_cited_papers(self.analyzer.df, n_papers=max_papers)
        top_cited_papers = list(top_cited_df['id'].values)
        self.assertListEqual(top_cited_papers, expected, name)


class TestPopularPapersSingle(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = PapersAnalyzer(MockLoaderSingle(), TestPopularPapersSingle.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.analyze_papers(ids, 'query')
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.df['id'])

    def test_find_max_gain_papers_count(self):
        max_gain_count = len(list(self.analyzer.max_gain_df['year'].values))
        self.assertEqual(max_gain_count, len(EXPECTED_MAX_GAIN.keys()))

    def test_find_max_gain_papers_years(self):
        max_gain_years = list(self.analyzer.max_gain_df['year'].values)
        self.assertCountEqual(max_gain_years, EXPECTED_MAX_GAIN.keys())

    def test_find_max_gain_papers_ids(self):
        max_gain = dict(self.analyzer.max_gain_df[['year', 'id']].values)
        self.assertDictEqual(max_gain, {1972: '1', 1974: '1'})

    def test_find_max_relative_gain_papers_count(self):
        max_rel_gain_count = len(list(self.analyzer.max_rel_gain_df['year'].values))
        self.assertEqual(max_rel_gain_count, len(EXPECTED_MAX_RELATIVE_GAIN.keys()))

    def test_find_max_relative_gain_papers_years(self):
        max_rel_gain_years = list(self.analyzer.max_rel_gain_df['year'].values)
        self.assertCountEqual(max_rel_gain_years, EXPECTED_MAX_RELATIVE_GAIN.keys())

    def test_find_max_relative_gain_papers_ids(self):
        max_rel_gain = dict(self.analyzer.max_rel_gain_df[['year', 'id']].values)
        self.assertDictEqual(max_rel_gain, {1972: '1', 1974: '1'})


if __name__ == '__main__':
    unittest.main()
