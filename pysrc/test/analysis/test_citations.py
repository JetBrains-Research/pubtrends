import unittest

from parameterized import parameterized

from pysrc.config import PubtrendsConfig, SHOW_TOPICS_DEFAULT
from pysrc.papers.analysis.citations import find_top_cited_papers
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.utils import SORT_MOST_CITED, IDS_ANALYSIS_TYPE
from pysrc.test.mock_loaders import MockLoader, \
    EXPECTED_MAX_GAIN, EXPECTED_MAX_RELATIVE_GAIN, MockLoaderSingle

PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

class TestPopularPapers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        loader = MockLoader()
        analyzer = PapersAnalyzer(loader, PUBTRENDS_CONFIG, test=True)
        ids = analyzer.search_terms(query='query')
        analyzer.analyze_papers(ids, SHOW_TOPICS_DEFAULT, test=True)
        cls.data = analyzer.save(IDS_ANALYSIS_TYPE, None, 'query', 'Pubmed', SORT_MOST_CITED, 10, False, None, None)
        cls.data.cit_df = analyzer.loader.load_citations(analyzer.df['id'])

    def test_find_max_gain_papers_count(self):
        max_gain_count = len(list(self.data.max_gain_df['year'].values))
        self.assertEqual(max_gain_count, len(EXPECTED_MAX_GAIN.keys()))

    def test_find_max_gain_papers_years(self):
        max_gain_years = list(self.data.max_gain_df['year'].values)
        self.assertCountEqual(max_gain_years, EXPECTED_MAX_GAIN.keys())

    def test_find_max_gain_papers_ids(self):
        max_gain = dict(self.data.max_gain_df[['year', 'id']].values)
        self.assertDictEqual(max_gain, EXPECTED_MAX_GAIN)

    def test_find_max_relative_gain_papers_count(self):
        max_rel_gain_count = len(list(self.data.max_rel_gain_df['year'].values))
        self.assertEqual(max_rel_gain_count, len(EXPECTED_MAX_RELATIVE_GAIN.keys()))

    def test_find_max_relative_gain_papers_years(self):
        max_rel_gain_years = list(self.data.max_rel_gain_df['year'].values)
        self.assertCountEqual(max_rel_gain_years, EXPECTED_MAX_RELATIVE_GAIN.keys())

    def test_find_max_relative_gain_papers_ids(self):
        max_rel_gain = dict(self.data.max_rel_gain_df[['year', 'id']].values)
        self.assertDictEqual(max_rel_gain, EXPECTED_MAX_RELATIVE_GAIN)

    @parameterized.expand([
        ('threshold 5', 5, ['3', '1', '4', '2', '5']),
        ('threshold 10', 10, ['3', '1', '4', '2', '5']),
        ('limit-2', 2, ['3', '1']),
        ('limit-4', 4, ['3', '1', '4', '2'])
    ])
    def test_find_top_cited_papers(self, name, max_papers, expected):
        top_cited_df = find_top_cited_papers(self.data.df, n_papers=max_papers)
        top_cited_papers = list(top_cited_df['id'].values)
        self.assertListEqual(top_cited_papers, expected, name)


class TestPopularPapersSingle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        analyzer = PapersAnalyzer(MockLoaderSingle(), PUBTRENDS_CONFIG, test=True)
        ids = analyzer.search_terms(query='query')
        analyzer.analyze_papers(ids, SHOW_TOPICS_DEFAULT, test=True)
        cls.data = analyzer.save(IDS_ANALYSIS_TYPE, None, 'query', 'Pubmed', SORT_MOST_CITED, 10, False, None, None)
        cls.data.cit_df = analyzer.loader.load_citations(analyzer.df['id'])

    def test_find_max_gain_papers_count(self):
        max_gain_count = len(list(self.data.max_gain_df['year'].values))
        self.assertEqual(max_gain_count, len(EXPECTED_MAX_GAIN.keys()))

    def test_find_max_gain_papers_years(self):
        max_gain_years = list(self.data.max_gain_df['year'].values)
        self.assertCountEqual(max_gain_years, EXPECTED_MAX_GAIN.keys())

    def test_find_max_gain_papers_ids(self):
        max_gain = dict(self.data.max_gain_df[['year', 'id']].values)
        self.assertDictEqual(max_gain, {1972: '1', 1974: '1'})

    def test_find_max_relative_gain_papers_count(self):
        max_rel_gain_count = len(list(self.data.max_rel_gain_df['year'].values))
        self.assertEqual(max_rel_gain_count, len(EXPECTED_MAX_RELATIVE_GAIN.keys()))

    def test_find_max_relative_gain_papers_years(self):
        max_rel_gain_years = list(self.data.max_rel_gain_df['year'].values)
        self.assertCountEqual(max_rel_gain_years, EXPECTED_MAX_RELATIVE_GAIN.keys())

    def test_find_max_relative_gain_papers_ids(self):
        max_rel_gain = dict(self.data.max_rel_gain_df[['year', 'id']].values)
        self.assertDictEqual(max_rel_gain, {1972: '1', 1974: '1'})


if __name__ == '__main__':
    unittest.main()
