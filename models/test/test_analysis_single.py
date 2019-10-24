import unittest

from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.test.mock_loader import EXPECTED_MAX_GAIN, EXPECTED_MAX_RELATIVE_GAIN, MockLoaderSingle


class TestKeyPaperAnalyzerSingle(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = KeyPaperAnalyzer(MockLoaderSingle(), TestKeyPaperAnalyzerSingle.PUBTRENDS_CONFIG, test=True)
        ids, pub_df = cls.analyzer.search_terms(query='query')
        cls.analyzer.analyze_papers(ids, pub_df, 'query')
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.ids)
        cls.analyzer.G = cls.analyzer.build_citation_graph(cls.analyzer.cit_df)

    def test_build_citation_graph_nodes_count(self):
        self.assertEqual(self.analyzer.G.number_of_nodes(), 0)

    def test_build_citation_graph_edges_count(self):
        self.assertEqual(self.analyzer.G.number_of_edges(), 0)

    def test_build_citation_graph_nodes(self):
        self.assertCountEqual(list(self.analyzer.G.nodes()), [])

    def test_build_citation_graph_edges(self):
        self.assertCountEqual(list(self.analyzer.G.edges()), [])

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

    def test_attrs(self):
        all_attrs = [
            'ids',
            'pub_df',
            'query',
            'n_papers',
            'pub_types',
            'cit_stats_df',
            'cit_df',
            'top_cited_papers',
            'top_cited_df',
            'max_gain_papers',
            'max_gain_df',
            'max_rel_gain_papers',
            'max_rel_gain_df',
            'journal_stats',
            'author_stats',
        ]
        for a in all_attrs:
            self.assertTrue(hasattr(self.analyzer, a), f'Missing attr {a}')

    def test_dump(self):
        dump = self.analyzer.dump()
        self.assertEqual('{"comp":{"0":0},"kwd":{"0":""}}', dump['df_kwd'])


if __name__ == '__main__':
    unittest.main()
