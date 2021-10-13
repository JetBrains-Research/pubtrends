import unittest

from pandas._testing import assert_frame_equal

from pysrc.papers.analysis.citations import build_cocit_grouped_df
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.test.mock_loaders import MockLoader, \
    MockLoaderEmpty, MockLoaderSingle, BIBCOUPLING_DF, COCITATION_DF


class TestPapersAnalyzer(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        loader = MockLoader()
        cls.analyzer = PapersAnalyzer(loader, TestPapersAnalyzer.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.TOPIC_MIN_SIZE = 0  # Disable merging for tests
        cls.analyzer.analyze_papers(ids, 'query', test=True)
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.df['id'])
        cls.analyzer.bibliographic_coupling_df = loader.load_bibliographic_coupling(cls.analyzer.df['id'])

    def test_bibcoupling(self):
        assert_frame_equal(BIBCOUPLING_DF, self.analyzer.bibliographic_coupling_df)

    def test_cocitation(self):
        df = build_cocit_grouped_df(self.analyzer.cocit_df)[
            ['cited_1', 'cited_2', 'total']].reset_index(drop=True)
        df.columns = ['cited_1', 'cited_2', 'total']
        assert_frame_equal(COCITATION_DF, df)

    def test_topic_analysis_all_nodes_assigned(self):
        nodes = self.analyzer.papers_graph.nodes()
        for row in self.analyzer.df.itertuples():
            if getattr(row, 'id') in nodes:
                self.assertGreaterEqual(getattr(row, 'comp'), 0)

    def test_topic_analysis_missing_nodes_set_to_default(self):
        nodes = self.analyzer.papers_graph.nodes()
        for row in self.analyzer.df.itertuples():
            if getattr(row, 'id') not in nodes:
                self.assertEqual(getattr(row, 'comp'), -1)

class TestPapersAnalyzerSingle(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = PapersAnalyzer(MockLoaderSingle(), TestPapersAnalyzerSingle.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.analyze_papers(ids, 'query', test=True)
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.df['id'])

    def test_attrs(self):
        all_attrs = [
            'df',
            'query',
            'sparse_papers_graph',
            'pub_types',
            'cit_stats_df',
            'cit_df',
            'top_cited_papers',
            'top_cited_df',
            'max_gain_papers',
            'max_gain_df',
            'max_rel_gain_papers',
            'max_rel_gain_df',
        ]
        for a in all_attrs:
            self.assertTrue(hasattr(self.analyzer, a), f'Missing attr {a}')

    def test_dump(self):
        dump = self.analyzer.dump()
        self.assertEqual(
            '{"comp":{"0":0},"kwd":{"0":"article:0.200,paper:0.200,term1:0.200,term2:0.200,term3:0.200"}}',
            dump['kwd_df'])


class TestPapersAnalyzerMissingPaper(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    def test_missing_paper(self):
        analyzer = PapersAnalyzer(MockLoaderSingle(), TestPapersAnalyzerSingle.PUBTRENDS_CONFIG, test=True)
        good_ids = list(analyzer.search_terms(query='query'))
        analyzer.analyze_papers(good_ids + ['non-existing-id'], 'query', test=True)
        self.assertEqual(good_ids, list(analyzer.df['id']))


class TestPapersAnalyzerEmpty(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = PapersAnalyzer(MockLoaderEmpty(),
                                      TestPapersAnalyzerEmpty.PUBTRENDS_CONFIG, test=True)

    def test_setup(self):
        with self.assertRaises(Exception):
            ids = self.analyzer.search_terms(query='query')
            self.analyzer.analyze_papers(ids, query='query', test=True)


if __name__ == '__main__':
    unittest.main()
