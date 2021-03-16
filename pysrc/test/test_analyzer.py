import unittest

import numpy as np
from pandas._testing import assert_frame_equal

from pysrc.papers.analysis.citations import build_cocit_grouped_df, merge_citation_stats
from pysrc.papers.analysis.graph import build_citation_graph
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.test.mock_loaders import MockLoader, \
    CITATION_YEARS, MockLoaderEmpty, MockLoaderSingle, BIBCOUPLING_DF, COCITATION_DF


class TestKeyPaperAnalyzer(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        loader = MockLoader()
        cls.analyzer = PapersAnalyzer(loader, TestKeyPaperAnalyzer.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.TOPIC_MIN_SIZE = 0  # Disable merging for tests
        cls.analyzer.analyze_papers(ids, 'query')
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.ids)
        cls.analyzer.citations_graph = build_citation_graph(cls.analyzer.cit_df)
        cls.analyzer.bibliographic_coupling_df = loader.load_bibliographic_coupling(cls.analyzer.ids)

    def test_bibcoupling(self):
        assert_frame_equal(BIBCOUPLING_DF, self.analyzer.bibliographic_coupling_df)

    def test_cocitation(self):
        df = build_cocit_grouped_df(self.analyzer.cocit_df)[
            ['cited_1', 'cited_2', 'total']].reset_index(drop=True)
        df.columns = ['cited_1', 'cited_2', 'total']
        assert_frame_equal(COCITATION_DF, df)

    def test_merge_comps_paper_count(self):
        self.assertEqual(len(self.analyzer.df), len(self.analyzer.pub_df))

    def test_topic_analysis_all_nodes_assigned(self):
        nodes = self.analyzer.similarity_graph.nodes()
        for row in self.analyzer.df.itertuples():
            if getattr(row, 'id') in nodes:
                self.assertGreaterEqual(getattr(row, 'comp'), 0)

    def test_topic_analysis_missing_nodes_set_to_default(self):
        nodes = self.analyzer.similarity_graph.nodes()
        for row in self.analyzer.df.itertuples():
            if getattr(row, 'id') not in nodes:
                self.assertEqual(getattr(row, 'comp'), -1)

    def test_merge_citation_stats_paper_count(self):
        df, _ = merge_citation_stats(self.analyzer.pub_df, self.analyzer.cit_stats_df)
        self.assertEqual(len(df), len(self.analyzer.pub_df))

    def test_merge_citation_stats_total_value_ge_0(self):
        df, _ = merge_citation_stats(self.analyzer.pub_df, self.analyzer.cit_stats_df)
        added_columns = self.analyzer.cit_stats_df.columns
        self.assertFalse(np.any(df[added_columns].isna()), msg='NaN values in citation stats')
        self.assertTrue(np.all(df['total'] >= 0), msg='Negative total citations count')

    def test_merge_citation_stats_citation_years(self):
        _, citation_years = merge_citation_stats(self.analyzer.pub_df, self.analyzer.cit_stats_df)
        self.assertCountEqual(citation_years, CITATION_YEARS)


class TestKeyPaperAnalyzerSingle(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = PapersAnalyzer(MockLoaderSingle(), TestKeyPaperAnalyzerSingle.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.analyze_papers(ids, 'query')
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.ids)
        cls.analyzer.citations_graph = build_citation_graph(cls.analyzer.cit_df)

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
            # These are optional
            # 'journal_stats',
            # 'author_stats',
        ]
        for a in all_attrs:
            self.assertTrue(hasattr(self.analyzer, a), f'Missing attr {a}')

    def test_dump(self):
        dump = self.analyzer.dump()
        self.assertEqual(
            '{"comp":{"0":0},"kwd":{"0":"article:0.167,paper:0.167,kw1:0.167,term1:0.167,term2:0.167,term3:0.167"}}',
            dump['kwd_df'])


class TestKeyPaperAnalyzerMissingPaper(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    def test_missing_paper(self):
        analyzer = PapersAnalyzer(MockLoaderSingle(), TestKeyPaperAnalyzerSingle.PUBTRENDS_CONFIG, test=True)
        good_ids = list(analyzer.search_terms(query='query'))
        analyzer.analyze_papers(good_ids + ['non-existing-id'], 'query')
        self.assertEqual(good_ids, list(analyzer.ids))


class TestKeyPaperAnalyzerEmpty(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = PapersAnalyzer(MockLoaderEmpty(),
                                      TestKeyPaperAnalyzerEmpty.PUBTRENDS_CONFIG, test=True)

    def test_setup(self):
        with self.assertRaises(Exception):
            ids = self.analyzer.search_terms(query='query')
            self.analyzer.analyze_papers(ids, query='query', task=None)


if __name__ == '__main__':
    unittest.main()
