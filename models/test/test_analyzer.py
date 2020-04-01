import unittest

import numpy as np
from parameterized import parameterized

from models.keypaper.analyzer import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.keypaper.pm_loader import PubmedLoader
from models.keypaper.ss_loader import SemanticScholarLoader
from models.test.mock_loaders import MockLoader, \
    CITATION_YEARS, EXPECTED_MAX_GAIN, EXPECTED_MAX_RELATIVE_GAIN, CITATION_GRAPH_NODES, CITATION_GRAPH_EDGES, \
    MockLoaderEmpty, MockLoaderSingle, SIMILARITY_GRAPH_EDGES


class TestKeyPaperAnalyzer(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        loader = MockLoader()
        cls.analyzer = KeyPaperAnalyzer(loader, TestKeyPaperAnalyzer.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.analyze_papers(ids, 'query')
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.ids)
        cls.analyzer.citations_graph = cls.analyzer.build_citation_graph(cls.analyzer.cit_df)
        cls.analyzer.bibliographic_coupling_df = loader.load_bibliographic_coupling(cls.analyzer.ids)

    @parameterized.expand([
        ('Pubmed', PubmedLoader(PUBTRENDS_CONFIG), False, 'Pubmed'),
        ('Semantic Scholar', SemanticScholarLoader(PUBTRENDS_CONFIG), False, 'Semantic Scholar')
    ])
    def test_valid_source(self, name, loader, test, expected):
        analyzer = KeyPaperAnalyzer(loader, TestKeyPaperAnalyzer.PUBTRENDS_CONFIG, test=test)
        self.assertEqual(analyzer.source, expected, name)

    def test_bad_source(self):
        with self.assertRaises(TypeError):
            KeyPaperAnalyzer(MockLoader(), TestKeyPaperAnalyzer.PUBTRENDS_CONFIG, test=False)

    def test_build_citation_graph_nodes_count(self):
        self.assertEqual(self.analyzer.citations_graph.number_of_nodes(), len(CITATION_GRAPH_NODES))

    def test_build_citation_graph_edges_count(self):
        self.assertEqual(self.analyzer.citations_graph.number_of_edges(), len(CITATION_GRAPH_EDGES))

    def test_build_citation_graph_nodes(self):
        self.assertCountEqual(list(self.analyzer.citations_graph.nodes()), CITATION_GRAPH_NODES)

    def test_build_citation_graph_edges(self):
        self.assertCountEqual(list(self.analyzer.citations_graph.edges()), CITATION_GRAPH_EDGES)

    def test_build_similarity_graph_edges(self):
        similarity_graph = self.analyzer.build_similarity_graph(
            self.analyzer.df,
            self.analyzer.citations_graph,
            self.analyzer.build_cocit_grouped_df(self.analyzer.cocit_df),
            self.analyzer.bibliographic_coupling_df
        )
        self.assertCountEqual(list(similarity_graph.edges(data=True)), SIMILARITY_GRAPH_EDGES)

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

    def test_merge_comps_paper_count(self):
        self.assertEqual(len(self.analyzer.df), len(self.analyzer.pub_df))

    def test_subtopic_analysis_all_nodes_assigned(self):
        nodes = self.analyzer.similarity_graph.nodes()
        for row in self.analyzer.df.itertuples():
            if getattr(row, 'id') in nodes:
                self.assertGreaterEqual(getattr(row, 'comp'), 0)

    def test_subtopic_analysis_missing_nodes_set_to_default(self):
        nodes = self.analyzer.similarity_graph.nodes()
        for row in self.analyzer.df.itertuples():
            if getattr(row, 'id') not in nodes:
                self.assertEqual(getattr(row, 'comp'), -1)

    @parameterized.expand([
        ('threshold 1', 5, 1, ['3', '1', '4', '2', '5']),
        ('threshold 1.5', 10, 1.5, ['3', '1', '4', '2', '5']),
        ('limit-1', 2, 0.5, ['3', '1']),
        ('limit-2', 4, 0.8, ['3', '1', '4', '2']),
        ('pick at least 1', 50, 0.1, ['3'])
    ])
    def test_find_top_cited_papers(self, name, max_papers, threshold, expected):
        _, top_cited_df = self.analyzer.find_top_cited_papers(self.analyzer.df, n_papers=max_papers,
                                                              threshold=threshold)
        top_cited_papers = list(top_cited_df['id'].values)
        self.assertListEqual(top_cited_papers, expected, name)

    @parameterized.expand([
        ('granularity 0', {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 0, ({1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 0)),
        ('granularity 1', {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 1, ({1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, 1)),
        ('granularity 0.5', {1: 0, 2: 1, 3: 1, 4: 1, 5: 2}, 0.5, ({1: 0, 2: 1, 3: 1, 4: 1, 5: 0}, 1)),
        ('do not merge one component', {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}, 0.5, ({1: 0, 2: 1, 3: 1, 4: 1, 5: 1}, 0))
    ])
    def test_merge_components(self, name, partition, granularity, expected):
        partition, n_components_merged = self.analyzer.merge_components(partition, granularity)
        expected_partition, expected_merged = expected
        self.assertEqual(partition, expected_partition, name)

    def test_merge_citation_stats_paper_count(self):
        df, _, _, _ = self.analyzer.merge_citation_stats(self.analyzer.pub_df, self.analyzer.cit_stats_df)
        self.assertEqual(len(df), len(self.analyzer.pub_df))

    def test_merge_citation_stats_total_value_ge_0(self):
        df, _, _, _ = self.analyzer.merge_citation_stats(self.analyzer.pub_df, self.analyzer.cit_stats_df)
        added_columns = self.analyzer.cit_stats_df.columns
        self.assertFalse(np.any(df[added_columns].isna()), msg='NaN values in citation stats')
        self.assertTrue(np.all(df['total'] >= 0), msg='Negative total citations count')

    def test_merge_citation_stats_citation_years(self):
        _, _, _, citation_years = self.analyzer.merge_citation_stats(self.analyzer.pub_df, self.analyzer.cit_stats_df)
        self.assertCountEqual(citation_years, CITATION_YEARS)

    def test_get_most_cited_papers_for_comps(self):
        comps = self.analyzer.get_most_cited_papers_for_comps(self.analyzer.df, self.analyzer.partition, 1)
        self.assertDictEqual(comps, {0: ['3']})


class TestKeyPaperAnalyzerSingle(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = KeyPaperAnalyzer(MockLoaderSingle(), TestKeyPaperAnalyzerSingle.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.analyze_papers(ids, 'query')
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.ids)
        cls.analyzer.citations_graph = cls.analyzer.build_citation_graph(cls.analyzer.cit_df)

    def test_build_citation_graph_nodes_count(self):
        self.assertEqual(self.analyzer.citations_graph.number_of_nodes(), 0)

    def test_build_citation_graph_edges_count(self):
        self.assertEqual(self.analyzer.citations_graph.number_of_edges(), 0)

    def test_build_citation_graph_nodes(self):
        self.assertCountEqual(list(self.analyzer.citations_graph.nodes()), [])

    def test_build_citation_graph_edges(self):
        self.assertCountEqual(list(self.analyzer.citations_graph.edges()), [])

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
        self.assertEqual('{"comp":{"0":0},"kwd":{"0":"article:0.500,paper:0.500"}}', dump['df_kwd'])


class TestKeyPaperAnalyzerMissingPaper(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    def test_missing_paper(self):
        analyzer = KeyPaperAnalyzer(MockLoaderSingle(), TestKeyPaperAnalyzerSingle.PUBTRENDS_CONFIG, test=True)
        good_ids = list(analyzer.search_terms(query='query'))
        analyzer.analyze_papers(good_ids + ['non-existing-id'], 'query')
        self.assertEqual(good_ids, list(analyzer.ids))


class TestKeyPaperAnalyzerEmpty(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = KeyPaperAnalyzer(MockLoaderEmpty(),
                                        TestKeyPaperAnalyzerEmpty.PUBTRENDS_CONFIG, test=True)

    def test_setup(self):
        with self.assertRaises(Exception):
            ids = self.analyzer.search_terms(query='query')
            self.analyzer.analyze_papers(ids, query='query', task=None)


if __name__ == '__main__':
    unittest.main()
