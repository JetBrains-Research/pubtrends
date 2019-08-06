import unittest

from parameterized import parameterized

from models.keypaper.analysis import KeyPaperAnalyzer
from models.test.mock_loader import MockLoader, COCITATION_GRAPH_EDGES, COCITATION_GRAPH_NODES, \
    CITATION_YEARS, EXPECTED_MAX_GAIN, EXPECTED_MAX_RELATIVE_GAIN


class TestKeyPaperAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.analyzer = KeyPaperAnalyzer(MockLoader(), test=True)
        cls.analyzer.launch()

    def test_build_cocitation_graph_nodes_count(self):
        self.assertEqual(self.analyzer.CG.number_of_nodes(), len(COCITATION_GRAPH_NODES))

    def test_build_cocitation_graph_edges_count(self):
        self.assertEqual(self.analyzer.CG.number_of_nodes(), len(COCITATION_GRAPH_NODES))

    def test_build_cocitation_graph_nodes(self):
        self.assertCountEqual(list(self.analyzer.CG.nodes()), COCITATION_GRAPH_NODES)

    def test_build_cocitation_graph_edges(self):
        # Convert edge data to networkx format
        expected_edges = [(v, u, {'weight': w}) for v, u, w in COCITATION_GRAPH_EDGES]
        self.assertCountEqual(list(self.analyzer.CG.edges(data=True)), expected_edges)

    def test_update_years(self):
        self.assertCountEqual(self.analyzer.citation_years, CITATION_YEARS)

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

    def test_subtopic_analysis_all_nodes_assigned(self):
        nodes = self.analyzer.CG.nodes()
        for row in self.analyzer.df.itertuples():
            if getattr(row, 'id') in nodes:
                self.assertGreaterEqual(getattr(row, 'comp'), 0)

    def test_subtopic_analysis_missing_nodes_set_to_default(self):
        nodes = self.analyzer.CG.nodes()
        for row in self.analyzer.df.itertuples():
            if getattr(row, 'id') not in nodes:
                self.assertEqual(getattr(row, 'comp'), -1)

    @parameterized.expand([
        ('threshold 1', 5, 1, ['3', '1', '4', '5', '2']),
        ('threshold 1.5', 10, 1.5, ['3', '1', '4', '5', '2']),
        ('limit-1', 2, 0.5, ['3', '1']),
        ('limit-2', 4, 0.8, ['3', '1', '4', '5']),
        ('pick at least 1', 50, 0.1, ['3'])
    ])
    def test_find_top_cited_papers(self, name, max_papers, threshold, expected):
        _, top_cited_df = self.analyzer.find_top_cited_papers(self.analyzer.df, max_papers=max_papers,
                                                              threshold=threshold)
        top_cited_papers = list(top_cited_df['id'].values)
        self.assertListEqual(top_cited_papers, expected)

    @parameterized.expand([
        ('granularity 0', {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 0, ({1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 0)),
        ('granularity 1', {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 1, ({1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, 1)),
        ('granularity 0.5', {1: 0, 2: 1, 3: 1, 4: 1, 5: 2}, 0.5, ({1: 0, 2: 1, 3: 1, 4: 1, 5: 0}, 1))
    ])
    def test_merge_components(self, name, partition, granularity, expected):
        partition, merged = self.analyzer.merge_components(partition, granularity)
        expected_partition, expected_merged = expected
        self.assertEqual(partition, expected_partition)

# class TestKeyPaperAnalyzerDataLeak(unittest.TestCase):
#
#     def setUp(self):
#         self.analyzer = KeyPaperAnalyzer(MockLoader())
#
#         # Load publications
#         self.analyzer.loader.search()
#         self.analyzer.loader.load_publications()
#         self.paper_count = len(self.analyzer.pub_df)
#
#         # Load citation stats
#         self.analyzer.loader.load_citation_stats()
#         self.analyzer.merge_citation_stats()
#         self.paper_count_after_stat_merge = len(self.analyzer.df)
#
#         # Load cocitations and perform subtopic analysis
#         self.analyzer.loader.load_cocitations()
#         self.analyzer.build_cocitation_graph(self.analyzer.cocit_grouped_df)
#         self.analyzer.cocit_df = self.analyzer.loader.cocit_df
#         self.analyzer.update_years()
#         self.analyzer.subtopic_analysis()
#         self.paper_count_after_comp_merge = len(self.analyzer.df)
#
#     def test_paper_count_after_citation_stats_merge(self):
#         self.assertEqual(self.paper_count_after_stat_merge, self.paper_count)
#
#     def test_paper_count_after_subtopic_data_merge(self):
#         self.assertEqual(self.paper_count_after_comp_merge, self.paper_count)
