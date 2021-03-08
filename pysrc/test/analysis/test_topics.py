import unittest

from parameterized import parameterized

from pysrc.papers.analysis.graph import build_citation_graph
from pysrc.papers.analysis.topics import merge_components
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.test.mock_loaders import MockLoader


class TestTopics(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        loader = MockLoader()
        cls.analyzer = PapersAnalyzer(loader, TestTopics.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.TOPIC_MIN_SIZE = 0  # Disable merging for tests
        cls.analyzer.analyze_papers(ids, 'query')
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.ids)
        cls.analyzer.citations_graph = build_citation_graph(cls.analyzer.cit_df)
        cls.analyzer.bibliographic_coupling_df = loader.load_bibliographic_coupling(cls.analyzer.ids)

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

    @parameterized.expand([
        ('5_0', {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 5, 0, ({1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 0)),
        ('4_0', {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 4, 0, ({1: 1, 2: 2, 3: 3, 4: 0, 5: 0}, 2)),
        ('1_0', {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}, 1, 0, ({1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, 2)),
        ('5_10', {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 1, 10, ({1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, 5)),
    ])
    def test_merge_components(self, name, partition, max_topics_number, topic_min_size, expected):
        partition, n_components_merged = merge_components(
            partition, topic_min_size=topic_min_size, max_topics_number=max_topics_number)
        expected_partition, expected_merged = expected
        self.assertEqual(n_components_merged, expected_merged, name)
        self.assertEqual(partition, expected_partition, name)


if __name__ == '__main__':
    unittest.main()
