import unittest

import numpy as np
import pandas as pd
from parameterized import parameterized

from pysrc.papers.analysis.graph import build_citation_graph
from pysrc.papers.analysis.topics import compute_similarity_matrix, _get_topics_description_cosine
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
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.df['id'])
        cls.analyzer.citations_graph = build_citation_graph(cls.analyzer.df, cls.analyzer.cit_df)
        cls.analyzer.bibliographic_coupling_df = loader.load_bibliographic_coupling(cls.analyzer.df['id'])

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

    def test_heatmap_topics_similarity(self):
        matrix = compute_similarity_matrix(self.analyzer.similarity_graph,
                                           PapersAnalyzer.similarity, self.analyzer.partition)
        # print(matrix)
        similarities = np.array([[3.583]])
        self.assertTrue(np.allclose(similarities, matrix, rtol=1e-3))

    @staticmethod
    def _get_topics_description_data():
        df = pd.DataFrame([['0'], ['1'], ['2']], columns=['id'])
        query = 'test'
        comps_pids = {0: ['0'], 1: ['1'], 2: ['2']}
        comps = {0: [0], 1: [1], 2: [2]}
        corpus_terms = ['frequent', 'rare-1', 'rare-2']
        corpus_counts = np.array([
            [30, 0, 0],  # 'frequent' x 30
            [30, 15, 0],  # 'frequent' x 30 + 'rare-1' x 15
            [30, 0, 15],  # 'frequent' x 30 + 'rare-2' x 15
        ])

        return df, query, comps_pids, comps, corpus_terms, corpus_counts

    def _compare_topics_descriptions(self, result, expected_result, name):
        self.assertEqual(result.keys(), expected_result.keys(), f'{name}: bad indices of components')

        for comp, data in result.items():
            expected_data = expected_result[comp]

            words = [v[0] for v in data]
            expected_words = [v[0] for v in expected_data]
            self.assertEqual(words, expected_words, f'{name}: words do not match for component {comp}')

            weights = np.array([v[1] for v in data])
            expected_weights = np.array([v[1] for v in expected_data])
            self.assertTrue(np.allclose(weights, expected_weights, rtol=1e-3),
                            f'{name}: weights do not match for component {comp}')

    @parameterized.expand([
        ('1word_all', 1, None, {0: [('frequent', 2.598)], 1: [('rare-1', 2.708)], 2: [('rare-2', 2.708)]}),
        ('2word_ignore0', 2, 0, {0: [],
                                 1: [('frequent', 2.895), ('rare-1', 2.708)],
                                 2: [('frequent', 2.895), ('rare-2', 2.708)]}),
    ])
    def test_get_topics_description_cosine(self, name, n_words, ignore_comp, expected_result):
        _, _, _, comps, corpus_terms, corpus_counts = TestTopics._get_topics_description_data()
        result = _get_topics_description_cosine(comps, corpus_terms, corpus_counts, n_words, ignore_comp=ignore_comp)
        self._compare_topics_descriptions(result, expected_result, name)


if __name__ == '__main__':
    unittest.main()
