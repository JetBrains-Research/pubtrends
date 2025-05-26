import unittest

import numpy as np
import pandas as pd
from parameterized import parameterized

from pysrc.config import PubtrendsConfig
from pysrc.papers.analysis.descriptions import _get_topics_description_cosine
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.utils import SORT_MOST_CITED
from pysrc.test.mock_loaders import MockLoader

PUBTRENDS_CONFIG = PubtrendsConfig(test=True)


class TestTopics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        analyzer = PapersAnalyzer(MockLoader(), PUBTRENDS_CONFIG, test=True)
        ids = analyzer.search_terms(query='query')
        analyzer.analyze_papers(ids, PUBTRENDS_CONFIG.show_topics_default_value, test=True)
        cls.data = analyzer.save(None, 'query', 'Pubmed', SORT_MOST_CITED, 10, False, None, None)

    def test_topic_analysis_all_nodes_assigned(self):
        nodes = self.data.papers_graph.nodes()
        for row in self.data.df.itertuples():
            if getattr(row, 'id') in nodes:
                self.assertGreaterEqual(getattr(row, 'comp'), 0)

    def test_topic_analysis_missing_nodes_set_to_default(self):
        nodes = self.data.papers_graph.nodes()
        for row in self.data.df.itertuples():
            if getattr(row, 'id') not in nodes:
                self.assertEqual(getattr(row, 'comp'), -1)

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
            self.assertTrue(np.allclose(weights, expected_weights, rtol=1e-2),
                            f'{name}: weights do not match for component {comp}')

    @parameterized.expand([
        ('1word_all', 1, None, {0: [('frequent', 2.60)], 1: [('rare-1', 2.77)], 2: [('rare-2', 2.77)]}),
        ('2word_ignore0', 2, 0,
         {0: [], 1: [('frequent', 2.90), ('rare-1', 2.77)], 2: [('frequent', 2.90), ('rare-2', 2.77)]}
         ),
    ])
    def test_get_topics_description_cosine(self, name, n_words, ignore_comp, expected_result):
        _, _, _, comps, corpus_terms, corpus_counts = TestTopics._get_topics_description_data()
        result = _get_topics_description_cosine(comps, corpus_terms, corpus_counts, n_words, ignore_comp=ignore_comp)
        # print(result)
        self._compare_topics_descriptions(result, expected_result, name)


if __name__ == '__main__':
    unittest.main()
