import unittest

from parameterized import parameterized

from pysrc.papers.analysis.evolution import topic_evolution_analysis, topic_evolution_descriptions
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.test.mock_loaders import MockLoader
from pysrc.test.test_analyzer import TestPapersAnalyzer


class TestTopicEvolution(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        loader = MockLoader()
        cls.analyzer = PapersAnalyzer(loader, TestPapersAnalyzer.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.analyze_papers(ids, 'query', test=True)
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.df['id'])
        cls.analyzer.bibliographic_coupling_df = loader.load_bibliographic_coupling(cls.analyzer.df['id'])

    @parameterized.expand([
        ('too large step', 20, True, None),
        ('3 steps', 5, False, [1975, 1970, 1965]),
        ('7 steps', 2, False, [1975, 1973, 1971, 1969, 1967, 1965, 1963])
    ])
    def test_topic_evolution(self, name, step, expect_none, expected_year_range):
        evolution_df, year_range = topic_evolution_analysis(self.analyzer.df, self.analyzer.cit_df,
                                                            self.analyzer.cocit_df,
                                                            self.analyzer.bibliographic_coupling_df,
                                                            PapersAnalyzer.SIMILARITY_COCITATION_MIN,
                                                            PapersAnalyzer.similarity, self.analyzer.corpus_counts,
                                                            self.analyzer.corpus_tokens_embedding,
                                                            PapersAnalyzer.GRAPH_EMBEDDINGS_FACTOR,
                                                            PapersAnalyzer.TEXT_EMBEDDINGS_FACTOR,
                                                            20, 20,
                                                            evolution_step=step)

        if expect_none:
            self.assertIsNone(evolution_df, msg=f'Evolution DataFrame is not None when step is too large {name}')

        if expected_year_range:
            self.assertListEqual(year_range, expected_year_range, msg=f'Wrong year range {name}')
            # Additional id column
            self.assertEqual(len(year_range), len(evolution_df.columns) - 1, msg=f'Wrong n_steps {name}')
        else:
            self.assertIsNone(year_range, msg=f'Year range is not None when step is too large {name}')

    def test_topic_evolution_description(self):
        evolution_df, year_range = topic_evolution_analysis(self.analyzer.df, self.analyzer.cit_df,
                                                            self.analyzer.cocit_df,
                                                            self.analyzer.bibliographic_coupling_df,
                                                            PapersAnalyzer.SIMILARITY_COCITATION_MIN,
                                                            PapersAnalyzer.similarity, self.analyzer.corpus_counts,
                                                            self.analyzer.corpus_tokens_embedding,
                                                            PapersAnalyzer.GRAPH_EMBEDDINGS_FACTOR,
                                                            PapersAnalyzer.TEXT_EMBEDDINGS_FACTOR,
                                                            20, 20,
                                                            evolution_step=5)

        evolution_kwds = topic_evolution_descriptions(
            self.analyzer.df, evolution_df, year_range,
            self.analyzer.corpus, self.analyzer.corpus_tokens, self.analyzer.corpus_counts,
            PapersAnalyzer.TOPIC_DESCRIPTION_WORDS,
            self.analyzer.progress
        )
        # print(evolution_kwds)
        expected_topics_kwds = {1965: {
            0: [('article', 0.2857142857142857), ('term2', 0.2857142857142857), ('term3', 0.2857142857142857),
                ('paper', 0.14285714285714285), ('term1', 0.14285714285714285), ('abstract', 0.14285714285714285),
                ('term4', 0.14285714285714285)], -1: []}, 1970: {
            0: [('article', 0.4444444444444444), ('term3', 0.3333333333333333), ('term4', 0.3333333333333333),
                ('paper', 0.2222222222222222), ('term1', 0.2222222222222222), ('term2', 0.2222222222222222),
                ('abstract', 0.2222222222222222), ('term5', 0.2222222222222222), ('interesting', 0.1111111111111111)],
            -1: []},
            1975: {
            0: [('article', 0.5), ('term1', 0.3), ('term2', 0.3), ('term3', 0.3), ('term4', 0.3), ('term5', 0.3),
                ('paper', 0.2), ('abstract', 0.2), ('interesting', 0.1), ('breakthrough', 0.1)], -1: []}}
        self.assertEquals(expected_topics_kwds, evolution_kwds)


if __name__ == '__main__':
    unittest.main()
