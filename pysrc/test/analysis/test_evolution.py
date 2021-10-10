import unittest

from parameterized import parameterized

from pysrc.papers.analysis.evolution import topic_evolution_analysis, topic_evolution_descriptions
from pysrc.papers.analysis.graph import build_citation_graph
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
        cls.analyzer.analyze_papers(ids, 'query')
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.df['id'])
        cls.analyzer.citations_graph = build_citation_graph(cls.analyzer.df, cls.analyzer.cit_df)
        cls.analyzer.bibliographic_coupling_df = loader.load_bibliographic_coupling(cls.analyzer.df['id'])

    @parameterized.expand([
        ('too large step', 20, True, None),
        ('3 steps', 5, False, [1975, 1970, 1965]),
        ('7 steps', 2, False, [1975, 1973, 1971, 1969, 1967, 1965, 1963])
    ])
    def test_topic_evolution(self, name, step, expect_none, expected_year_range):
        evolution_df, year_range = topic_evolution_analysis(
            self.analyzer.df, self.analyzer.cit_df, self.analyzer.cocit_df,
            self.analyzer.bibliographic_coupling_df,
            self.analyzer.texts_similarity, PapersAnalyzer.SIMILARITY_COCITATION_MIN, PapersAnalyzer.TOPIC_MIN_SIZE,
            PapersAnalyzer.TOPICS_MAX_NUMBER, similarity_func=PapersAnalyzer.similarity,
            evolution_step=step
        )

        if expect_none:
            self.assertIsNone(evolution_df, msg=f'Evolution DataFrame is not None when step is too large {name}')

        if expected_year_range:
            self.assertListEqual(year_range, expected_year_range, msg=f'Wrong year range {name}')
            # Additional id column
            self.assertEqual(len(year_range), len(evolution_df.columns) - 1, msg=f'Wrong n_steps {name}')
        else:
            self.assertIsNone(year_range, msg=f'Year range is not None when step is too large {name}')

    def test_topic_evolution_description(self):
        evolution_df, year_range = topic_evolution_analysis(
            self.analyzer.df, self.analyzer.cit_df, self.analyzer.cocit_df,
            self.analyzer.bibliographic_coupling_df,
            self.analyzer.texts_similarity, PapersAnalyzer.SIMILARITY_COCITATION_MIN, PapersAnalyzer.TOPIC_MIN_SIZE,
            PapersAnalyzer.TOPICS_MAX_NUMBER, similarity_func=PapersAnalyzer.similarity,
            evolution_step=5
        )

        evolution_kwds = topic_evolution_descriptions(
            self.analyzer.df, evolution_df, year_range,
            self.analyzer.corpus_tokens, self.analyzer.corpus_counts, None, PapersAnalyzer.TOPIC_DESCRIPTION_WORDS,
            self.analyzer.progress
        )
        expected_topics_kwds = {1965: {
            0: [('article', 0.2), ('term2', 0.2), ('term3', 0.2), ('kw2', 0.2), ('paper', 0.1), ('term1', 0.1),
                ('kw1', 0.1), ('abstract', 0.1), ('term4', 0.1), ('kw3', 0.1)], -1: []}, 1970: {
            0: [('article', 0.2857142857142857), ('term3', 0.21428571428571427), ('term4', 0.21428571428571427),
                ('paper', 0.14285714285714285), ('term1', 0.14285714285714285), ('term2', 0.14285714285714285),
                ('kw2', 0.14285714285714285), ('abstract', 0.14285714285714285), ('kw3', 0.14285714285714285),
                ('term5', 0.14285714285714285)], -1: []}, 1975: {
            0: [('article', 0.3333333333333333), ('term1', 0.2), ('term2', 0.2), ('term3', 0.2), ('term4', 0.2),
                ('term5', 0.2), ('paper', 0.13333333333333333), ('kw1', 0.13333333333333333),
                ('kw2', 0.13333333333333333), ('abstract', 0.13333333333333333)], -1: []}}
        self.assertEquals(expected_topics_kwds, evolution_kwds)


if __name__ == '__main__':
    unittest.main()
