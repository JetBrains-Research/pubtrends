import unittest

from parameterized import parameterized

from pysrc.papers.analysis.evolution import topic_evolution_analysis
from pysrc.papers.analysis.graph import build_citation_graph
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.test.mock_loaders import MockLoader
from pysrc.test.test_analyzer import TestKeyPaperAnalyzer


class TestTopicEvolution(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        loader = MockLoader()
        cls.analyzer = PapersAnalyzer(loader, TestKeyPaperAnalyzer.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.analyze_papers(ids, 'query')
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.ids)
        cls.analyzer.citations_graph = build_citation_graph(cls.analyzer.cit_df)
        cls.analyzer.bibliographic_coupling_df = loader.load_bibliographic_coupling(cls.analyzer.ids)

    @parameterized.expand([
        ('too large step', 10, True, None),
        ('2 steps', 5, False, [1975, 1970]),
        ('5 steps', 2, False, [1975, 1973, 1971, 1969, 1967])
    ])
    def test_topic_evolution(self, name, step, expect_none, expected_year_range):
        evolution_df, year_range = topic_evolution_analysis(
            self.analyzer.df, self.analyzer.cit_df, self.analyzer.cocit_df,
            self.analyzer.bibliographic_coupling_df,
            self.analyzer.texts_similarity, PapersAnalyzer.SIMILARITY_COCITATION_MIN, PapersAnalyzer.TOPIC_MIN_SIZE,
            PapersAnalyzer.TOPICS_MAX_NUMBER, similarity_func=PapersAnalyzer.similarity,
            evolution_step=step,
            progress=self.analyzer.progress
        )

        if expect_none:
            self.assertIsNone(evolution_df, msg=f'Evolution DataFrame is not None when step is too large {name}')

        if expected_year_range:
            self.assertListEqual(year_range, expected_year_range, msg=f'Wrong year range {name}')
            self.assertEqual(len(year_range), len(evolution_df.columns) - 2, msg=f'Wrong n_steps {name}')
        else:
            self.assertIsNone(year_range, msg=f'Year range is not None when step is too large {name}')