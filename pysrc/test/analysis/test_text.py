import unittest
import numpy as np

from pysrc.papers.analysis.graph import build_citation_graph
from pysrc.papers.analysis.text import tokenize
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.test.mock_loaders import MockLoader


class TestText(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        loader = MockLoader()
        cls.analyzer = PapersAnalyzer(loader, TestText.PUBTRENDS_CONFIG, test=True)
        ids = cls.analyzer.search_terms(query='query')
        cls.analyzer.TOPIC_MIN_SIZE = 0  # Disable merging for tests
        cls.analyzer.analyze_papers(ids, 'query')
        cls.analyzer.cit_df = cls.analyzer.loader.load_citations(cls.analyzer.df['id'])
        cls.analyzer.citations_graph = build_citation_graph(cls.analyzer.df, cls.analyzer.cit_df)
        cls.analyzer.bibliographic_coupling_df = loader.load_bibliographic_coupling(cls.analyzer.df['id'])

    def test_tokenizer(self):
        text = """Very interesting article about elephants and donkeys.
        There are two types of elephants - Indian and African.
        Both of them are really beautiful, but in my opinion Indian are even cuter"""
        # nouns and adjectives from text excluding comparative and superlative forms
        expected = ['interesting', 'article', 'elephant', 'donkey',
                    'type', 'elephant', 'indian', 'african', 'beautiful',
                    'opinion', 'indian']
        actual = tokenize(text)
        self.assertSequenceEqual(actual, expected)

    def test_corpus_vectorization(self):
        self.assertEqual(
            self.analyzer.corpus_terms,
            ['abstract', 'kw1', 'kw2', 'kw3', 'kw4', 'kw5', 'paper', 'term1', 'term2', 'term3', 'term4', 'term5']
        )
        print(self.analyzer.corpus_counts.toarray())
        self.assertTrue(np.array_equal(
            self.analyzer.corpus_counts.toarray(),
            [[0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
             [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
             [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
             [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
             [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1]]))
