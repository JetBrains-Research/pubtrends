import unittest

import numpy as np
from parameterized import parameterized

from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.keypaper.pm_loader import PubmedLoader
from models.keypaper.ss_loader import SemanticScholarLoader
from models.test.mock_loader import MockLoader, COCITATION_GRAPH_EDGES, COCITATION_GRAPH_NODES, \
    CITATION_YEARS, EXPECTED_MAX_GAIN, EXPECTED_MAX_RELATIVE_GAIN, CITATION_GRAPH_NODES, CITATION_GRAPH_EDGES, \
    MockLoaderEmpty


class TestKeyPaperAnalyzerEmpty(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = KeyPaperAnalyzer(MockLoaderEmpty(), TestKeyPaperAnalyzerEmpty.PUBTRENDS_CONFIG, test=True)

    def test_setup(self):
        with self.assertRaises(Exception):
            self.analyzer.search_terms(query='query')


if __name__ == '__main__':
    unittest.main()
