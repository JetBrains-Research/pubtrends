import unittest

import numpy as np
from parameterized import parameterized

from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.keypaper.loader import Loader
from models.keypaper.pm_loader import PubmedLoader
from models.keypaper.ss_loader import SemanticScholarLoader
from models.test.mock_loader import MockLoader, COCITATION_GRAPH_EDGES, COCITATION_GRAPH_NODES, \
    CITATION_YEARS, EXPECTED_MAX_GAIN, EXPECTED_MAX_RELATIVE_GAIN, CITATION_GRAPH_NODES, CITATION_GRAPH_EDGES


class TestLoader(unittest.TestCase):

    @parameterized.expand([
        ('FooBar', '"\'FooBar\'"'),
        ('Foo Bar', '"\'Foo\' AND \'Bar\'"'),
        ('"Foo Bar"', '\'"Foo Bar"\''),
        ('"Foo" Bar"', '\'"Foo Bar"\''),
        ('Foo-Bar', '"\'Foo-Bar\'"'),
        ('&^Foo-Bar', '"\'Foo-Bar\'"'),
    ])
    def test_valid_source(self, terms, expected):
        self.assertEqual(expected, Loader.preprocess_search_string(terms, 0))

    def test_too_many_words(self):
        self.assertEqual('"\'Foo\'"', Loader.preprocess_search_string('Foo', 1))
        with self.assertRaises(Exception):
            self.assertEqual(Loader.preprocess_search_string('Foo', 2), '')


if __name__ == '__main__':
    unittest.main()
