import unittest

from parameterized import parameterized

from pysrc.papers.db.neo4j_utils import preprocess_search_query_for_neo4j


class TestNeo4jUtils(unittest.TestCase):
    @parameterized.expand([
        ('FooBar', 'FooBar'),
        ('Foo Bar', 'Foo AND Bar'),
        ('"Foo Bar"', '"Foo Bar"'),
        ('Foo-Bar', '"Foo-Bar"'),
        ('&^Foo-Bar', '"Foo-Bar"'),
        ("Alzheimer's disease", 'Alzheimer AND disease'),
        ('Foo, Bar', 'Foo OR Bar'),
        ('Foo, Bar Baz', 'Foo OR (Bar AND Baz)'),
        ('Foo, "Bar Baz"', 'Foo OR "Bar Baz"'),
    ])
    def test_preprocess_search_valid_source(self, terms, expected):
        self.assertEqual(expected, preprocess_search_query_for_neo4j(terms, 0))

    def test_preprocess_search_too_few_words(self):
        self.assertEqual('Foo', preprocess_search_query_for_neo4j('Foo', 1))
        with self.assertRaises(Exception):
            preprocess_search_query_for_neo4j('Foo', 2)

    def test_preprocess_search_too_few_words_whitespaces(self):
        with self.assertRaises(Exception):
            preprocess_search_query_for_neo4j('Foo  ', 2)

    def test_preprocess_search_too_few_words_stems(self):
        with self.assertRaises(Exception):
            preprocess_search_query_for_neo4j('Humans Humanity', 2)

    def test_preprocess_search_dash_split(self):
        self.assertEqual('"Covid-19"', preprocess_search_query_for_neo4j('Covid-19', 2))

    def test_preprocess_search_or(self):
        self.assertEqual(
            '"COVID-19" OR Coronavirus OR "Corona virus" OR "2019-nCoV" OR "SARS-CoV" OR "MERS-CoV" OR '
            '"Severe Acute Respiratory Syndrome" OR "Middle East Respiratory Syndrome"',
            preprocess_search_query_for_neo4j(
                'COVID-19, Coronavirus, "Corona virus", 2019-nCoV, SARS-CoV, '
                'MERS-CoV, "Severe Acute Respiratory Syndrome", '
                '"Middle East Respiratory Syndrome"',
                0
            )
        )

    def test_preprocess_search_illegal_string(self):
        with self.assertRaises(Exception):
            preprocess_search_query_for_neo4j('"Foo" Bar"', 2)
        with self.assertRaises(Exception):
            preprocess_search_query_for_neo4j('&&&', 2)
