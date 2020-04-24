import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal
from parameterized import parameterized

from models.keypaper.utils import tokenize, cut_authors_list, split_df_list, crc32, preprocess_search_query, \
    preprocess_doi, preprocess_pubmed_search_title, hex2rgb, rgb2hex


class TestUtils(unittest.TestCase):
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

    def test_cut_authors_list_limit_size(self):
        limit_size_list = "first, second, third"
        actual = cut_authors_list(limit_size_list, limit=len(limit_size_list))
        self.assertEqual(actual, limit_size_list)

    def test_cut_authors_list_less(self):
        long_list = "first, second, third"
        actual = cut_authors_list(long_list, 2)
        expected = "first,...,third"
        self.assertEqual(actual, expected)

    def test_cut_authors_list_greater(self):
        short_list = "first, second, third"
        actual = cut_authors_list(short_list, 4)
        self.assertEqual(actual, short_list)

    def test_split_df_list(self):
        data_for_df = [[2, 'a, b, c'],
                       [1, 'c, a, d'],
                       [4, 'd, c']]

        df_with_list_column = pd.DataFrame(data_for_df, columns=['id', 'list'])

        expected_data = [[2, 'a'], [2, 'b'], [2, 'c'],
                         [1, 'c'], [1, 'a'], [1, 'd'],
                         [4, 'd'], [4, 'c']]
        expected_df = pd.DataFrame(expected_data, columns=['id', 'list'])
        actual_df = split_df_list(df_with_list_column, target_column='list', separator=', ')
        assert_frame_equal(expected_df, actual_df, "Splitting list into several rows works incorrectly")

    @parameterized.expand([
        ('cc77a65ff80a9d060e48461603bcf06bb0ef9294', -189727251, 'negative'),
        ('6d8484217c9fa02419536c9118435715d3a8e705', 1979136599, 'positive')
    ])
    def test_crc32(self, ssid, crc32id, case):
        self.assertEqual(crc32(ssid), crc32id, f"Hashed id is wrong ({case} case)")

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
        self.assertEqual(expected, preprocess_search_query(terms, 0))

    def test_preprocess_search_too_few_words(self):
        self.assertEqual('Foo', preprocess_search_query('Foo', 1))
        with self.assertRaises(Exception):
            preprocess_search_query('Foo', 2)

    def test_preprocess_search_too_few_words_whitespaces(self):
        with self.assertRaises(Exception):
            preprocess_search_query('Foo  ', 2)

    def test_preprocess_search_too_few_words_stems(self):
        with self.assertRaises(Exception):
            preprocess_search_query('Humans Humanity', 2)

    def test_preprocess_search_dash_split(self):
        self.assertEqual('"Covid-19"', preprocess_search_query('Covid-19', 2))

    def test_preprocess_search_or(self):
        self.assertEqual(
            '"COVID-19" OR Coronavirus OR "Corona virus" OR "2019-nCoV" OR "SARS-CoV" OR "MERS-CoV" OR '
            '"Severe Acute Respiratory Syndrome" OR "Middle East Respiratory Syndrome"',
            preprocess_search_query('COVID-19, Coronavirus, "Corona virus", 2019-nCoV, SARS-CoV, '
                                    'MERS-CoV, "Severe Acute Respiratory Syndrome", '
                                    '"Middle East Respiratory Syndrome"', 0)
        )

    def test_preprocess_search_illegal_string(self):
        with self.assertRaises(Exception):
            preprocess_search_query('"Foo" Bar"', 2)
        with self.assertRaises(Exception):
            preprocess_search_query('&&&', 2)

    @parameterized.expand([
        ('dx.doi.org prefix', 'http://dx.doi.org/10.1037/a0028240', '10.1037/a0028240'),
        ('doi.org prefix', 'http://doi.org/10.3352/jeehp.2013.10.3', '10.3352/jeehp.2013.10.3'),
        ('no changes', '10.1037/a0028240', '10.1037/a0028240')
    ])
    def test_preprocess_doi(self, case, doi, expected):
        self.assertEqual(preprocess_doi(doi), expected, case)

    def test_preprocess_pubmed_search_title(self):
        title = '[DNA methylation age.]'
        expected = 'DNA methylation age'
        self.assertEqual(preprocess_pubmed_search_title(title), expected)

    def test_hex2rgb(self, color, expected):
        self.assertEqual(hex2rgb(color), expected)

    @parameterized.expand([
        ([145, 200, 47], '#91c82f'),
        ([143, 254, 9], '#8ffe09'),
        ('red', '#ff0000'),
        ('blue', '#0000ff')
    ])
    def test_color2hex(self, color, expected):
        self.assertEqual(rgb2hex(color), expected)
