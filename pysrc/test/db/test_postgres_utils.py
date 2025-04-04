import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from parameterized import parameterized

from pysrc.papers.db.postgres_utils import preprocess_search_query_for_postgres, \
    process_bibliographic_coupling_postgres, preprocess_quotes


class TestPostgresUtils(unittest.TestCase):

    @parameterized.expand([
        ('FooBar', 'foobar'),
        ('Foo Bar', 'foo & bar'),
        ('"Foo Bar"', 'foo<->bar'),
        ('Foo-Bar', 'foo-bar'),
        ('&^Foo-Bar', 'foo-bar'),
        ("Alzheimer's disease", 'alzheimer & disease'),
        ('Foo, Bar', 'foo | bar'),
        ('Foo, Bar Baz', 'foo | bar & baz'),
        ('Foo, "Bar Baz"', 'foo | bar<->baz'),
        (' Foo', 'foo'),
        ('Foo, "Bar Baz" Bzzzz, ', 'foo | bar<->baz & bzzzz'),
        ('Foo, "Bar Baz" Bzzzz, ', 'foo | bar<->baz & bzzzz'),
        ("user' or '1' == '1", 'user & or & 1 & 1'),
        ("foo | bar<->baz & bzzzz", 'foo & bar-baz & bzzzz'),
    ])
    def test_preprocess_search_valid_source(self, terms, expected):
        self.assertEqual(expected, preprocess_search_query_for_postgres(terms)[0])


    def test_preprocess_search_odd_quotes(self):
        with self.assertRaises(Exception):
            preprocess_search_query_for_postgres('"foo" "')

    def test_preprocess_search_empty_phrase(self):
        with self.assertRaises(Exception):
            preprocess_search_query_for_postgres('"" bar baz')

    def test_preprocess_search_or(self):
        self.assertEqual(
            'covid-19 | coronavirus | corona<->virus | 2019-ncov | sars-cov | mers-cov |'
            ' severe<->acute<->respiratory<->syndrome | middle<->east<->respiratory<->syndrome',
            preprocess_search_query_for_postgres(
                'COVID-19, Coronavirus, "Corona virus", 2019-nCoV, SARS-CoV, '
                'MERS-CoV, "Severe Acute Respiratory Syndrome", '
                '"Middle East Respiratory Syndrome"',
            )[0]
        )

    @parameterized.expand([
        ('"Corona virus"',
         "(P.title IS NOT NULL AND P.title ~* '(\mcorona\s+virus\M)' OR "
         "P.abstract IS NOT NULL AND P.abstract ~* '(\mcorona\s+virus\M)')"),
        ('COVID-19,  Respiratory Syndrome', ''),
        ('COVID-19, "Corona virus" , Respiratory Syndrome',
         "(P.title IS NOT NULL AND P.title ~* '(\mcorona\s+virus\M)' OR "
         "P.abstract IS NOT NULL AND P.abstract ~* '(\mcorona\s+virus\M)')")
    ])
    def test_no_stemming_filter(self, query, expected_phrase_filter):
        query, phrase_filter = preprocess_search_query_for_postgres(query)
        self.assertEqual(expected_phrase_filter, phrase_filter)

    def test_non_english_query(self):
        with self.assertRaises(Exception):
            self.assertEqual('ОЧЕНЬ & СТРАШНАЯ & БОЛЕЗНЬ',
                             preprocess_search_query_for_postgres('ОЧЕНЬ СТРАШНАЯ БОЛЕЗНЬ')[0])

    def test_preprocess_search_illegal_string(self):
        with self.assertRaises(Exception):
            preprocess_search_query_for_postgres('"Foo" Bar"')
        with self.assertRaises(Exception):
            preprocess_search_query_for_postgres('&&&')

    def test_preprocess_quotes(self):
        self.assertEqual(") or '1' == '1", preprocess_quotes("''') or '1' == ''1"))


    def test_process_bibliographic_coupling_empty(self):
        df = process_bibliographic_coupling_postgres([], [])
        self.assertTrue(len(df) == 0)
        self.assertTrue(len(df['total']) == 0)

    def test_process_bibliographic_coupling(self):
        df = process_bibliographic_coupling_postgres(
            ['1', '2', '3', '4', '5'],
            [
                ('1', ['2', '3', '4', '5']),
                ('2', ['3', '4', '5']),
                ('3', ['4', '5']),
                ('4', ['5'])
            ])
        # print(df)
        expected_df = pd.DataFrame([
            ['2', '3', 1],
            ['2', '4', 1],
            ['2', '5', 1],
            ['3', '4', 2],
            ['3', '5', 2],
            ['4', '5', 3],
        ], columns=['citing_1', 'citing_2', 'total'])
        expected_df['total'] = expected_df['total'].astype(np.int8)
        assert_frame_equal(expected_df, df)


if __name__ == '__main__':
    unittest.main()
