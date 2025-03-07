import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from parameterized import parameterized

from pysrc.papers.db.postgres_utils import preprocess_search_query_for_postgres, no_stemming_filter_for_phrases, \
    process_bibliographic_coupling_postgres


class TestPostgresUtils(unittest.TestCase):
    @parameterized.expand([
        ('FooBar', 'FooBar'),
        ('Foo Bar', 'Foo & Bar'),
        ('"Foo Bar"', 'Foo<->Bar'),
        ('Foo-Bar', 'Foo-Bar'),
        ('&^Foo-Bar', '&^Foo-Bar'),
        ("Alzheimer's disease", 'Alzheimer & disease'),
        ('Foo, Bar', 'Foo | Bar'),
        ('Foo, Bar Baz', 'Foo | Bar & Baz'),
        ('Foo, "Bar Baz"', 'Foo | Bar<->Baz'),
        (' Foo', 'Foo'),
        ('Foo, "Bar Baz" Bzzzz, ', 'Foo | Bar<->Baz & Bzzzz'),
    ])
    def test_preprocess_search_valid_source(self, terms, expected):
        self.assertEqual(expected, preprocess_search_query_for_postgres(terms, 0))

    def test_preprocess_search_too_few_words(self):
        self.assertEqual('Foo', preprocess_search_query_for_postgres('Foo', 1))
        with self.assertRaises(Exception):
            preprocess_search_query_for_postgres('Foo', 2)

    def test_preprocess_search_too_few_words_whitespaces(self):
        with self.assertRaises(Exception):
            preprocess_search_query_for_postgres('Foo  ', 2)

    def test_preprocess_search_too_few_words_stems(self):
        with self.assertRaises(Exception):
            preprocess_search_query_for_postgres('Humans Humanity', 2)

    def test_preprocess_search_dash_split(self):
        self.assertEqual('Covid-19', preprocess_search_query_for_postgres('Covid-19', 2))

    def test_preprocess_search_odd_quotes(self):
        with self.assertRaises(Exception):
            preprocess_search_query_for_postgres('"foo" "', 2)

    def test_preprocess_search_empty_phrase(self):
        with self.assertRaises(Exception):
            preprocess_search_query_for_postgres('"" bar baz', 2)

    def test_preprocess_search_or(self):
        self.assertEqual(
            'COVID-19 | Coronavirus | Corona<->virus | 2019-nCoV | SARS-CoV | MERS-CoV |'
            ' Severe<->Acute<->Respiratory<->Syndrome | Middle<->East<->Respiratory<->Syndrome',
            preprocess_search_query_for_postgres(
                'COVID-19, Coronavirus, "Corona virus", 2019-nCoV, SARS-CoV, '
                'MERS-CoV, "Severe Acute Respiratory Syndrome", '
                '"Middle East Respiratory Syndrome"',
                0
            )
        )

    @parameterized.expand([
        ('Corona<->virus',
         " AND ((P.title IS NOT NULL AND P.title ~* '(\mcorona\s+virus\M)' OR "
         "P.abstract IS NOT NULL AND P.abstract ~* '(\mcorona\s+virus\M)'))"),
        ('COVID-19 | Respiratory & Syndrome', ''),
        ('COVID-19 | Corona<->virus | Respiratory & Syndrome',
         " AND ((P.title IS NOT NULL AND P.title ~* '(\mcorona\s+virus\M)' OR "
         "P.abstract IS NOT NULL AND P.abstract ~* '(\mcorona\s+virus\M)'))"),
        ('COVID-19 | Corona<->virus & Respiratory & Syndrome',
         " AND ((P.title IS NOT NULL AND P.title ~* '(\mcorona\s+virus\M)' OR "
         "P.abstract IS NOT NULL AND P.abstract ~* '(\mcorona\s+virus\M)'))"),
        ('Corona<->virus & Respiratory<->Syndrome',
         " AND ((P.title IS NOT NULL AND P.title ~* '(\mcorona\s+virus\M)' OR "
         "P.abstract IS NOT NULL AND P.abstract ~* '(\mcorona\s+virus\M)') "
         "AND (P.title IS NOT NULL AND P.title ~* '(\mrespiratory\s+syndrome\M)' "
         "OR P.abstract IS NOT NULL AND P.abstract ~* '(\mrespiratory\s+syndrome\M)'))")
    ])
    def test_no_stemming_filter(self, query, phrase_filter):
        # print(no_stemming_filter_for_phrases(query))
        self.assertEqual(phrase_filter, no_stemming_filter_for_phrases(query))

    def test_non_english_query(self):
        self.assertEqual('ОЧЕНЬ & СТРАШНАЯ & БОЛЕЗНЬ',
                         preprocess_search_query_for_postgres('ОЧЕНЬ СТРАШНАЯ БОЛЕЗНЬ', 0))

    def test_preprocess_search_illegal_string(self):
        with self.assertRaises(Exception):
            preprocess_search_query_for_postgres('"Foo" Bar"', 2)
        with self.assertRaises(Exception):
            preprocess_search_query_for_postgres('&&&', 2)

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
