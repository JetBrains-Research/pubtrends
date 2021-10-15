import unittest

import numpy as np
from pandas.testing import assert_frame_equal
from parameterized import parameterized

from pysrc.papers.db.article import AuxInfo, Author, Journal
from pysrc.papers.db.pm_article import PubmedArticle
from pysrc.papers.utils import SORT_MOST_RECENT, SORT_MOST_CITED
from pysrc.test.db.pm_test_articles import EXPECTED_PUB_DF, \
    EXPECTED_CIT_STATS_DF, INNER_CITATIONS, EXPECTED_CIT_DF, EXPECTED_COCIT_DF, EXPECTED_PUB_DF_GIVEN_IDS, \
    PART_OF_ARTICLES, EXPANDED_IDS_DF, EXPANDED_TOP_CITED_3_DF, EXPANDED_TOP_CITED_4_DF

from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.pm_postgres_loader import PubmedPostgresLoader
from pysrc.papers.db.pm_postgres_writer import PubmedPostgresWriter
from pysrc.test.db.pm_test_articles import REQUIRED_ARTICLES, ARTICLES, CITATIONS


class TestPubmedPostgresLoader(unittest.TestCase):
    test_config = PubtrendsConfig(test=True)
    loader = PubmedPostgresLoader(test_config)

    @classmethod
    def setUpClass(cls):
        cls.loader = TestPubmedPostgresLoader.loader

        # Text search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.pmid, REQUIRED_ARTICLES))

        # Reset and load data to the test database
        writer = PubmedPostgresWriter(config=TestPubmedPostgresLoader.test_config)
        writer.init_pubmed_database()
        writer.insert_pubmed_publications(ARTICLES)
        writer.insert_pubmed_publications([
            PubmedArticle(pmid=16960519,
                          title='Comparison of the 2001 BRFSS and the IPAQ Physical Activity Questionnaires',
                          abstract='''
Purpose: ... (BRFSS) physical activity module ...
Results: ... for the lowest category (inactive) by ...
''',
                          year=2006,
                          doi='10.1249/01.mss.0000229457.73333.9a',
                          aux=AuxInfo(
                              authors=[Author(name='Barbara E Ainsworth')],
                              journal=Journal(name='Med Sci Sports Exerc.')))
        ])
        writer.insert_pubmed_citations(CITATIONS)

        # Get data via loader methods
        cls.pub_df = cls.loader.load_publications(cls.ids)
        cls.cit_stats_df = cls.loader.load_citations_by_year(cls.ids)
        cls.cit_df = cls.loader.load_citations(cls.ids)
        cls.cocit_df = cls.loader.load_cocitations(cls.ids)

    @classmethod
    def tearDownClass(cls):
        cls.loader.close_connection()

    @staticmethod
    def get_row(df, article):
        return df[df['id'] == str(article.pmid)]

    def test_load_publications_count(self):
        self.assertEqual(len(self.pub_df), len(self.ids))

    def test_load_publications_check_ids(self):
        self.assertCountEqual(list(self.pub_df['id'].values), list(map(str, self.ids)))

    def test_load_publications_fill_null_abstract(self):
        self.assertFalse(np.any(self.pub_df['abstract'].isna()), msg='NaN abstract found')
        for article in REQUIRED_ARTICLES:
            if article.abstract is None:
                row = self.pub_df[self.pub_df['id'] == str(article.pmid)]
                self.assertEqual(row['abstract'].values[0], '', msg='Null abstract was filled with wrong value')

    def test_load_publications_authors(self):
        actual = []
        expected = []

        for article in REQUIRED_ARTICLES:
            expected_names = [author.name for author in article.aux.authors]
            expected.append(', '.join(expected_names))

            actual_names = self.get_row(self.pub_df, article)['authors'].values[0]
            actual.append(actual_names)

        self.assertListEqual(actual, expected, msg='Wrong authors extracted')

    def test_load_publications_journals(self):
        expected = list(map(lambda article: article.aux.journal.name, REQUIRED_ARTICLES))
        actual = list(map(lambda article: self.get_row(self.pub_df, article)['journal'].values[0],
                          REQUIRED_ARTICLES))
        self.assertListEqual(actual, expected, msg='Wrong journals extracted')

    def test_load_publications_data_frame(self):
        assert_frame_equal(self.pub_df, EXPECTED_PUB_DF, 'Wrong publication data', check_like=True)

    @parameterized.expand([
        ('Article', 1, SORT_MOST_RECENT, ['5']),
        ('Article', 1, SORT_MOST_CITED, ['4']),
    ])
    def test_search(self, query, limit, sort, expected_ids):
        ids = self.loader.search(query, limit=limit, sort=sort)
        self.assertListEqual(expected_ids, ids, 'Wrong IDs of papers')

    def test_search_wo_order(self):
        ids = self.loader.search('Article', limit=10, sort=SORT_MOST_CITED)
        self.assertListEqual(sorted(['1', '10', '2', '3', '4', '5', '7', '8', '9']),
                             sorted(ids), 'Wrong IDs of papers')

    def test_search_noreviews(self):
        # Use sorted to avoid ambiguity
        ids = self.loader.search('Article', limit=10, sort=SORT_MOST_RECENT)
        self.assertListEqual(sorted(ids), ['1', '10', '2', '3', '4', '5', '7', '8', '9'], 'Wrong IDs of papers')

        idswithreview = self.loader.search('Article', limit=10, sort=SORT_MOST_RECENT, noreviews=False)
        self.assertListEqual(sorted(idswithreview), ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9'],
                             'Wrong IDs of papers')

    def test_search_phrase(self):
        self.assertListEqual(['16960519'], self.loader.search('"activity module"', limit=100, sort=SORT_MOST_CITED),
                             'Wrong IDs of papers')
        # word inactive won't be matched
        self.assertListEqual([], self.loader.search('"active module"', limit=100, sort=SORT_MOST_CITED),
                             'Wrong IDs of papers')

    def test_load_citation_stats_data_frame(self):
        # Sort to compare with expected
        actual = self.cit_stats_df.sort_values(by=['id', 'year']).reset_index(drop=True)
        assert_frame_equal(EXPECTED_CIT_STATS_DF, actual, 'Wrong citation stats data', check_like=True)

    def test_load_citations_count(self):
        self.assertEqual(len(self.cit_df), len(INNER_CITATIONS), 'Wrong number of citations')

    def test_load_citations_data_frame(self):
        assert_frame_equal(self.cit_df, EXPECTED_CIT_DF, 'Wrong citation data', check_like=True)

    def test_load_cocitations_data_frame(self):
        expected = EXPECTED_COCIT_DF.sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(drop=True)
        actual = self.cocit_df.sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(drop=True)
        # Sort for comparison
        assert_frame_equal(expected, actual, 'Wrong co-citation data', check_like=True)

    def test_search_with_given_ids(self):
        ids_list = list(map(lambda article: article.pmid, PART_OF_ARTICLES))
        assert_frame_equal(EXPECTED_PUB_DF_GIVEN_IDS, self.loader.load_publications(ids_list),
                           "Wrong publications extracted", check_like=True)

    @parameterized.expand([
        ('1', 1, []),
        ('2', 10, ['1']),
        ('3', 10, ['2']),
        ('4', 10, ['3']),
        ('7', 1, ['1']),
        ('7', 2, ['1', '3']),
        ('7', 10, ['1', '3', '2']),
    ])
    def test_load_references(self, pid, limit, expected_ids):
        self.assertEqual(self.loader.load_references(pid, limit), expected_ids)

    def test_expand(self):
        ids_list = list(map(lambda article: str(article.pmid), PART_OF_ARTICLES))
        actual = self.loader.expand(ids_list, 1000)
        assert_frame_equal(
            EXPANDED_IDS_DF,
            actual.sort_values(by=['total', 'id']).reset_index(drop=True),
            "Wrong list of expanded ids"
        )

    def test_expand_limited(self):
        ids_list = list(map(lambda article: str(article.pmid), PART_OF_ARTICLES))
        actual = self.loader.expand(ids_list, 3)
        assert_frame_equal(
            EXPANDED_TOP_CITED_3_DF,
            actual.sort_values(by=['total', 'id']).reset_index(drop=True),
            "Wrong list of expanded 3 ids"
        )
        actual = self.loader.expand(ids_list, 4)
        assert_frame_equal(
            EXPANDED_TOP_CITED_4_DF,
            actual.sort_values(by=['total', 'id']).reset_index(drop=True),
            "Wrong list of expanded 4 ids"
        )

    @parameterized.expand([
        ('id search', 'id', '1', ['1']),
        ('id search - with spaces', 'id', '  1  ', ['1']),
        ('title search - lower case', 'title', 'article title 2', ['2']),
        ('title search - special characters', 'title', '[article title 3.]', ['3']),
        ('title search - Title Case', 'title', 'Article Title 4', ['4']),
        ('title search - with spaces', 'title', '       Article Title 4        ', ['4']),
        ('dx.doi.org search', 'doi', 'http://dx.doi.org/10.000/0000', ['1']),
        ('doi.org search', 'doi', 'http://doi.org/10.000/0000', ['1']),
        ('doi search', 'doi', '10.000/0000', ['1']),
        ('doi with spaces', 'doi', '      10.000/0000       ', ['1']),
    ])
    def test_find_match(self, case, key, value, expected):
        actual = self.loader.find(key, value)
        self.assertListEqual(sorted(actual), sorted(expected), case)

    @parameterized.expand([
        ('no such id', 'id', '0'),
        ('abstract words in query', 'title', 'abstract'),
        ('no such title', 'title', 'Article Title 0'),
        ('no such doi', 'doi', '10.000/0001')
    ])
    def test_find_no_match(self, case, key, value):
        actual = self.loader.find(key, value)
        self.assertTrue(len(actual) == 0, case)

    def test_find_raise_pmid_not_integer(self):
        with self.assertRaises(Exception):
            self.loader.find('id', 'abc')


if __name__ == "__main__":
    unittest.main()
