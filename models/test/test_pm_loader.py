import logging
import unittest

import numpy as np
from pandas.util.testing import assert_frame_equal
from parameterized import parameterized

from models.keypaper.config import PubtrendsConfig
from models.keypaper.pm_loader import PubmedLoader
from models.keypaper.utils import SORT_MOST_RECENT, SORT_MOST_RELEVANT, SORT_MOST_CITED
from models.test.mock_database_loader import MockDatabaseLoader
from models.test.pm_articles import REQUIRED_ARTICLES, ARTICLES, EXPECTED_PUB_DF, \
    INNER_CITATIONS, CITATIONS, EXPECTED_CIT_DF, EXPECTED_COCIT_DF, EXPECTED_CIT_STATS_DF, \
    EXPANDED_IDS, PART_OF_ARTICLES, EXPECTED_PUB_DF_GIVEN_IDS


class TestPubmedLoader(unittest.TestCase):
    loader = PubmedLoader(pubtrends_config=PubtrendsConfig(test=True))

    @classmethod
    def setUpClass(cls):
        cls.loader.set_logger(logging.getLogger(__name__))

        # Text search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.pmid, REQUIRED_ARTICLES))

        # Reset and load data to the test database
        mock_database_loader = MockDatabaseLoader()
        mock_database_loader.init_pubmed_database()
        mock_database_loader.insert_pubmed_publications(ARTICLES)
        mock_database_loader.insert_pubmed_citations(CITATIONS)

        # Get data via PubmedLoader methods
        cls.pub_df = cls.loader.load_publications(cls.ids)
        cls.cit_stats_df = cls.loader.load_citation_stats(cls.ids)
        cls.cit_df = cls.loader.load_citations(cls.ids)
        cls.cocit_df = cls.loader.load_cocitations(cls.ids)

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
        actual = list(map(lambda article: self.get_row(self.pub_df, article)['journal'].values[0], REQUIRED_ARTICLES))
        self.assertListEqual(actual, expected, msg='Wrong journals extracted')

    def test_load_publications_data_frame(self):
        assert_frame_equal(self.pub_df, EXPECTED_PUB_DF, 'Wrong publication data', check_like=True)

    @parameterized.expand([
        ('Article', 1, SORT_MOST_RECENT, ['5']),
        ('Abstract', 5, SORT_MOST_RELEVANT, ['2', '3']),
        ('Article', 1, SORT_MOST_CITED, ['4']),
    ])
    def test_search(self, query, limit, sort, expected_ids):
        # Use sorted to avoid ambiguity
        ids = sorted(self.loader.search(query, limit=limit, sort=sort))
        self.assertListEqual(ids, sorted(expected_ids), 'Wrong IDs of papers')

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

    def test_expand(self):
        expected = EXPANDED_IDS
        ids_list = list(map(lambda article: str(article.pmid), PART_OF_ARTICLES))
        actual = self.loader.expand(ids_list)
        print(expected)
        print(actual)
        self.assertSequenceEqual(sorted(expected), sorted(actual), "Wrong list of expanded ids")

    @classmethod
    def tearDownClass(cls):
        cls.loader.close_connection()


if __name__ == "__main__":
    unittest.main()
