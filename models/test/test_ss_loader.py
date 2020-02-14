import logging
import unittest

from pandas.util.testing import assert_frame_equal
from parameterized import parameterized

from models.keypaper.config import PubtrendsConfig
from models.keypaper.ss_loader import SemanticScholarLoader
from models.keypaper.utils import SORT_MOST_RECENT, SORT_MOST_CITED, SORT_MOST_RELEVANT
from models.test.ss_database_supplier import SSTestDatabaseSupplier
from models.test.ss_database_articles import required_articles, extra_articles, required_citations, \
    expected_cit_stats_df, expected_cit_df, extra_citations, expected_cocit_df, part_of_articles, expanded_articles


class TestSemanticScholarLoader(unittest.TestCase):
    loader = SemanticScholarLoader(pubtrends_config=PubtrendsConfig(test=True))

    @classmethod
    def setUpClass(cls):
        cls.loader.set_progress(logging.getLogger(__name__))

        # Text search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.ssid, required_articles))

        supplier = SSTestDatabaseSupplier()
        supplier.init_semantic_scholar_database()
        supplier.insert_semantic_scholar_publications(required_articles + extra_articles)
        supplier.insert_semantic_scholar_citations(required_citations + extra_citations)

        # Get data via SemanticScholar methods
        cls.pub_df = cls.loader.load_publications(cls.ids)
        cls.cit_stats_df = cls.loader.load_citation_stats(cls.ids)
        cls.cit_df = cls.loader.load_citations(cls.ids)
        cls.cocit_df = cls.loader.load_cocitations(cls.ids)

    @parameterized.expand([
        ('limit 3, most recent', 3, SORT_MOST_RECENT, ['5a63b4199bb58992882b0bf60bc1b1b3f392e5a5',
                                                       '5451b1ef43678d473575bdfa7016d024146f2b53',
                                                       'cad767094c2c4fff5206793fd8674a10e7fba3fe']),
        ('limit 3, most cited', 3, SORT_MOST_CITED, ['3cf82f53a52867aaade081324dff65dd35b5b7eb',
                                                     'e7cdbddc7af4b6138227139d714df28e2090bd5f',
                                                     '5451b1ef43678d473575bdfa7016d024146f2b53']),
        ('limit 3, most relevant', 3, SORT_MOST_RELEVANT, ['cad767094c2c4fff5206793fd8674a10e7fba3fe',
                                                           'e7cdbddc7af4b6138227139d714df28e2090bd5f',
                                                           '3cf82f53a52867aaade081324dff65dd35b5b7eb']),
    ])
    def test_search(self, name, limit, sort, expected):
        # Use sorted to avoid ambiguity
        ids = self.loader.search('find search', limit=limit, sort=sort)
        self.assertListEqual(sorted(expected), sorted(ids), name)

    def test_citations_stats_rows(self):
        expected_rows = expected_cit_stats_df.shape[0]
        actual_rows = self.cit_stats_df.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in citations statistics is incorrect")

    def test_load_citation_stats_data_frame(self):
        assert_frame_equal(expected_cit_stats_df,
                           self.cit_stats_df.sort_values(by=['id', 'year']).reset_index(drop=True),
                           "Citations statistics is incorrect",
                           check_like=True)

    def test_load_citations_count(self):
        self.assertEqual(len(required_citations), len(self.cit_df), 'Wrong number of citations')

    def test_load_citations_data_frame(self):
        assert_frame_equal(expected_cit_df, self.cit_df, 'Wrong citation data', check_like=True)

    def test_load_cocitations_count(self):
        expected_rows = expected_cocit_df.shape[0]
        actual_rows = self.cocit_df.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in co-citations dataframe is incorrect")

    def test_load_cocitations_data_frame(self):
        actual = self.cocit_df.sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(drop=True)
        print(expected_cocit_df)
        print(actual)
        assert_frame_equal(expected_cocit_df, actual, "Co-citations dataframe is incorrect", check_like=True)

    def test_expand(self):
        ids = list(map(lambda article: article.ssid, part_of_articles))
        expected = list(map(lambda article: article.ssid, expanded_articles))
        actual = self.loader.expand(ids)
        self.assertSequenceEqual(sorted(expected), sorted(actual), "Wrong list of expanded ids")


if __name__ == "__main__":
    unittest.main()
