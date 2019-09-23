import logging
import unittest

from pandas.util.testing import assert_frame_equal
from parameterized import parameterized

from models.keypaper.config import PubtrendsConfig
from models.keypaper.ss_loader import SemanticScholarLoader
from models.test.mock_database_loader import MockDatabaseLoader
from models.test.ss_articles import required_articles, extra_articles, required_citations, cit_stats_df, \
    cit_df, extra_citations, raw_cocitations_df, part_of_articles, expanded_articles


class TestSemanticScholarLoader(unittest.TestCase):
    loader = SemanticScholarLoader(pubtrends_config=PubtrendsConfig(test=True))

    @classmethod
    def setUpClass(cls):
        cls.loader.set_logger(logging.getLogger(__name__))

        # Text search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.ssid, required_articles))

        mock_database_loader = MockDatabaseLoader()
        mock_database_loader.init_semantic_scholar_database()
        mock_database_loader.insert_semantic_scholar_publications(required_articles + extra_articles)
        mock_database_loader.insert_semantic_scholar_citations(required_citations + extra_citations)

        # Get data via SemanticScholar methods
        cls.pub_df = cls.loader.load_publications(cls.ids)
        cls.cit_stats_df = cls.loader.load_citation_stats(cls.ids)
        cls.cit_df = cls.loader.load_citations(cls.ids)
        cls.cocit_df = cls.loader.load_cocitations(cls.ids)

    @parameterized.expand([
        ('limit 3, most recent', 3, 'year', ['5a63b4199bb58992882b0bf60bc1b1b3f392e5a5',
                                             '5451b1ef43678d473575bdfa7016d024146f2b53',
                                             'cad767094c2c4fff5206793fd8674a10e7fba3fe']),
        ('limit 3, most cited', 3, 'citations', ['3cf82f53a52867aaade081324dff65dd35b5b7eb',
                                                 'e7cdbddc7af4b6138227139d714df28e2090bd5f',
                                                 '5451b1ef43678d473575bdfa7016d024146f2b53']),
        ('limit 3, most relevant', 3, 'relevance', ['cad767094c2c4fff5206793fd8674a10e7fba3fe',
                                                    'e7cdbddc7af4b6138227139d714df28e2090bd5f',
                                                    '3cf82f53a52867aaade081324dff65dd35b5b7eb']),
    ])
    def test_search(self, name, limit, sort, expected):
        ids, _ = self.loader.search('find search', limit=limit, sort=sort)

        self.assertListEqual(ids, expected, name)

    def test_citations_stats_rows(self):
        expected_rows = cit_stats_df.shape[0]
        actual_rows = self.cit_stats_df.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in citations statistics is incorrect")

    def test_load_citation_stats_data_frame(self):
        assert_frame_equal(self.cit_stats_df.sort_values(by=['id', 'year']).reset_index(drop=True),
                           cit_stats_df,
                           "Citations statistics is incorrect")

    def test_load_citation_stats_null_year_is_ignored(self):
        expected_ignored_citations = 0
        citations = required_citations + extra_citations
        for article_out, article_in in citations:
            if article_out.year is None:
                expected_ignored_citations += 1

        ignored_diff = len(citations) - int(cit_stats_df['count'].sum())
        self.assertEqual(expected_ignored_citations, ignored_diff,
                         'Error with citations from articles with year NULL')

    def test_load_citations_count(self):
        self.assertEqual(len(self.cit_df), len(required_citations), 'Wrong number of citations')

    def test_load_citations_data_frame(self):
        assert_frame_equal(self.cit_df, cit_df, 'Wrong citation data')

    def test_load_cocitations_count(self):
        expected_rows = raw_cocitations_df.shape[0]
        actual_rows = self.cocit_df.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in co-citations dataframe is incorrect")

    def test_load_cocitations_data_frame(self):
        expected_cocit_df = raw_cocitations_df
        actual = self.cocit_df.sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(drop=True)
        assert_frame_equal(expected_cocit_df, actual, "Co-citations dataframe is incorrect")

    def test_expand(self):
        ids = list(map(lambda article: article.ssid, part_of_articles))
        expected = list(map(lambda article: article.ssid, expanded_articles))
        actual = self.loader.expand(ids)
        self.assertSequenceEqual(sorted(expected), sorted(actual), "Wrong list of expanded ids")

    @classmethod
    def tearDownClass(cls):
        query = '''
        drop table sscitations;
        drop table sspublications;'''
        with cls.loader.conn.cursor() as cursor:
            cursor.execute(query)

        cls.loader.close_connection()


if __name__ == "__main__":
    unittest.main()
