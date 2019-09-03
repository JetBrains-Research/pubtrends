import logging
import re
import unittest

from pandas.util.testing import assert_frame_equal
from parameterized import parameterized

from models.keypaper.config import PubtrendsConfig
from models.keypaper.ss_loader import SemanticScholarLoader
from models.test.mock_database_loader import MockDatabaseLoader
from models.test.ss_articles import required_articles, extra_articles, required_citations, cit_stats_df, \
    pub_df, cit_df, extra_citations, raw_cocitations_df, part_of_articles, pub_df_given_ids, expanded_articles


class TestSemanticScholarLoader(unittest.TestCase):
    VALUES_REGEX = re.compile(r'\$VALUES\$')
    loader = None

    @classmethod
    def setUpClass(cls):
        cls.loader = SemanticScholarLoader(pubtrends_config=PubtrendsConfig(test=True))
        cls.loader.set_logger(logging.getLogger(__name__))

        mock_database_loader = MockDatabaseLoader(PubtrendsConfig(test=True))
        mock_database_loader.init_semantic_scholar_database()
        mock_database_loader.insert_semantic_scholar_publications(required_articles + extra_articles)
        mock_database_loader.insert_semantic_scholar_citations(required_citations + extra_citations)
        cls._load_publications()

        cls.citations_stats = cls._load_citations_stats()
        cls.citations = cls.loader.load_citations()
        cls.cocitations_df = cls._load_cocitations()

    @classmethod
    def _load_publications(cls):
        values = ', '.join(map(lambda article: article.indexes(), required_articles))
        query = re.sub(cls.VALUES_REGEX, values, '''
                DROP TABLE IF EXISTS temp_ssids;
                WITH vals(ssid, crc32id) AS (VALUES $VALUES$)
                SELECT crc32id, ssid INTO temporary table temp_ssids FROM vals;
                DROP INDEX IF EXISTS temp_ssids_index;
                CREATE INDEX temp_ssids_index ON temp_ssids USING btree (crc32id);''')

        with cls.loader.conn.cursor() as cursor:
            cursor.execute(query)
            cls.loader.conn.commit()

        cls.loader.ids = list(map(lambda article: article.ssid, required_articles))
        cls.loader.crc32ids = list(map(lambda article: article.crc32id, required_articles))
        cls.loader.values = ', '.join(
            ['({0}, \'{1}\')'.format(i, j) for (i, j) in
             zip(cls.loader.crc32ids, cls.loader.ids)])

        cls.loader.pub_df = pub_df

    @classmethod
    def _load_citations_stats(cls):
        cit_stats_df_from_query = cls.loader.load_citation_stats()
        return cit_stats_df_from_query.sort_values(by=['id', 'year']).reset_index(drop=True)

    @classmethod
    def _load_cocitations(cls):
        cocitations = cls.loader.load_cocitations()
        return cocitations.sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(drop=True)

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

        self.assertListEqual(ids, expected)

    def test_citations_stats_rows(self):
        expected_rows = cit_stats_df.shape[0]
        actual_rows = self.citations_stats.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in citations statistics is incorrect")

    def test_load_citation_stats_data_frame(self):
        assert_frame_equal(self.citations_stats, cit_stats_df, "Citations statistics is incorrect")

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
        self.assertEqual(len(self.citations), len(required_citations), 'Wrong number of citations')

    def test_load_citations_data_frame(self):
        assert_frame_equal(self.citations, cit_df, 'Wrong citation data')

    def test_load_cocitations_count(self):
        expected_rows = raw_cocitations_df.shape[0]
        actual_rows = self.cocitations_df.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in co-citations dataframe is incorrect")

    def test_load_cocitations_data_frame(self):
        expected_cocit_df = raw_cocitations_df
        actual = self.cocitations_df
        assert_frame_equal(expected_cocit_df, actual, "Co-citations dataframe is incorrect")

    def test_search_with_given_ids(self):
        initial_columns = ['abstract', 'aux', 'crc32id', 'id', 'title', 'year']
        ids_list = list(map(lambda article: article.ssid, part_of_articles))
        expected = pub_df_given_ids
        actual_pub_given_ids = self.loader.search_with_given_ids(ids_list)
        assert_frame_equal(expected, actual_pub_given_ids[initial_columns], "Wrong publications extracted")

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
