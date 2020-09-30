from abc import ABCMeta, abstractmethod

import numpy as np
from pandas.testing import assert_frame_equal
from parameterized import parameterized

from pysrc.papers.utils import SORT_MOST_RECENT, SORT_MOST_RELEVANT, SORT_MOST_CITED
from pysrc.test.db.pm_test_articles import REQUIRED_ARTICLES, EXPECTED_PUB_DF, \
    EXPECTED_CIT_STATS_DF, INNER_CITATIONS, EXPECTED_CIT_DF, EXPECTED_COCIT_DF, EXPECTED_PUB_DF_GIVEN_IDS, \
    PART_OF_ARTICLES, EXPANDED_IDS, EXPANDED_TOP_CITED_3, EXPANDED_TOP_CITED_4


# Don't make it subclass of unittest.TestCase to avoid tests execution
class AbstractTestPubmedLoader(metaclass=ABCMeta):

    # @classmethod
    # def setUpClass(cls):
    # TODO: example of initialization
    #
    #     cls.loader = Loader(config=PubtrendsConfig(test=True))
    #     cls.loader.set_progress(logging.getLogger(__name__))
    #
    #     # Text search is not tested, imitating search results
    #     cls.ids = list(map(lambda article: article.pmid, REQUIRED_ARTICLES))
    #
    #     # Reset and load data to the test database
    #     writer = Writer()
    #     writer.init_database()
    #     writer.insert_publications(ARTICLES)
    #     writer.insert_citations(CITATIONS)
    #
    #     # Get data via loader methods
    #     cls.pub_df = cls.loader.load_publications(cls.ids)
    #     cls.cit_stats_df = cls.loader.load_citation_stats(cls.ids)
    #     cls.cit_df = cls.loader.load_citations(cls.ids)
    #     cls.cocit_df = cls.loader.load_cocitations(cls.ids)

    @abstractmethod
    def getLoader(self):
        """:return Loader instance"""

    @abstractmethod
    def getIds(self):
        """:return list of string ids"""

    @abstractmethod
    def getPublicationsDataframe(self):
        """:return Publications pandas dataframe"""

    @abstractmethod
    def getCitationsStatsDataframe(self):
        """:return citations stats pandas dataframe"""

    @abstractmethod
    def getCitationsDataframe(self):
        """:return citations dataframe"""

    @abstractmethod
    def getCoCitationsDataframe(self):
        """:return co-citations dataframe"""

    @staticmethod
    def get_row(df, article):
        return df[df['id'] == str(article.pmid)]

    def test_load_publications_count(self):
        self.assertEqual(len(self.getPublicationsDataframe()), len(self.getIds()))

    def test_load_publications_check_ids(self):
        self.assertCountEqual(list(self.getPublicationsDataframe()['id'].values), list(map(str, self.getIds())))

    def test_load_publications_fill_null_abstract(self):
        self.assertFalse(np.any(self.getPublicationsDataframe()['abstract'].isna()), msg='NaN abstract found')
        for article in REQUIRED_ARTICLES:
            if article.abstract is None:
                row = self.getPublicationsDataframe()[self.getPublicationsDataframe()['id'] == str(article.pmid)]
                self.assertEqual(row['abstract'].values[0], '', msg='Null abstract was filled with wrong value')

    def test_load_publications_authors(self):
        actual = []
        expected = []

        for article in REQUIRED_ARTICLES:
            expected_names = [author.name for author in article.aux.authors]
            expected.append(', '.join(expected_names))

            actual_names = self.get_row(self.getPublicationsDataframe(), article)['authors'].values[0]
            actual.append(actual_names)

        self.assertListEqual(actual, expected, msg='Wrong authors extracted')

    def test_load_publications_journals(self):
        expected = list(map(lambda article: article.aux.journal.name, REQUIRED_ARTICLES))
        actual = list(map(lambda article: self.get_row(self.getPublicationsDataframe(), article)['journal'].values[0],
                          REQUIRED_ARTICLES))
        self.assertListEqual(actual, expected, msg='Wrong journals extracted')

    def test_load_publications_data_frame(self):
        assert_frame_equal(self.getPublicationsDataframe(), EXPECTED_PUB_DF, 'Wrong publication data', check_like=True)

    @parameterized.expand([
        ('Article', 1, SORT_MOST_RECENT, ['5']),
        ('Abstract', 5, SORT_MOST_RELEVANT, ['2', '3']),
        ('Article', 1, SORT_MOST_CITED, ['4']),
        ('Article', 10, SORT_MOST_CITED, ['1', '10', '2', '3', '4', '5', '7', '8', '9']),
    ])
    def test_search(self, query, limit, sort, expected_ids):
        # Use sorted to avoid ambiguity
        ids = self.getLoader().search(query, limit=limit, sort=sort)
        self.assertListEqual(sorted(expected_ids), sorted(ids), 'Wrong IDs of papers')

    def test_search_noreviews(self):
        # Use sorted to avoid ambiguity
        ids = self.getLoader().search('Article', limit=10, sort=SORT_MOST_RECENT)
        self.assertListEqual(sorted(ids), ['1', '10', '2', '3', '4', '5', '7', '8', '9'], 'Wrong IDs of papers')

        idswithreview = self.getLoader().search('Article', limit=10, sort=SORT_MOST_RECENT, noreviews=False)
        self.assertListEqual(sorted(idswithreview), ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9'],
                             'Wrong IDs of papers')

    def test_load_citation_stats_data_frame(self):
        # Sort to compare with expected
        actual = self.getCitationsStatsDataframe().sort_values(by=['id', 'year']).reset_index(drop=True)
        assert_frame_equal(EXPECTED_CIT_STATS_DF, actual, 'Wrong citation stats data', check_like=True)

    def test_load_citations_count(self):
        self.assertEqual(len(self.getCitationsDataframe()), len(INNER_CITATIONS), 'Wrong number of citations')

    def test_load_citations_data_frame(self):
        assert_frame_equal(self.getCitationsDataframe(), EXPECTED_CIT_DF, 'Wrong citation data', check_like=True)

    def test_load_cocitations_data_frame(self):
        expected = EXPECTED_COCIT_DF.sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(drop=True)
        actual = self.getCoCitationsDataframe().sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(drop=True)
        # Sort for comparison
        assert_frame_equal(expected, actual, 'Wrong co-citation data', check_like=True)

    def test_search_with_given_ids(self):
        ids_list = list(map(lambda article: article.pmid, PART_OF_ARTICLES))
        assert_frame_equal(EXPECTED_PUB_DF_GIVEN_IDS, self.getLoader().load_publications(ids_list),
                           "Wrong publications extracted", check_like=True)

    def test_expand(self):
        ids_list = list(map(lambda article: str(article.pmid), PART_OF_ARTICLES))
        actual = self.getLoader().expand(ids_list, 1000)
        self.assertSequenceEqual(sorted(EXPANDED_IDS), sorted(actual), "Wrong list of expanded ids")

    def test_expand_limited(self):
        ids_list = list(map(lambda article: str(article.pmid), PART_OF_ARTICLES))
        actual = self.getLoader().expand(ids_list, 3)
        self.assertSequenceEqual(sorted(EXPANDED_TOP_CITED_3), sorted(actual), "Wrong list of expanded 3 ids")
        actual = self.getLoader().expand(ids_list, 4)
        self.assertSequenceEqual(sorted(EXPANDED_TOP_CITED_4), sorted(actual), "Wrong list of expanded 4 ids")


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
        actual = self.getLoader().find(key, value)
        self.assertListEqual(sorted(actual), sorted(expected), case)

    @parameterized.expand([
        ('no such id', 'id', '0'),
        ('abstract words in query', 'title', 'abstract'),
        ('no such title', 'title', 'Article Title 0'),
        ('no such doi', 'doi', '10.000/0001')
    ])
    def test_find_no_match(self, case, key, value):
        actual = self.getLoader().find(key, value)
        self.assertTrue(len(actual) == 0, case)

    def test_find_raise_pmid_not_integer(self):
        with self.assertRaises(Exception):
            self.getLoader().find('id', 'abc')
