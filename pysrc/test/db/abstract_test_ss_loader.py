from abc import ABCMeta, abstractmethod
from pandas.testing import assert_frame_equal
from parameterized import parameterized

from pysrc.papers.utils import SORT_MOST_RECENT, SORT_MOST_CITED
# Don't make it subclass of unittest.TestCase to avoid tests execution
from pysrc.test.db.ss_test_articles import EXPECTED_CIT_STATS_DF, REQUIRED_CITATIONS, EXPECTED_CIT_DF, \
    EXPECTED_COCIT_DF, ARTICLES_LIST, EXPANDED_ARTICLES_DF


class AbstractTestSemanticScholarLoader(metaclass=ABCMeta):

    @abstractmethod
    def get_loader(self):
        """:return Loader instance"""

    @abstractmethod
    def get_citations_stats_dataframe(self):
        """:return citations stats pandas dataframe"""

    @abstractmethod
    def get_citations_dataframe(self):
        """:return citations dataframe"""

    @abstractmethod
    def get_cocitations_dataframe(self):
        """:return co-citations dataframe"""

    @parameterized.expand([
        ('3 most recent', 3, SORT_MOST_RECENT, ['5a63b4199bb58992882b0bf60bc1b1b3f392e5a5',
                                                '5451b1ef43678d473575bdfa7016d024146f2b53',
                                                'cad767094c2c4fff5206793fd8674a10e7fba3fe']),
        ('10 most cited', 10, SORT_MOST_CITED, ['3cf82f53a52867aaade081324dff65dd35b5b7eb',
                                                '5451b1ef43678d473575bdfa7016d024146f2b53',
                                                '5a63b4199bb58992882b0bf60bc1b1b3f392e5a5',
                                                'cad767094c2c4fff5206793fd8674a10e7fba3fe',
                                                'e7cdbddc7af4b6138227139d714df28e2090bd5f']),
    ])
    def test_search(self, name, limit, sort, expected):
        # Use sorted to avoid ambiguity
        ids = self.get_loader().search('find search', limit=limit, sort=sort)
        self.assertListEqual(sorted(expected), sorted(ids), name)

    def test_citations_stats_rows(self):
        expected_rows = EXPECTED_CIT_STATS_DF.shape[0]
        actual_rows = self.get_citations_stats_dataframe().shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in citations statistics is incorrect")

    def test_load_citation_stats_data_frame(self):
        assert_frame_equal(EXPECTED_CIT_STATS_DF,
                           self.get_citations_stats_dataframe().sort_values(by=['id', 'year']).reset_index(drop=True),
                           "Citations statistics is incorrect",
                           check_like=True)

    def test_load_citations_count(self):
        self.assertEqual(len(REQUIRED_CITATIONS), len(self.get_citations_dataframe()), 'Wrong number of citations')

    def test_load_citations_data_frame(self):
        assert_frame_equal(EXPECTED_CIT_DF, self.get_citations_dataframe(), 'Wrong citation data', check_like=True)

    def test_load_cocitations_count(self):
        expected_rows = EXPECTED_COCIT_DF.shape[0]
        actual_rows = self.get_cocitations_dataframe().shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in co-citations dataframe is incorrect")

    def test_load_cocitations_data_frame(self):
        actual = self.get_cocitations_dataframe().sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(
            drop=True)
        assert_frame_equal(EXPECTED_COCIT_DF, actual, "Co-citations dataframe is incorrect", check_like=True)

    def test_expand(self):
        ids = list(map(lambda article: article.ssid, ARTICLES_LIST))
        actual = self.get_loader().expand(ids, 6)
        expected = EXPANDED_ARTICLES_DF.sort_values(by=['total', 'id']).reset_index(drop=True)
        actual = actual.sort_values(by=['total', 'id']).reset_index(drop=True)
        self.assertEqual(set(expected['id']), set(actual['id']))
        assert_frame_equal(
            expected,
            actual,
            "Wrong list of expanded ids"
        )

    @parameterized.expand([
        ('id search', 'id', '5451b1ef43678d473575bdfa7016d024146f2b53', ['5451b1ef43678d473575bdfa7016d024146f2b53']),
        ('id spaces', 'id', ' 5451b1ef43678d473575bdfa7016d024146f2b53 ', ['5451b1ef43678d473575bdfa7016d024146f2b53']),
        ('title search - lower case', 'title', 'can find using search.', ['cad767094c2c4fff5206793fd8674a10e7fba3fe']),
        ('title search - Title Case', 'title', 'Can Find Using Search.', ['cad767094c2c4fff5206793fd8674a10e7fba3fe']),
        ('title search with spaces', 'title', ' Can Find Using Search. ', ['cad767094c2c4fff5206793fd8674a10e7fba3fe']),
        ('title search no dot', 'title', 'Can Find Using Search', ['cad767094c2c4fff5206793fd8674a10e7fba3fe']),
        ('dx.doi.org search', 'doi', 'http://dx.doi.org/10.000/0000', ['5451b1ef43678d473575bdfa7016d024146f2b53']),
        ('doi.org search', 'doi', 'http://doi.org/10.000/0000', ['5451b1ef43678d473575bdfa7016d024146f2b53']),
        ('doi search', 'doi', '10.000/0000', ['5451b1ef43678d473575bdfa7016d024146f2b53']),
        ('doi with spaces', 'doi', '     10.000/0000      ', ['5451b1ef43678d473575bdfa7016d024146f2b53']),
    ])
    def test_find_match(self, case, key, value, expected):
        actual = self.get_loader().find(key, value)
        self.assertListEqual(sorted(actual), sorted(expected), case)

    @parameterized.expand([
        ('no such id', 'id', '0'),
        ('no such title', 'title', 'Article Title 0'),
        ('no such doi', 'doi', '10.000/0001')
    ])
    def test_find_no_match(self, case, key, value):
        actual = self.get_loader().find(key, value)
        self.assertTrue(len(actual) == 0, case)
