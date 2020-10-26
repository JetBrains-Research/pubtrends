from abc import ABCMeta, abstractmethod

from pandas.testing import assert_frame_equal
from parameterized import parameterized

from pysrc.papers.utils import SORT_MOST_RECENT, SORT_MOST_CITED, SORT_MOST_RELEVANT
from pysrc.test.db.ss_test_articles import required_citations, \
    expected_cit_stats_df, expected_cit_df, expected_cocit_df, part_of_articles, expanded_articles_df


# Don't make it subclass of unittest.TestCase to avoid tests execution
class AbstractTestSemanticScholarLoader(metaclass=ABCMeta):

    # @classmethod
    # def setUpClass(cls):
    # TODO: example of initialization
    #
    #     cls.loader = Loader(config=PubtrendsConfig(test=True))
    #     cls.loader.set_progress(logging.getLogger(__name__))
    #
    #     # Text search is not tested, imitating search results
    #     cls.ids = list(map(lambda article: article.ssid, required_articles))
    #
    #     writer = Writer()
    #     writer.init_database()
    #     writer.insert_publications(required_articles + extra_articles)
    #     writer.insert_citations(required_citations + extra_citations)
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
    def getCitationsStatsDataframe(self):
        """:return citations stats pandas dataframe"""

    @abstractmethod
    def getCitationsDataframe(self):
        """:return citations dataframe"""

    @abstractmethod
    def getCoCitationsDataframe(self):
        """:return co-citations dataframe"""

    @parameterized.expand([
        ('3 most recent', 3, SORT_MOST_RECENT, ['5a63b4199bb58992882b0bf60bc1b1b3f392e5a5',
                                                '5451b1ef43678d473575bdfa7016d024146f2b53',
                                                'cad767094c2c4fff5206793fd8674a10e7fba3fe']),
        ('4 most relevant', 4, SORT_MOST_RELEVANT, ['cad767094c2c4fff5206793fd8674a10e7fba3fe',
                                                    'e7cdbddc7af4b6138227139d714df28e2090bd5f',
                                                    '3cf82f53a52867aaade081324dff65dd35b5b7eb',
                                                    '5a63b4199bb58992882b0bf60bc1b1b3f392e5a5']),
        ('10 most cited', 10, SORT_MOST_CITED, ['3cf82f53a52867aaade081324dff65dd35b5b7eb',
                                                '5451b1ef43678d473575bdfa7016d024146f2b53',
                                                '5a63b4199bb58992882b0bf60bc1b1b3f392e5a5',
                                                'cad767094c2c4fff5206793fd8674a10e7fba3fe',
                                                'e7cdbddc7af4b6138227139d714df28e2090bd5f']),
    ])
    def test_search(self, name, limit, sort, expected):
        # Use sorted to avoid ambiguity
        ids = self.getLoader().search('find search', limit=limit, sort=sort)
        self.assertListEqual(sorted(expected), sorted(ids), name)

    def test_citations_stats_rows(self):
        expected_rows = expected_cit_stats_df.shape[0]
        actual_rows = self.getCitationsStatsDataframe().shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in citations statistics is incorrect")

    def test_load_citation_stats_data_frame(self):
        assert_frame_equal(expected_cit_stats_df,
                           self.getCitationsStatsDataframe().sort_values(by=['id', 'year']).reset_index(drop=True),
                           "Citations statistics is incorrect",
                           check_like=True)

    def test_load_citations_count(self):
        self.assertEqual(len(required_citations), len(self.getCitationsDataframe()), 'Wrong number of citations')

    def test_load_citations_data_frame(self):
        assert_frame_equal(expected_cit_df, self.getCitationsDataframe(), 'Wrong citation data', check_like=True)

    def test_load_cocitations_count(self):
        expected_rows = expected_cocit_df.shape[0]
        actual_rows = self.getCoCitationsDataframe().shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in co-citations dataframe is incorrect")

    def test_load_cocitations_data_frame(self):
        actual = self.getCoCitationsDataframe().sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(drop=True)
        assert_frame_equal(expected_cocit_df, actual, "Co-citations dataframe is incorrect", check_like=True)

    def test_expand(self):
        ids = list(map(lambda article: article.ssid, part_of_articles))
        actual = self.getLoader().expand(ids, 6)
        expected = expanded_articles_df.sort_values(by=['total', 'id']).reset_index(drop=True)
        actual = actual.sort_values(by=['total', 'id']).reset_index(drop=True)
        print(expected['id'])
        print(actual['id'])
        print(actual['total'])
        self.assertEqual(set(expected['id']), set(actual['id']))
        assert_frame_equal(
            expected,
            actual,
            "Wrong list of expanded ids"
        )

    @parameterized.expand([
        ('id search', 'id', '5451b1ef43678d473575bdfa7016d024146f2b53', ['5451b1ef43678d473575bdfa7016d024146f2b53']),
        ('id spaces', 'id', ' 5451b1ef43678d473575bdfa7016d024146f2b53 ', ['5451b1ef43678d473575bdfa7016d024146f2b53']),
        ('title search - lower case', 'title', 'can find using search', ['cad767094c2c4fff5206793fd8674a10e7fba3fe']),
        ('title search - Title Case', 'title', 'Can Find Using Search', ['cad767094c2c4fff5206793fd8674a10e7fba3fe']),
        ('title search with spaces', 'title', ' Can Find Using Search ', ['cad767094c2c4fff5206793fd8674a10e7fba3fe']),
        ('dx.doi.org search', 'doi', 'http://dx.doi.org/10.000/0000', ['5451b1ef43678d473575bdfa7016d024146f2b53']),
        ('doi.org search', 'doi', 'http://doi.org/10.000/0000', ['5451b1ef43678d473575bdfa7016d024146f2b53']),
        ('doi search', 'doi', '10.000/0000', ['5451b1ef43678d473575bdfa7016d024146f2b53']),
        ('doi with spaces', 'doi', '     10.000/0000      ', ['5451b1ef43678d473575bdfa7016d024146f2b53']),
    ])
    def test_find_match(self, case, key, value, expected):
        actual = self.getLoader().find(key, value)
        self.assertListEqual(sorted(actual), sorted(expected), case)

    @parameterized.expand([
        ('no such id', 'id', '0'),
        ('no such title', 'title', 'Article Title 0'),
        ('no such doi', 'doi', '10.000/0001')
    ])
    def test_find_no_match(self, case, key, value):
        actual = self.getLoader().find(key, value)
        self.assertTrue(len(actual) == 0, case)
