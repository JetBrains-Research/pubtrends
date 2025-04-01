from abc import ABCMeta, abstractmethod

import numpy as np
from pandas.testing import assert_frame_equal
from parameterized import parameterized

from pysrc.papers.utils import SORT_MOST_RECENT, SORT_MOST_CITED
from pysrc.test.db.pm_test_articles import REQUIRED_ARTICLES, EXPECTED_PUB_DF, INNER_CITATIONS, PART_OF_ARTICLES, \
    EXPECTED_PUB_DF_GIVEN_IDS, EXPANDED_IDS_DF, EXPANDED_TOP_CITED_5_DF, \
    EXPECTED_CIT_STATS_DF, EXPECTED_CIT_DF, EXPECTED_COCIT_DF


# Don't make it subclass of unittest.TestCase to avoid tests execution
class AbstractTestPubmedLoader(metaclass=ABCMeta):

    @abstractmethod
    def get_loader(self):
        """:return Loader instance"""

    @abstractmethod
    def get_publications_dataframe(self):
        """:return publications dataframe"""

    @abstractmethod
    def get_citations_stats_dataframe(self):
        """:return citations stats pandas dataframe"""

    @abstractmethod
    def get_citations_dataframe(self):
        """:return citations dataframe"""

    @abstractmethod
    def get_cocitations_dataframe(self):
        """:return co-citations dataframe"""

    def get_row(self, article):
        df = self.get_publications_dataframe()
        return df[df['id'] == str(article.pmid)]

    def test_load_publications_count(self):
        self.assertEqual(len(self.get_publications_dataframe()), len(self.ids))

    def test_load_publications_check_ids(self):
        self.assertCountEqual(list(self.get_publications_dataframe()['id'].values), list(map(str, self.ids)))

    def test_load_publications_fill_null_abstract(self):
        self.assertFalse(np.any(self.get_publications_dataframe()['abstract'].isna()), msg='NaN abstract found')
        for article in REQUIRED_ARTICLES:
            if article.abstract is None:
                df = self.get_publications_dataframe()
                row = df[df['id'] == str(article.pmid)]
                self.assertEqual(row['abstract'].values[0], '', msg='Null abstract was filled with wrong value')

    def test_load_publications_authors(self):
        actual = []
        expected = []

        for article in REQUIRED_ARTICLES:
            expected_names = [author.name for author in article.aux.authors]
            expected.append(', '.join(expected_names))

            actual_names = self.get_row(article)['authors'].values[0]
            actual.append(actual_names)

        self.assertListEqual(actual, expected, msg='Wrong authors extracted')

    def test_load_publications_journals(self):
        expected = list(map(lambda article: article.aux.journal.name, REQUIRED_ARTICLES))
        actual = list(map(lambda article: self.get_row(article)['journal'].values[0],
                          REQUIRED_ARTICLES))
        self.assertListEqual(actual, expected, msg='Wrong journals extracted')

    def test_load_publications_data_frame(self):
        assert_frame_equal(self.get_publications_dataframe(), EXPECTED_PUB_DF, 'Wrong publication data',
                           check_like=True)

    @parameterized.expand([
        ('Article', 1, SORT_MOST_RECENT, ['5']),
        ('Article', 1, SORT_MOST_CITED, ['4']),
    ])
    def test_search(self, query, limit, sort, expected_ids):
        ids = self.get_loader().search(query, limit=limit, sort=sort)
        self.assertListEqual(expected_ids, ids, 'Wrong IDs of papers')

    def test_search_wo_order(self):
        ids = self.get_loader().search('Article', limit=10, sort=SORT_MOST_CITED)
        self.assertListEqual(sorted(['1', '10', '2', '3', '4', '5', '7', '8', '9']),
                             sorted(ids), 'Wrong IDs of papers')

    def test_search_noreviews(self):
        # Use sorted to avoid ambiguity
        ids = self.get_loader().search('Article', limit=10, sort=SORT_MOST_RECENT)
        self.assertListEqual(sorted(ids), ['1', '10', '2', '3', '4', '5', '7', '8', '9'], 'Wrong IDs of papers')

        ids_with_review = self.get_loader().search('Article', limit=10, sort=SORT_MOST_RECENT, noreviews=False)
        self.assertListEqual(sorted(ids_with_review), ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9'],
                             'Wrong IDs of papers')

    def test_search_phrase(self):
        self.assertListEqual(['100'], self.get_loader().search('"activity module"', limit=100, sort=SORT_MOST_CITED),
                             'Wrong IDs of papers')
        self.assertListEqual(['100'], self.get_loader().search('"activating modules"', limit=100, sort=SORT_MOST_CITED),
                             'Wrong IDs of papers')
        # Words in all fields
        self.assertListEqual(['100'], self.get_loader().search('dna methylation', limit=100, sort=SORT_MOST_CITED),
                             'Wrong IDs of papers')

        # there is not phrase, but words
        self.assertListEqual([], self.get_loader().search('"active module"', limit=100, sort=SORT_MOST_CITED),
                             'Wrong IDs of papers')
        # plural check
        self.assertListEqual([], self.get_loader().search('"activating module"', limit=100, sort=SORT_MOST_CITED),
                             'Wrong IDs of papers')
        # whole word match
        self.assertListEqual([], self.get_loader().search('positive', limit=100, sort=SORT_MOST_CITED),
                             'Wrong IDs of papers')
        # Words different fields
        self.assertListEqual([], self.get_loader().search('"dna methylation"', limit=100, sort=SORT_MOST_CITED),
                             'Wrong IDs of papers')

    def test_load_citation_stats_data_frame(self):
        # Sort to compare with expected
        actual = self.get_citations_stats_dataframe().sort_values(by=['id', 'year']).reset_index(drop=True)
        assert_frame_equal(EXPECTED_CIT_STATS_DF, actual, 'Wrong citation stats data', check_like=True)

    def test_load_citations_count(self):
        self.assertEqual(len(self.get_citations_dataframe()), len(INNER_CITATIONS), 'Wrong number of citations')

    def test_load_citations_data_frame(self):
        assert_frame_equal(self.get_citations_dataframe(), EXPECTED_CIT_DF, 'Wrong citation data', check_like=True)

    def test_load_cocitations_data_frame(self):
        expected = EXPECTED_COCIT_DF.sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(drop=True)
        actual = self.get_cocitations_dataframe().sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(
            drop=True)
        # Sort for comparison
        assert_frame_equal(expected, actual, 'Wrong co-citation data', check_like=True)

    def test_search_with_given_ids(self):
        ids_list = list(map(lambda article: article.pmid, PART_OF_ARTICLES))
        assert_frame_equal(EXPECTED_PUB_DF_GIVEN_IDS, self.get_loader().load_publications(ids_list),
                           "Wrong publications extracted", check_like=True)

    @parameterized.expand([
        ('1', 1, []),
        ('2', 10, ['1']),
        ('3', 10, []),
        ('7', 10, ['1', '3']),
    ])
    def test_load_references(self, pid, limit, expected_ids):
        self.assertEqual(self.get_loader().load_references(pid, limit), expected_ids)

    def test_expand(self):
        ids_list = list(map(lambda article: str(article.pmid), PART_OF_ARTICLES))
        actual = self.get_loader().expand(ids_list, 1000)
        actual = actual.sort_values(by=['total', 'id']).reset_index(drop=True)
        assert_frame_equal(
            EXPANDED_IDS_DF,
            actual,
            "Wrong list of expanded ids"
        )

    def test_expand_limited(self):
        ids_list = list(map(lambda article: str(article.pmid), PART_OF_ARTICLES))
        actual = self.get_loader().expand(ids_list, 5).sort_values(by=['total', 'id']).reset_index(drop=True)
        assert_frame_equal(
            EXPANDED_TOP_CITED_5_DF,
            actual.sort_values(by=['total', 'id']).reset_index(drop=True),
            "Wrong list of expanded 5 ids"
        )

    @parameterized.expand([
        ('id search', 'id', '1', ['1']),
        ('id search - with spaces', 'id', '  1  ', ['1']),
        ('title search - lower case', 'title', 'article title 2', ['2']),
        ('title search - Title Case', 'title', 'Article Title 4', ['4']),
        ('relaxed title search wrong order - Title Case', 'title', 'Title Article  4', ['4']),
        ('title search - with spaces', 'title', '       Article Title 4        ', ['4']),
        ('dx.doi.org search', 'doi', 'http://dx.doi.org/10.000/0000', ['1']),
        ('doi.org search', 'doi', 'http://doi.org/10.000/0000', ['1']),
        ('doi search', 'doi', '10.000/0000', ['1']),
        ('doi with spaces', 'doi', '      10.000/0000       ', ['1']),
    ])
    def test_find_match(self, case, key, value, expected):
        actual = self.get_loader().search_key_value(key, value)
        self.assertListEqual(sorted(actual), sorted(expected), case)

    @parameterized.expand([
        ('no such id', 'id', '0'),
        ('abstract words in query', 'title', 'abstract'),
        ('no such title', 'title', 'Article Title 0'),
        ('no such doi', 'doi', '10.000/0001')
    ])
    def test_find_no_match(self, case, key, value):
        actual = self.get_loader().search_key_value(key, value)
        self.assertTrue(len(actual) == 0, case)

    def test_find_raise_pmid_not_integer(self):
        with self.assertRaises(Exception):
            self.get_loader().search_key_value('id', 'abc')
