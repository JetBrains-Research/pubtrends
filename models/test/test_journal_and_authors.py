import unittest

from pandas.util.testing import assert_frame_equal

from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.config import PubtrendsConfig
from models.keypaper.loader import Loader
from models.test.articles_for_analysis_testing import df_authors_and_journals, author_df, journal_df


class TestPopularAuthorsAndJournals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.analyzer = KeyPaperAnalyzer(TestLoader())
        cls.analyzer.df = df_authors_and_journals
        cls.analyzer.popular_authors()
        cls.analyzer.popular_journals()

    def test_author_stats_rows(self):
        expected_rows = author_df.shape[0]
        actual_rows = self.analyzer.author_stats.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in author statistics is incorrect")

    def test_author_stats(self):
        assert_frame_equal(self.analyzer.author_stats.reset_index(drop=True), author_df.reset_index(drop=True),
                           "Popular authors are counted incorrectly")

    def test_journal_stats_rows(self):
        expected_rows = journal_df.shape[0]
        actual_rows = self.analyzer.journal_stats.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in journal statistics is incorrect")

    def test_journal_stats(self):
        assert_frame_equal(self.analyzer.journal_stats.reset_index(drop=True), journal_df.reset_index(drop=True),
                           "Popular journals are counted incorrectly")


if __name__ == "__main__":
    unittest.main()


class TestLoader(Loader):
    def __init__(self):
        config = PubtrendsConfig(test=True)
        super(TestLoader, self).__init__(config, connect=False)
