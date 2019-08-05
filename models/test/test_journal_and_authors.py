import unittest

from pandas.util.testing import assert_frame_equal

from models.keypaper.analysis import KeyPaperAnalyzer
from models.test.articles_for_analysis_testing import df_authors_and_journals, author_df, journal_df
from models.test.test_loader import TestLoader


class TestPopularAuthorsAndJournals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.analyzer = KeyPaperAnalyzer(TestLoader())
        cls.analyzer.df = df_authors_and_journals
        cls.author_stats = cls.analyzer.popular_authors(cls.analyzer.df)
        cls.journal_stats = cls.analyzer.popular_journals(cls.analyzer.df)

    def test_author_stats_rows(self):
        expected_rows = author_df.shape[0]
        actual_rows = self.author_stats.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in author statistics is incorrect")

    def test_author_stats(self):
        assert_frame_equal(self.author_stats.reset_index(drop=True), author_df.reset_index(drop=True),
                           "Popular authors are counted incorrectly")

    def test_journal_stats_rows(self):
        expected_rows = journal_df.shape[0]
        actual_rows = self.journal_stats.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in journal statistics is incorrect")

    def test_journal_stats(self):
        assert_frame_equal(self.journal_stats.reset_index(drop=True), journal_df.reset_index(drop=True),
                           "Popular journals are counted incorrectly")


if __name__ == "__main__":
    unittest.main()
