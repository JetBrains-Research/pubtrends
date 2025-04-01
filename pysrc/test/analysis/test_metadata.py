import unittest
from dataclasses import dataclass

import pandas as pd
from pandas.testing import assert_frame_equal

from pysrc.config import *
from pysrc.papers.analysis.metadata import popular_authors, popular_journals, split_df_list
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.test.mock_loaders import MockLoader


class TestPaperMetadata(unittest.TestCase):
    PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

    @classmethod
    def setUpClass(cls):
        cls.analyzer = PapersAnalyzer(MockLoader(), TestPaperMetadata.PUBTRENDS_CONFIG, test=True)
        cls.analyzer.df = df_authors_and_journals
        cls.author_stats = popular_authors(cls.analyzer.df, cls.analyzer.config.popular_authors)
        cls.journal_stats = popular_journals(cls.analyzer.df, cls.analyzer.config.popular_journals)

    def test_author_stats_rows(self):
        expected_rows = author_df.shape[0]
        actual_rows = self.author_stats.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in author statistics is incorrect")

    def test_author_stats(self):
        assert_frame_equal(
            self.author_stats.sort_values(by=['sum', 'author']).reset_index(drop=True),
            author_df.sort_values(by=['sum', 'author']).reset_index(drop=True),
            "Popular authors are counted incorrectly"
        )

    def test_journal_stats_rows(self):
        expected_rows = journal_df.shape[0]
        actual_rows = self.journal_stats.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in journal statistics is incorrect")

    def test_journal_stats(self):
        assert_frame_equal(self.journal_stats.reset_index(drop=True), journal_df.reset_index(drop=True),
                           "Popular journals are counted incorrectly")

    def test_split_df_list(self):
        data_for_df = [[2, 'a, b, c'],
                       [1, 'c, a, d'],
                       [4, 'd, c']]

        df_with_list_column = pd.DataFrame(data_for_df, columns=['id', 'list'])

        expected_data = [[2, 'a'], [2, 'b'], [2, 'c'],
                         [1, 'c'], [1, 'a'], [1, 'd'],
                         [4, 'd'], [4, 'c']]
        expected_df = pd.DataFrame(expected_data, columns=['id', 'list'])
        actual_df = split_df_list(df_with_list_column, target_column='list', separator=', ')
        assert_frame_equal(expected_df, actual_df, "Splitting list into several rows works incorrectly")


@dataclass
class Article:
    id: int
    comp: int
    authors: str
    journal: str

    def to_dict(self):
        return {
            'id': self.id,
            'comp': self.comp,
            'authors': self.authors,
            'journal': self.journal
        }


articles = [(Article(id=1, comp=0, authors='Geller R, Geller M, Bing Ch', journal='Nature')),
            (Article(id=2, comp=1, authors='Buffay Ph, Geller M, Doe J', journal='Science')),
            (Article(id=3, comp=1, authors='Doe J, Buffay Ph', journal='Nature')),
            (Article(id=4, comp=1, authors='Doe J, Geller R', journal='Science')),
            (Article(id=5, comp=0, authors='Green R, Geller R, Doe J', journal='Nature')),
            (Article(id=6, comp=0, authors='', journal=''))]

df_authors_and_journals = pd.DataFrame.from_records([article.to_dict() for article in articles])

author_stats = [['Doe J', [1, 0], [3, 1], 4],
                ['Geller R', [0, 1], [2, 1], 3],
                ['Buffay Ph', [1], [2], 2],
                ['Geller M', [0, 1], [1, 1], 2],
                ['Bing Ch', [0], [1], 1],
                ['Green R', [0], [1], 1]]

author_df = pd.DataFrame(author_stats, columns=['author', 'comp', 'counts', 'sum'])

journal_stats = [['Nature', [0, 1], [2, 1], 3],
                 ['Science', [1], [2], 2]]

journal_df = pd.DataFrame(journal_stats, columns=['journal', 'comp', 'counts', 'sum'])

if __name__ == "__main__":
    unittest.main()
