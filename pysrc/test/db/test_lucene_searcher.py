import os
import shutil
import tempfile
import unittest

from pysrc.papers.db.lucene_searcher import LuceneIndexManager
from pysrc.papers.utils import SearchError, SORT_MOST_RECENT
from pysrc.test.db.pm_test_articles import REQUIRED_ARTICLES, EXTRA_ARTICLE
from pysrc.test.db.ss_test_articles import REQUIRED_ARTICLES as SS_REQUIRED_ARTICLES


class TestLuceneSearcher(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = LuceneIndexManager(self.temp_dir)
        self.manager.reset()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_insert_and_search_pubmed_articles(self):
        self.manager.insert_pubmed_articles(REQUIRED_ARTICLES + [EXTRA_ARTICLE])

        # Test simple term query
        results = self.manager.search('Article', limit=10, noreviews=False)
        self.assertCountEqual(['1', '2', '3', '4', '5', '6'], results)

        # Test noreviews filter (article 6 has type='Review')
        results_noreviews = self.manager.search('Article', limit=10, noreviews=True)
        self.assertCountEqual(['1', '2', '3', '4', '5'], results_noreviews)
        self.assertNotIn('6', results_noreviews)

    def test_search_phrase(self):
        self.manager.insert_pubmed_articles(REQUIRED_ARTICLES + [EXTRA_ARTICLE])

        # Exact phrase in abstract
        self.assertEqual(['100'], self.manager.search('"activity module"'))
        self.assertEqual(['100'], self.manager.search('"activating modules"'))

        # Non-phrase words in different positions
        self.assertEqual(['100'], self.manager.search('dna methylation'))

        # Not matching phrase
        self.assertEqual([], self.manager.search('"active module"'))

    def test_search_or_comma(self):
        self.manager.insert_pubmed_articles(REQUIRED_ARTICLES + [EXTRA_ARTICLE])

        # Comma-separated OR query
        results = self.manager.search('"Title 1", "Title 2"')
        self.assertCountEqual(['1', '2'], results)

    def test_search_negation(self):
        self.manager.insert_pubmed_articles(REQUIRED_ARTICLES + [EXTRA_ARTICLE])

        # Negation with minus: exclude 'Title 1' (pmid 1)
        results = self.manager.search('Article -1', limit=10, noreviews=False)
        self.assertIn('2', results)
        self.assertIn('3', results)
        self.assertNotIn('1', results)

    def test_search_year_filter(self):
        self.manager.insert_pubmed_articles(REQUIRED_ARTICLES + [EXTRA_ARTICLE])

        # Year range filter
        results = self.manager.search('Article', min_year=1964, max_year=1968, noreviews=False)
        self.assertCountEqual(['2', '3', '4'], results)

    def test_sort_most_recent(self):
        self.manager.insert_pubmed_articles(REQUIRED_ARTICLES + [EXTRA_ARTICLE])

        results = self.manager.search('Article', sort=SORT_MOST_RECENT, limit=3, noreviews=True)
        self.assertEqual('5', results[0])  # 1975

    def test_search_title(self):
        self.manager.insert_pubmed_articles(REQUIRED_ARTICLES)

        results = self.manager.search_title('Article Title 1')
        self.assertIn('1', results)

        results_empty = self.manager.search_title('')
        self.assertEqual([], results_empty)

    def test_delete_documents(self):
        self.manager.insert_pubmed_articles(REQUIRED_ARTICLES)
        self.assertIn('1', self.manager.search('Title', limit=10, noreviews=False))

        self.manager.delete(['1'])
        self.assertNotIn('1', self.manager.search('Title', limit=10, noreviews=False))

    def test_semantic_scholar_articles(self):
        self.manager.insert_semantic_scholar_articles(SS_REQUIRED_ARTICLES)

        results = self.manager.search('search', limit=10)
        expected_ids = [str(a.ssid) for a in SS_REQUIRED_ARTICLES[:4]]
        self.assertCountEqual(expected_ids, results)

    def test_illegal_queries(self):
        with self.assertRaises(SearchError):
            self.manager.search('')

        with self.assertRaises(SearchError):
            self.manager.search('   ')

        with self.assertRaises(SearchError):
            self.manager.search('"unclosed quote')


if __name__ == '__main__':
    unittest.main()
