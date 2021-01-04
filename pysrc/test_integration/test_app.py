import re
import time
import unittest
from urllib.parse import quote

from pysrc.app.app import app
from pysrc.papers import pubtrends_config
from pysrc.papers.db.pm_postgres_loader import PubmedPostgresLoader
from pysrc.papers.db.pm_postgres_writer import PubmedPostgresWriter
from pysrc.papers.utils import SORT_MOST_CITED
from pysrc.test.db.pm_test_articles import REQUIRED_ARTICLES, ARTICLES, CITATIONS


class TestApp(unittest.TestCase):
    """
    This is integration test of the whole App. Launch Testing container AND Celery before launching this test.
    TODO: should be rewritten to avoid external celery tasks executing
    """

    test_config = pubtrends_config.PubtrendsConfig(test=True)
    loader = PubmedPostgresLoader(test_config)

    @classmethod
    def setUpClass(cls):
        cls.loader = TestApp.loader

        # Text search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.pmid, REQUIRED_ARTICLES))

        # Reset and load data to the test database
        writer = PubmedPostgresWriter(config=TestApp.test_config)
        writer.init_pubmed_database()
        writer.insert_pubmed_publications(ARTICLES)
        writer.insert_pubmed_citations(CITATIONS)

        # Get data via loader methods
        cls.pub_df = cls.loader.load_publications(cls.ids)
        cls.cit_stats_df = cls.loader.load_citations_by_year(cls.ids)
        cls.cit_df = cls.loader.load_citations(cls.ids)
        cls.cocit_df = cls.loader.load_cocitations(cls.ids)

    @classmethod
    def tearDownClass(cls):
        cls.loader.close_connection()

    def getLoader(self):
        return self.loader

    def test_terms_search(self):
        app.config['TESTING'] = True
        with app.test_client() as c:
            rv = c.post('/search_terms', data={
                'query': 'Article Title',
                'source': 'Pubmed',
                'sort': SORT_MOST_CITED,
                'limit': '100',
                'noreviews': 'on',
                'expand': '25'
            }, follow_redirects=True)
            self.assertEqual(200, rv.status_code)
            response = rv.data.decode('utf-8')
            self.assertIn('progressbar', response)  # Should get process page
            time.sleep(20)  # Should be enough
            args = re.search('var args = [^\n]+', rv.data.decode('utf-8')).group(0)
            args = '&'.join('='.join(quote(x) for x in v.split(': '))
                            for v in args[len('var args = '):].strip('{};').replace('\'', '').split(', '))
            rv = c.get(f'/result?{args}')  # Result should be fine
            self.assertEqual(200, rv.status_code)
            response = rv.data.decode('utf-8')
            self.assertTrue(re.findall('<div id="n_papers"(.*\n)+.*<b>9</b>', response))
            self.assertTrue(re.findall('<div id="n_citations"(.*\n)+.*<b>15</b>', response))
            self.assertTrue(re.findall('<div id="n_topics"(.*\n)+.*<b>1</b>', response))
