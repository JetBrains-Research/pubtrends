import re
import time
import unittest
from urllib.parse import quote

# Workaround for celery test worker and missing task, will be fixed in Celery 5+
# See: https://github.com/celery/celery/issues/4851
# noinspection PyUnresolvedReferences
from celery.contrib.testing.tasks import ping
from celery.contrib.testing.worker import start_worker

from pysrc.app.pubtrends_app import pubtrends_app
from pysrc.celery.tasks import pubtrends_celery
from pysrc.config import PubtrendsConfig
from pysrc.papers.db.pm_postgres_loader import PubmedPostgresLoader
from pysrc.papers.db.pm_postgres_writer import PubmedPostgresWriter
from pysrc.papers.utils import SORT_MOST_CITED
from pysrc.test.db.pm_test_articles import REQUIRED_ARTICLES, ARTICLES, CITATIONS


class TestApp(unittest.TestCase):
    """
    This is integration test of the whole Pubtrends application.
    """

    @classmethod
    def setUpClass(cls):
        test_config = PubtrendsConfig(test=True)
        cls.loader = PubmedPostgresLoader(test_config)

        # Configure celery not to use broker
        pubtrends_celery.conf.broker_url = 'memory://'
        pubtrends_celery.conf.result_backend = 'cache+memory://'

        cls.celery_worker = start_worker(pubtrends_celery, perform_ping_check=False)
        cls.celery_worker.__enter__()

        # Text search is not tested, imitating search results
        ids = list(map(lambda article: article.pmid, REQUIRED_ARTICLES))

        # Reset and load data to the test database
        writer = PubmedPostgresWriter(config=test_config)
        writer.init_pubmed_database()
        writer.insert_pubmed_publications(ARTICLES)
        writer.insert_pubmed_citations(CITATIONS)

        # Get data via loader methods
        cls.pub_df = cls.loader.load_publications(ids)
        cls.cit_stats_df = cls.loader.load_citations_by_year(ids)
        cls.cit_df = cls.loader.load_citations(ids)
        cls.cocit_df = cls.loader.load_cocitations(ids)

    @classmethod
    def tearDownClass(cls):
        cls.celery_worker.__exit__(None, None, None)
        cls.loader.close_connection()

    def test_terms_search(self):
        pubtrends_app.config['TESTING'] = True
        with pubtrends_app.test_client() as c:
            rv = c.post('/search_terms', data={
                'query': 'Article Title',
                'source': 'Pubmed',
                'sort': SORT_MOST_CITED,
                'limit': '100',
                'noreviews': 'on',
                'min_year': '',
                'max_year': '',
                'topics': '10'
            }, follow_redirects=True)
            self.assertEqual(200, rv.status_code)
            response = rv.data.decode('utf-8')
            self.assertIn('progressbar', response)  # Should get process page
            time.sleep(20)  # Should be enough time for processing
            # Form correct url to check results
            args = re.search('var args = [^\n]+', rv.data.decode('utf-8')).group(0)
            args = '&'.join('='.join(quote(x) for x in v.split(': '))
                            for v in args[len('var args = '):].strip('{};').replace('\'', '').split(', '))
            rv = c.get(f'/result?{args}')  # Result should be fine
            self.assertEqual(200, rv.status_code)
            response = rv.data.decode('utf-8')
            self.assertTrue('Analyzed <strong>9</strong> papers and <strong>1</strong> topics.' in response)
