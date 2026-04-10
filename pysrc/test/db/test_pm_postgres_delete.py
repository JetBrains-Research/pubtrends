import unittest

from pysrc.papers.db.pm_postgres_loader import PubmedPostgresLoader
from pysrc.papers.db.pm_postgres_writer import PubmedPostgresWriter
from pysrc.papers.utils import SORT_MOST_CITED
from pysrc.test.conftest import get_test_config
from pysrc.test.db.pm_test_articles import REQUIRED_ARTICLES, ARTICLES, CITATIONS


class TestPubmedPostgresDelete(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_config = get_test_config()
        cls.loader = PubmedPostgresLoader(config=cls.test_config)
        cls.writer = PubmedPostgresWriter(config=cls.test_config)

        # Text search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.pmid, REQUIRED_ARTICLES))

        # Reset and load data to the test database
        cls.writer.init_pubmed_database()
        cls.writer.insert_pubmed_publications(ARTICLES)
        cls.writer.insert_pubmed_citations(CITATIONS)

        # Get data via loader methods
        cls.pub_df = cls.loader.load_publications(cls.ids)

    @classmethod
    def tearDownClass(cls):
        cls.loader.close_connection()

    def test_delete(self):
        self.writer.delete(['2'])
        self.assertListEqual(['3'],
                             sorted(self.loader.search('Abstract', limit=5, sort=SORT_MOST_CITED)),
                             'Wrong IDs of papers')
        self.writer.delete(['3'])
        self.assertListEqual([],
                             sorted(self.loader.search('Abstract', limit=5, sort=SORT_MOST_CITED)),
                             'Wrong IDs of papers')


if __name__ == "__main__":
    unittest.main()
