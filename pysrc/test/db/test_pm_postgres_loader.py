import unittest

from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.pm_postgres_loader import PubmedPostgresLoader
from pysrc.papers.db.pm_postgres_writer import PubmedPostgresWriter
from pysrc.test.db.abstract_test_pm_loader import AbstractTestPubmedLoader
from pysrc.test.db.pm_test_articles import EXTRA_ARTICLE
from pysrc.test.db.pm_test_articles import REQUIRED_ARTICLES, ARTICLES, CITATIONS


class TestPubmedPostgresLoader(unittest.TestCase, AbstractTestPubmedLoader):
    test_config = PubtrendsConfig(test=True)
    loader = PubmedPostgresLoader(test_config)

    @classmethod
    def setUpClass(cls):
        cls.loader = TestPubmedPostgresLoader.loader

        # Text search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.pmid, REQUIRED_ARTICLES))

        # Reset and load data to the test database
        writer = PubmedPostgresWriter(config=TestPubmedPostgresLoader.test_config)
        writer.init_pubmed_database()
        writer.insert_pubmed_publications(ARTICLES + [EXTRA_ARTICLE])
        writer.insert_pubmed_citations(CITATIONS)

        # Get data via loader methods
        cls.pub_df = cls.loader.load_publications(cls.ids)
        cls.cit_stats_df = cls.loader.load_citations_by_year(cls.ids)
        cls.cit_df = cls.loader.load_citations(cls.ids)
        cls.cocit_df = cls.loader.load_cocitations(cls.ids)

    @classmethod
    def tearDownClass(cls):
        cls.loader.close_connection()

    def getPublicationsDataframe(self):
        return TestPubmedPostgresLoader.pub_df

    def getLoader(self):
        return TestPubmedPostgresLoader.loader

    def getCitationsStatsDataframe(self):
        return TestPubmedPostgresLoader.cit_stats_df

    def getCitationsDataframe(self):
        return TestPubmedPostgresLoader.cit_df

    def getCoCitationsDataframe(self):
        return TestPubmedPostgresLoader.cocit_df


if __name__ == "__main__":
    unittest.main()
