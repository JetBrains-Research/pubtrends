import os
import subprocess
import unittest

from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.pm_postgres_loader import PubmedPostgresLoader
from pysrc.test.db.abstract_test_pm_loader import AbstractTestPubmedLoader

PUBTRENDS_JAR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build/libs/pubtrends-dev.jar'))


class TestPubmedKotlinPostgresLoader(unittest.TestCase, AbstractTestPubmedLoader):
    test_config = PubtrendsConfig(test=True)
    loader = PubmedPostgresLoader(test_config)

    @classmethod
    def setUpClass(cls):
        subprocess.run(['java', '-cp', PUBTRENDS_JAR, 'org.jetbrains.bio.pubtrends.DBWriter', 'PubmedPostgresWriter'])
        # Get data via loader methods
        cls.loader = TestPubmedKotlinPostgresLoader.loader
        cls.ids = [1, 2, 3, 4, 5, 6]
        cls.pub_df = cls.loader.load_publications(cls.ids)
        cls.cit_stats_df = cls.loader.load_citations_by_year(cls.ids)
        cls.cit_df = cls.loader.load_citations(cls.ids)
        cls.cocit_df = cls.loader.load_cocitations(cls.ids)

    @classmethod
    def tearDownClass(cls):
        cls.loader.close_connection()

    def get_publications_dataframe(self):
        return TestPubmedKotlinPostgresLoader.pub_df

    def get_loader(self):
        return TestPubmedKotlinPostgresLoader.loader

    def get_citations_stats_dataframe(self):
        return TestPubmedKotlinPostgresLoader.cit_stats_df

    def get_citations_dataframe(self):
        return TestPubmedKotlinPostgresLoader.cit_df

    def get_cocitations_dataframe(self):
        return TestPubmedKotlinPostgresLoader.cocit_df


if __name__ == "__main__":
    unittest.main()
