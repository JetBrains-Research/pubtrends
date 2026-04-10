import os
import subprocess
import unittest

from pysrc.papers.db.pm_postgres_loader import PubmedPostgresLoader
from pysrc.test.db.abstract_test_pm_loader import AbstractTestPubmedLoader
from pysrc.test.conftest import get_test_config

PUBTRENDS_JAR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build/libs/pubtrends-dev.jar'))


class TestPubmedKotlinPostgresLoader(unittest.TestCase, AbstractTestPubmedLoader):

    @classmethod
    def setUpClass(cls):
        cls.test_config = get_test_config()
        cls.loader = PubmedPostgresLoader(cls.test_config)
        cfg = cls.test_config
        subprocess.run([
            'java', '-cp', PUBTRENDS_JAR, 'org.jetbrains.bio.pubtrends.DBWriter',
            'PubmedPostgresWriter',
            '--host', cfg.postgres_host,
            '--port', str(cfg.postgres_port),
            '--database', cfg.postgres_database,
            '--username', cfg.postgres_username,
            '--password', cfg.postgres_password,
        ])
        # Get data via loader methods
        cls.ids = [1, 2, 3, 4, 5, 6]
        cls.pub_df = cls.loader.load_publications(cls.ids)
        cls.cit_stats_df = cls.loader.load_citations_by_year(cls.ids)
        cls.cit_df = cls.loader.load_citations(cls.ids)
        cls.cocit_df = cls.loader.load_cocitations(cls.ids)

    @classmethod
    def tearDownClass(cls):
        cls.loader.close_connection()

    def get_publications_dataframe(self):
        return self.pub_df

    def get_loader(self):
        return self.loader

    def get_citations_stats_dataframe(self):
        return self.cit_stats_df

    def get_citations_dataframe(self):
        return self.cit_df

    def get_cocitations_dataframe(self):
        return self.cocit_df


if __name__ == "__main__":
    unittest.main()
