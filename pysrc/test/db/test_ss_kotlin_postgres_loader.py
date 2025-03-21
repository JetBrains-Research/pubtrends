import os
import subprocess
import unittest

from pysrc.config import PubtrendsConfig
from pysrc.papers.db.ss_postgres_loader import SemanticScholarPostgresLoader
from pysrc.test.db.abstract_test_ss_loader import AbstractTestSemanticScholarLoader
from pysrc.test.db.ss_test_articles import REQUIRED_ARTICLES

PUBTRENDS_JAR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build/libs/pubtrends-dev.jar'))


class TestSemanticScholarKotlinPostgresLoader(unittest.TestCase, AbstractTestSemanticScholarLoader):
    test_config = PubtrendsConfig(test=True)
    loader = SemanticScholarPostgresLoader(test_config)

    @classmethod
    def setUpClass(cls):
        subprocess.run(
            ['java', '-cp', PUBTRENDS_JAR, 'org.jetbrains.bio.pubtrends.DBWriter', 'SemanticScholarPostgresWriter'])

        # Get data via loader methods
        cls.loader = TestSemanticScholarKotlinPostgresLoader.loader
        cls.ids = list(map(lambda article: article.ssid, REQUIRED_ARTICLES))
        cls.pub_df = cls.loader.load_publications(cls.ids)
        cls.cit_stats_df = cls.loader.load_citations_by_year(cls.ids)
        cls.cit_df = cls.loader.load_citations(cls.ids)
        cls.cocit_df = cls.loader.load_cocitations(cls.ids)

    @classmethod
    def tearDownClass(cls):
        cls.loader.close_connection()

    def get_loader(self):
        return TestSemanticScholarKotlinPostgresLoader.loader

    def get_citations_stats_dataframe(self):
        return TestSemanticScholarKotlinPostgresLoader.cit_stats_df

    def get_citations_dataframe(self):
        return TestSemanticScholarKotlinPostgresLoader.cit_df

    def get_cocitations_dataframe(self):
        return TestSemanticScholarKotlinPostgresLoader.cocit_df


if __name__ == "__main__":
    unittest.main()
