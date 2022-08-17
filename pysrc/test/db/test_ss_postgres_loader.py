import unittest

from pysrc.test.db.abstract_test_ss_loader import AbstractTestSemanticScholarLoader
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.ss_postgres_loader import SemanticScholarPostgresLoader
from pysrc.papers.db.ss_postgres_writer import SemanticScholarPostgresWriter
from pysrc.test.db.ss_test_articles import REQUIRED_ARTICLES, EXTRA_ARTICLES, REQUIRED_CITATIONS, \
    EXTRA_CITATIONS


class TestSemanticScholarPostgresLoader(unittest.TestCase, AbstractTestSemanticScholarLoader):
    test_config = PubtrendsConfig(test=True)
    loader = SemanticScholarPostgresLoader(test_config)

    @classmethod
    def setUpClass(cls):
        writer = SemanticScholarPostgresWriter(TestSemanticScholarPostgresLoader.test_config)
        writer.init_semantic_scholar_database()
        writer.insert_semantic_scholar_publications(REQUIRED_ARTICLES + EXTRA_ARTICLES)
        writer.insert_semantic_scholar_citations(REQUIRED_CITATIONS + EXTRA_CITATIONS)

        # Get data via loader methods
        cls.ids = list(map(lambda article: article.ssid, REQUIRED_ARTICLES))
        cls.loader = TestSemanticScholarPostgresLoader.loader
        cls.pub_df = cls.loader.load_publications(cls.ids)
        cls.cit_stats_df = cls.loader.load_citations_by_year(cls.ids)
        cls.cit_df = cls.loader.load_citations(cls.ids)
        cls.cocit_df = cls.loader.load_cocitations(cls.ids)

    @classmethod
    def tearDownClass(cls):
        TestSemanticScholarPostgresLoader.loader.close_connection()

    def get_loader(self):
        return TestSemanticScholarPostgresLoader.loader

    def get_citations_stats_dataframe(self):
        return TestSemanticScholarPostgresLoader.cit_stats_df

    def get_citations_dataframe(self):
        return TestSemanticScholarPostgresLoader.cit_df

    def get_cocitations_dataframe(self):
        return TestSemanticScholarPostgresLoader.cocit_df


if __name__ == "__main__":
    unittest.main()
