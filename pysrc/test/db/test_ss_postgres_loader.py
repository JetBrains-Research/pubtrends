import unittest

from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.ss_postgres_loader import SemanticScholarPostgresLoader
from pysrc.papers.db.ss_postgres_writer import SemanticScholarPostgresWriter
from pysrc.test.db.abstract_test_ss_loader import AbstractTestSemanticScholarLoader
from pysrc.test.db.ss_test_articles import required_articles, extra_articles, required_citations, \
    extra_citations


class TestSemanticScholarPostgresLoader(unittest.TestCase, AbstractTestSemanticScholarLoader):
    test_config = PubtrendsConfig(test=True)
    loader = SemanticScholarPostgresLoader(test_config)

    @classmethod
    def setUpClass(cls):
        cls.loader = TestSemanticScholarPostgresLoader.loader

        # Text search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.ssid, required_articles))

        writer = SemanticScholarPostgresWriter(TestSemanticScholarPostgresLoader.test_config)
        writer.init_semantic_scholar_database()
        writer.insert_semantic_scholar_publications(required_articles + extra_articles)
        writer.insert_semantic_scholar_citations(required_citations + extra_citations)

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

    def getIds(self):
        return self.ids

    def getPublicationsDataframe(self):
        return self.pub_df

    def getCitationsStatsDataframe(self):
        return self.cit_stats_df

    def getCitationsDataframe(self):
        return self.cit_df

    def getCoCitationsDataframe(self):
        return self.cocit_df


if __name__ == "__main__":
    unittest.main()
