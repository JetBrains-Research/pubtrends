import unittest

from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.ss_neo4j_loader import SemanticScholarNeo4jLoader
from pysrc.papers.db.ss_neo4j_writer import SemanticScholarNeo4jWriter
from pysrc.test.db.abstract_test_ss_loader import AbstractTestSemanticScholarLoader
from pysrc.test.db.ss_test_articles import REQUIRED_ARTICLES, EXTRA_ARTICLES, REQUIRED_CITATIONS, \
    EXTRA_CITATIONS


class TestSemanticScholarNeo4jLoader(unittest.TestCase, AbstractTestSemanticScholarLoader):
    loader = SemanticScholarNeo4jLoader(config=PubtrendsConfig(test=True))

    @classmethod
    def setUpClass(cls):
        cls.loader = TestSemanticScholarNeo4jLoader.loader

        # Text search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.ssid, REQUIRED_ARTICLES))

        writer = SemanticScholarNeo4jWriter()
        writer.init_semantic_scholar_database()
        writer.insert_semantic_scholar_publications(REQUIRED_ARTICLES + EXTRA_ARTICLES)
        writer.insert_semantic_scholar_citations(REQUIRED_CITATIONS + EXTRA_CITATIONS)

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

    def test_expand(self):
        # TODO: fix me!
        pass


if __name__ == "__main__":
    unittest.main()
