import logging
import unittest

from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.ss_neo4j_loader import SemanticScholarNeo4jLoader
from pysrc.papers.db.ss_neo4j_writer import SemanticScholarNeo4jWriter
from pysrc.test.db.abstract_test_ss_loader import AbstractTestSemanticScholarLoader
from pysrc.test.db.ss_test_articles import required_articles, extra_articles, required_citations, \
    extra_citations


class TestSemanticScholarNeo4jLoader(unittest.TestCase, AbstractTestSemanticScholarLoader):
    loader = SemanticScholarNeo4jLoader(config=PubtrendsConfig(test=True))

    @classmethod
    def setUpClass(cls):
        cls.loader = TestSemanticScholarNeo4jLoader.loader
        cls.loader.set_progress(logging.getLogger(__name__))

        # Text search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.ssid, required_articles))

        writer = SemanticScholarNeo4jWriter()
        writer.init_semantic_scholar_database()
        writer.insert_semantic_scholar_publications(required_articles + extra_articles)
        writer.insert_semantic_scholar_citations(required_citations + extra_citations)

        # Get data via loader methods
        cls.pub_df = cls.loader.load_publications(cls.ids)
        cls.cit_stats_df = cls.loader.load_citation_stats(cls.ids)
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