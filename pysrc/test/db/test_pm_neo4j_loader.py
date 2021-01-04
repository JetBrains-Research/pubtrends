import unittest

from parameterized import parameterized

from pysrc.papers.pubtrends_config import PubtrendsConfig
from pysrc.papers.db.pm_neo4j_loader import PubmedNeo4jLoader
from pysrc.papers.db.pm_neo4j_writer import PubmedNeo4jWriter
from pysrc.test.db.abstract_test_pm_loader import AbstractTestPubmedLoader
from pysrc.test.db.pm_test_articles import REQUIRED_ARTICLES, ARTICLES, CITATIONS


class TestPubmedNeo4jLoader(unittest.TestCase, AbstractTestPubmedLoader):
    test_config = PubtrendsConfig(test=True)
    loader = PubmedNeo4jLoader(config=test_config)
    writer = PubmedNeo4jWriter(config=test_config)

    @classmethod
    def setUpClass(cls):
        cls.loader = TestPubmedNeo4jLoader.loader

        # Text search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.pmid, REQUIRED_ARTICLES))

        # Reset and load data to the test database
        writer = TestPubmedNeo4jLoader.writer
        writer.init_pubmed_database()
        writer.insert_pubmed_publications(ARTICLES)
        writer.insert_pubmed_citations(CITATIONS)

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

    @parameterized.expand([
        ('1', 1, []),
        ('2', 10, ['1']),
        ('3', 10, ['2']),
        ('4', 10, ['3']),
        ('7', 1, ['1']),
        ('7', 2, ['1', '3']),
        ('7', 10, ['1', '3', '2']),
    ])
    def test_load_references(self, pid, limit, expected_ids):
        # TODO: fix me!
        pass

    def test_expand(self):
        # TODO: fix me!
        pass


if __name__ == "__main__":
    unittest.main()
