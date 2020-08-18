import logging
import unittest
from abc import ABCMeta

from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.pm_loader import PubmedLoader
from pysrc.papers.db.pm_writer import PubmedWriter
from pysrc.test.db.pm_test_articles import REQUIRED_ARTICLES, ARTICLES, CITATIONS


class TestPubmedLoader(unittest.TestCase, metaclass=ABCMeta):
    loader = PubmedLoader(config=PubtrendsConfig(test=True))

    @classmethod
    def setUpClass(cls):
        cls.loader.set_progress(logging.getLogger(__name__))

        # Text search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.pmid, REQUIRED_ARTICLES))

        # Reset and load data to the test database
        writer = PubmedWriter()
        writer.init_pubmed_database()
        writer.insert_pubmed_publications(ARTICLES)
        writer.insert_pubmed_citations(CITATIONS)

        # Get data via PubmedLoader methods
        cls.pub_df = cls.loader.load_publications(cls.ids)
        cls.cit_stats_df = cls.loader.load_citation_stats(cls.ids)
        cls.cit_df = cls.loader.load_citations(cls.ids)
        cls.cocit_df = cls.loader.load_cocitations(cls.ids)


if __name__ == "__main__":
    unittest.main()
