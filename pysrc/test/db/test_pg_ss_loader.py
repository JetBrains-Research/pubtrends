import logging
import unittest
from abc import ABCMeta

from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.pg_ss_loader import SemanticScholarLoader
from pysrc.papers.db.pg_ss_writer import SemanticScholarWriter
from pysrc.test.db.ss_test_articles import required_articles, extra_articles, required_citations, \
    extra_citations


class TestSemanticScholarLoader(unittest.TestCase, metaclass=ABCMeta):
    loader = SemanticScholarLoader(PubtrendsConfig(test=True))

    @classmethod
    def setUpClass(cls):
        cls.loader.set_progress(logging.getLogger(__name__))

        # Text search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.ssid, required_articles))

        writer = SemanticScholarWriter()
        writer.init_semantic_scholar_database()
        writer.insert_semantic_scholar_publications(required_articles + extra_articles)
        writer.insert_semantic_scholar_citations(required_citations + extra_citations)

        # Get data via SemanticScholar methods
        cls.pub_df = cls.loader.load_publications(cls.ids)
        cls.cit_stats_df = cls.loader.load_citation_stats(cls.ids)
        cls.cit_df = cls.loader.load_citations(cls.ids)
        cls.cocit_df = cls.loader.load_cocitations(cls.ids)


if __name__ == "__main__":
    unittest.main()
