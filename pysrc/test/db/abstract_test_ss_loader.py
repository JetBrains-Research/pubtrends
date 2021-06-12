from abc import ABCMeta, abstractmethod

from pandas.testing import assert_frame_equal
from parameterized import parameterized

from pysrc.papers.utils import SORT_MOST_RECENT, SORT_MOST_CITED
from pysrc.test.db.ss_test_articles import REQUIRED_CITATIONS, \
    EXPECTED_CIT_STATS_DF, EXPECTED_CIT_DF, EXPECTED_COCIT_DF, ARTICLES_LIST, EXPANDED_ARTICLES_DF


# Don't make it subclass of unittest.TestCase to avoid tests execution
class AbstractTestSemanticScholarLoader(metaclass=ABCMeta):

    # @classmethod
    # def setUpClass(cls):
    # TODO: example of initialization
    #
    #     cls.loader = Loader(config=PubtrendsConfig(test=True))
    #     cls.loader.set_progress(logging.getLogger(__name__))
    #
    #     # Text search is not tested, imitating search results
    #     cls.ids = list(map(lambda article: article.ssid, required_articles))
    #
    #     writer = Writer()
    #     writer.init_database()
    #     writer.insert_publications(required_articles + extra_articles)
    #     writer.insert_citations(required_citations + extra_citations)
    #
    #     # Get data via loader methods
    #     cls.pub_df = cls.loader.load_publications(cls.ids)
    #     cls.cit_stats_df = cls.loader.load_citation_stats(cls.ids)
    #     cls.cit_df = cls.loader.load_citations(cls.ids)
    #     cls.cocit_df = cls.loader.load_cocitations(cls.ids)

    @abstractmethod
    def getLoader(self):
        """:return Loader instance"""

    @abstractmethod
    def getCitationsStatsDataframe(self):
        """:return citations stats pandas dataframe"""

    @abstractmethod
    def getCitationsDataframe(self):
        """:return citations dataframe"""

    @abstractmethod
    def getCoCitationsDataframe(self):
        """:return co-citations dataframe"""

