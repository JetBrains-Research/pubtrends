import unittest

from pandas._testing import assert_frame_equal

from pysrc.config import PubtrendsConfig
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.utils import SORT_MOST_CITED, IDS_ANALYSIS_TYPE
from pysrc.test.mock_loaders import MockLoader, \
    MockLoaderEmpty, MockLoaderSingle, BIBCOUPLING_DF, COCITATION_GROUPED_DF

PUBTRENDS_CONFIG = PubtrendsConfig(test=True)

class TestPapersAnalyzer(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        analyzer = PapersAnalyzer(MockLoader(), PUBTRENDS_CONFIG, test=True)
        ids = analyzer.search_terms(query='query')
        analyzer.analyze_papers(ids, PUBTRENDS_CONFIG.show_topics_default_value, test=True)
        cls.data = analyzer.save(IDS_ANALYSIS_TYPE, None, 'query', 'Pubmed', SORT_MOST_CITED, 10, False, None, None)

    def test_bibcoupling(self):
        assert_frame_equal(BIBCOUPLING_DF, self.data.bibliographic_coupling_df)

    def test_cocitation(self):
        assert_frame_equal(COCITATION_GROUPED_DF, self.data.cocit_grouped_df)

    def test_topic_analysis_all_nodes_assigned(self):
        nodes = self.data.papers_graph.nodes()
        for row in self.data.df.itertuples():
            if getattr(row, 'id') in nodes:
                self.assertGreaterEqual(getattr(row, 'comp'), 0)

    def test_topic_analysis_missing_nodes_set_to_default(self):
        nodes = self.data.papers_graph.nodes()
        for row in self.data.df.itertuples():
            if getattr(row, 'id') not in nodes:
                self.assertEqual(getattr(row, 'comp'), -1)

class TestPapersAnalyzerMissingPaper(unittest.TestCase):

    def test_missing_paper(self):
        analyzer = PapersAnalyzer(MockLoaderSingle(), PUBTRENDS_CONFIG, test=True)
        good_ids = list(analyzer.search_terms(query='query'))
        analyzer.analyze_papers(
            good_ids + ['non-existing-id'], PUBTRENDS_CONFIG.show_topics_default_value, test=True
        )
        self.assertEqual(good_ids, list(analyzer.df['id']))



class TestPapersAnalyzerEmpty(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.analyzer = PapersAnalyzer(MockLoaderEmpty(), PUBTRENDS_CONFIG, test=True)

    def test_setup(self):
        with self.assertRaises(Exception):
            ids = self.analyzer.search_terms(query='query')
            self.analyzer.analyze_papers(
                ids, PUBTRENDS_CONFIG.show_topics_default_value, test=True
            )


if __name__ == '__main__':
    unittest.main()
