import logging
import re
import unittest

import numpy as np
from pandas.util.testing import assert_frame_equal

from models.keypaper.config import PubtrendsConfig
from models.keypaper.pm_loader import PubmedLoader
from models.test.pm_articles import REQUIRED_ARTICLES, ARTICLES, EXPECTED_PUB_DF, \
    INNER_CITATIONS, CITATIONS, EXPECTED_CIT_DF, EXPECTED_COCIT_DF, EXPECTED_CIT_STATS_DF


class TestPubmedLoader(unittest.TestCase):
    VALUES_REGEX = re.compile(r'\$VALUES\$')
    loader = PubmedLoader(pubtrends_config=PubtrendsConfig(test=True))

    @classmethod
    def setUpClass(cls):
        cls.loader.set_logger(logging.getLogger(__name__))

        # Entrez search is not tested, imitating search results
        cls.ids = list(map(lambda article: article.pmid, REQUIRED_ARTICLES))
        cls.loader.values = ', '.join(['({})'.format(i) for i in sorted(cls.ids)])

        # Reset and load data to the test database
        cls._init_database()
        cls._insert_publications()
        cls._insert_citations()

        # Get data via PubmedLoader methods
        cls.pub_df = cls.loader.load_publications()
        cls.cit_stats_df = cls.loader.load_citation_stats()
        cls.cit_df = cls.loader.load_citations()
        cls.cocit_df = cls.loader.load_cocitations()

    @classmethod
    def _init_database(cls):
        query_citations = '''
        DROP TABLE IF EXISTS PMCitations;
        CREATE TABLE PMCitations (
            pmid_out    INTEGER,
            pmid_in     INTEGER
        );
        '''

        query_publications = '''
        DROP TABLE IF EXISTS PMPublications;
        CREATE TABLE PMPublications (
            pmid        INTEGER PRIMARY KEY,
            date        DATE NULL,
            title       VARCHAR(1023),
            abstract    TEXT NULL,
            aux         JSONB
        );
        '''

        with cls.loader.conn:
            cls.loader.cursor.execute(query_citations)
            cls.loader.cursor.execute(query_publications)

    @classmethod
    def _insert_publications(cls):
        articles = ', '.join(list(map(str, ARTICLES)))

        query = re.sub(cls.VALUES_REGEX, articles, '''
            INSERT INTO PMPublications(pmid, title, aux, abstract, date) VALUES $VALUES$;
        ''')
        with cls.loader.conn:
            cls.loader.cursor.execute(query)

    @classmethod
    def _insert_citations(cls):
        citations = [f'({id_out}, {id_in})' for id_out, id_in in CITATIONS]

        query = re.sub(cls.VALUES_REGEX, ', '.join(citations), '''
            INSERT INTO PMCitations(pmid_out, pmid_in) VALUES $VALUES$;
        ''')

        with cls.loader.conn:
            cls.loader.cursor.execute(query)

    @staticmethod
    def get_row(df, article):
        return df[df['id'] == str(article.pmid)]

    def test_load_publications_count(self):
        self.assertEqual(len(self.pub_df), len(self.ids))

    def test_load_publications_check_ids(self):
        self.assertCountEqual(list(self.pub_df['id'].values), list(map(str, self.ids)))

    def test_load_publications_fill_null_abstract(self):
        self.assertFalse(np.any(self.pub_df['abstract'].isna()), msg='NaN abstract found')
        for article in REQUIRED_ARTICLES:
            if article.abstract is None:
                row = self.pub_df[self.pub_df['id'] == str(article.pmid)]
                self.assertEqual(row['abstract'].values[0], '', msg='Null abstract was filled with wrong value')

    def test_load_publications_fill_null_year(self):
        self.assertFalse(np.any(self.pub_df['year'].isna()), msg='NaN year found')
        for article in REQUIRED_ARTICLES:
            if article.date is None:
                row = self.pub_df[self.pub_df['id'] == str(article.pmid)]
                self.assertEqual(row['year'].values[0], 0, msg='Null year was filled with wrong value')

    def test_load_publications_authors(self):
        actual = []
        expected = []

        for article in REQUIRED_ARTICLES:
            expected_names = [author.name for author in article.aux.authors]
            expected.append(', '.join(expected_names))

            actual_names = self.get_row(self.pub_df, article)['authors'].values[0]
            actual.append(actual_names)

        self.assertListEqual(actual, expected, msg='Wrong authors extracted')

    def test_load_publications_journals(self):
        expected = list(map(lambda article: article.aux.journal.name, REQUIRED_ARTICLES))
        actual = list(map(lambda article: self.get_row(self.pub_df, article)['journal'].values[0], REQUIRED_ARTICLES))
        self.assertListEqual(actual, expected, msg='Wrong journals extracted')

    def test_load_publications_data_frame(self):
        assert_frame_equal(self.pub_df, EXPECTED_PUB_DF, 'Wrong publication data')

    def test_load_citation_stats_null_year_is_ignored(self):
        expected_ignored_citations = []
        for id_out, id_in in CITATIONS:
            if ARTICLES[int(id_out) - 1].date is None:
                expected_ignored_citations.append((id_out, id_in))

        ignored_diff = len(CITATIONS) - int(self.cit_stats_df['count'].sum())
        self.assertEqual(len(expected_ignored_citations), ignored_diff,
                         'Error with citations from articles with date NULL')

    def test_load_citation_stats_data_frame(self):
        # Sort to compare with expected
        self.cit_stats_df = self.cit_stats_df.sort_values(by=['id', 'year'],
                                                          ascending=[True, True]).reset_index(drop=True)
        assert_frame_equal(self.cit_stats_df, EXPECTED_CIT_STATS_DF, 'Wrong citation stats data')

    def test_load_citations_count(self):
        self.assertEqual(len(self.cit_df), len(INNER_CITATIONS), 'Wrong number of citations')

    def test_load_citations_data_frame(self):
        assert_frame_equal(self.cit_df, EXPECTED_CIT_DF, 'Wrong citation data')

    def test_load_cocitations_data_frame(self):
        # Sort to compare with expected
        self.cocit_df = self.cocit_df.sort_values(by=['citing', 'cited_1', 'cited_2'],
                                                  ascending=[True, True, True]).reset_index(drop=True)
        assert_frame_equal(self.cocit_df, EXPECTED_COCIT_DF, 'Wrong co-citation data')

    @classmethod
    def tearDownClass(cls):
        cls.loader.cursor.close()
        cls.loader.conn.close()


if __name__ == "__main__":
    unittest.main()
