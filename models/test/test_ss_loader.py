import logging
import re
import unittest

from pandas.util.testing import assert_frame_equal

from models.keypaper.config import PubtrendsConfig
from models.keypaper.ss_loader import SemanticScholarLoader
from models.test.ss_articles import required_articles, extra_articles, required_citations, cit_stats_df, \
    pub_df, cit_df, extra_citations, cocitations_df


class TestSemanticScholarLoader(unittest.TestCase):
    VALUES_REGEX = re.compile(r'\$VALUES\$')
    loader = None

    @classmethod
    def setUpClass(cls):
        cls.loader = SemanticScholarLoader(pubtrends_config=PubtrendsConfig(test=True))
        cls.loader.set_logger(logging.getLogger(__name__))

        cls.init_database()
        cls.insert_publications()
        cls.insert_citations()
        cls._load_publications()

        cls.citations_stats = cls._load_citations_stats()
        cls.citations = cls.loader.load_citations()
        cls.cocitations_df = cls._load_cocitations()

    @classmethod
    def init_database(cls):
        query_citations = '''
                drop table if exists sscitations;
                create table sscitations (
                    crc32id_out integer,
                    crc32id_in  integer,
                    id_out      varchar(40) not null,
                    id_in       varchar(40) not null
                );
                create index if not exists sscitations_crc32id_out_crc32id_in_index
                on sscitations (crc32id_out, crc32id_in);
                '''

        query_publications = '''
                drop table if exists sspublications;
                create table sspublications(
                    ssid    varchar(40) not null,
                    crc32id integer     not null,
                    title   varchar(1023),
                    year    integer
                );
                create index if not exists sspublications_crc32id_index
                on sspublications (crc32id);
                '''

        with cls.loader.conn:
            cls.loader.cursor.execute(query_citations)
            cls.loader.cursor.execute(query_publications)

    @classmethod
    def insert_publications(cls):
        articles = ', '.join(
            map(lambda article: article.to_db_publication(), (required_articles + extra_articles)))

        query = re.sub(cls.VALUES_REGEX, articles, '''
        insert into sspublications(ssid, crc32id, title, year) values $VALUES$;
        ''')
        with cls.loader.conn:
            cls.loader.cursor.execute(query)

    @classmethod
    def insert_citations(cls):
        citations_str = ', '.join(
            "('{0}', {1}, '{2}', {3})".format(citation[0].ssid, citation[0].crc32id,
                                              citation[1].ssid, citation[1].crc32id) for citation in
            (required_citations + extra_citations))

        query = re.sub(cls.VALUES_REGEX, citations_str, '''
        insert into sscitations (id_out, crc32id_out, id_in, crc32id_in) values $VALUES$;
        ''')

        with cls.loader.conn:
            cls.loader.cursor.execute(query)

    @classmethod
    def _load_publications(cls):
        values = ', '.join(map(lambda article: article.indexes(), required_articles))
        query = re.sub(cls.VALUES_REGEX, values, '''
                DROP TABLE IF EXISTS temp_ssids;
                WITH vals(ssid, crc32id) AS (VALUES $VALUES$)
                SELECT crc32id, ssid INTO temporary table temp_ssids FROM vals;
                DROP INDEX IF EXISTS temp_ssids_index;
                CREATE INDEX temp_ssids_index ON temp_ssids USING btree (crc32id);''')

        with cls.loader.conn:
            cls.loader.cursor.execute(query)

        cls.loader.ids = list(map(lambda article: article.ssid, required_articles))
        cls.loader.crc32ids = list(map(lambda article: article.crc32id, required_articles))
        cls.loader.values = ', '.join(
            ['({0}, \'{1}\')'.format(i, j) for (i, j) in
             zip(cls.loader.crc32ids, cls.loader.ids)])

        cls.loader.pub_df = pub_df

    @classmethod
    def _load_citations_stats(cls):
        cit_stats_df_from_query = cls.loader.load_citation_stats()
        return cit_stats_df_from_query.sort_values(by=['id', 'year']).reset_index(drop=True)

    @classmethod
    def _load_cocitations(cls):
        cocitations = cls.loader.load_cocitations()
        return cocitations.sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(drop=True)

    def test_citations_stats_rows(self):
        expected_rows = cit_stats_df.shape[0]
        actual_rows = self.citations_stats.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in citations statistics is incorrect")

    def test_load_citation_stats_data_frame(self):
        assert_frame_equal(self.citations_stats, cit_stats_df, "Citations statistics is incorrect")

    def test_load_citation_stats_null_year_is_ignored(self):
        expected_ignored_citations = 0
        citations = required_citations + extra_citations
        for article_out, article_in in citations:
            if article_out.year is None:
                expected_ignored_citations += 1

        ignored_diff = len(citations) - int(cit_stats_df['count'].sum())
        self.assertEqual(expected_ignored_citations, ignored_diff,
                         'Error with citations from articles with year NULL')

    def test_load_citations_count(self):
        self.assertEqual(len(self.citations), len(required_citations), 'Wrong number of citations')

    def test_load_citations_data_frame(self):
        assert_frame_equal(self.citations, cit_df, 'Wrong citation data')

    def test_load_cocitations_count(self):
        expected_rows = cocitations_df.shape[0]
        actual_rows = self.cocitations_df.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in co-citations dataframe is incorrect")

    def test_load_cocitations_data_frame(self):
        expected_cocit_df = cocitations_df
        actual = self.cocitations_df
        assert_frame_equal(expected_cocit_df, actual, "Co-citations dataframe is incorrect")

    @classmethod
    def tearDownClass(cls):
        query = '''
        drop table sscitations;
        drop table sspublications;'''
        with cls.loader.conn:
            cls.loader.cursor.execute(query)

        cls.loader.close_connection()


if __name__ == "__main__":
    unittest.main()
