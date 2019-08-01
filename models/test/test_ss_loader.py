import logging
import re
import unittest

import networkx as nx
from pandas.util.testing import assert_frame_equal, assert_series_equal

from models.keypaper.config import PubtrendsConfig
from models.keypaper.ss_loader import SemanticScholarLoader
from models.test.articles import required_articles, extra_articles, required_citations, cit_stats_df, \
    pub_df, extra_citations, expected_graph, cocitations_df


class TestSemanticScholarLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.loader = SemanticScholarLoader(
            pubtrends_config=PubtrendsConfig(test=True),
            publications_table='sspublications_test',
            citations_table='sscitations_test',
            temp_ids_table='temp_ssids_test')
        cls.loader.set_logger(logging.getLogger(__name__))
        query_citations = '''
        drop table if exists sscitations_test;
        create table sscitations_test (
            crc32id_out integer,
            crc32id_in  integer,
            id_out      varchar(40) not null,
            id_in       varchar(40) not null
        );
        create index if not exists sscitations_test_crc32id_out_crc32id_in_index
        on sscitations_test (crc32id_out, crc32id_in);
        '''

        query_publications = '''
        drop table if exists sspublications_test;
        create table sspublications_test(
            ssid    varchar(40) not null,
            crc32id integer     not null,
            title   varchar(1023),
            year    integer
        );
        create index if not exists sspublications_test_crc32id_index
        on sspublications_test (crc32id);
        '''

        with cls.loader.conn:
            cls.loader.cursor.execute(query_citations)
            cls.loader.cursor.execute(query_publications)

        cls.VALUES_REGEX = re.compile(r'\$VALUES\$')

        cls._insert_publications()
        cls._insert_citations()
        cls._load_publications()

        cls.citations_stats = cls._load_citations_stats()
        cls.citations_graph = cls._load_citations_graph()
        cls.cocitations_df = cls._load_cocitations()

    @classmethod
    def _insert_publications(cls):
        articles = ', '.join(
            map(lambda article: article.to_db_publication(), (required_articles + extra_articles)))

        query = re.sub(cls.VALUES_REGEX, articles, '''
        insert into sspublications_test(ssid, crc32id, title, year) values $VALUES$;
        ''')
        with cls.loader.conn:
            cls.loader.cursor.execute(query)

    @classmethod
    def _insert_citations(cls):
        citations_str = ', '.join(
            "('{0}', {1}, '{2}', {3})".format(citation[0].ssid, citation[0].crc32id,
                                              citation[1].ssid,
                                              citation[1].crc32id) for citation in
            (required_citations + extra_citations))

        query = re.sub(cls.VALUES_REGEX, citations_str, '''
        insert into sscitations_test (id_out, crc32id_out, id_in, crc32id_in) values $VALUES$;
        ''')

        with cls.loader.conn:
            cls.loader.cursor.execute(query)

    @classmethod
    def _load_publications(cls):
        values = ', '.join(map(lambda article: article.indexes(), required_articles))
        query = re.sub(cls.VALUES_REGEX, values, '''
                DROP TABLE IF EXISTS temp_ssids_test;
                WITH vals(ssid, crc32id) AS (VALUES $VALUES$)
                SELECT crc32id, ssid INTO temporary table temp_ssids_test FROM vals;
                DROP INDEX IF EXISTS temp_ssids_index;
                CREATE INDEX temp_ssids_index ON temp_ssids_test USING btree (crc32id);''')

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
    def _load_citations_graph(cls):
        G = cls.loader.load_citations()
        return G

    @classmethod
    def _load_cocitations(cls):
        cocit_grouped_df = cls.loader.load_cocitations()
        # flatten dataframe with multi index
        actual = cocit_grouped_df.sort_values(by=['cited_1', 'cited_2']).reset_index(drop=True)
        for col in actual['citing'].columns:
            actual[col] = actual['citing'][col]
        actual.drop(['citing', 'year'], axis=1, level=0, inplace=True)
        actual.columns = [col[0] for col in actual.columns]
        return actual

    def test_citations_stats_rows(self):
        expected_rows = cit_stats_df.shape[0]
        actual_rows = self.citations_stats.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in citations statistics is incorrect")

    def test_load_citations_stats(self):
        assert_frame_equal(self.citations_stats, cit_stats_df, "Citations statistics is incorrect")

    def test_citation_nodes_amount(self):
        expected_number_of_nodes = len(expected_graph.nodes())
        actual_number_of_nodes = len(self.citations_graph.nodes())
        self.assertEqual(expected_number_of_nodes, actual_number_of_nodes,
                         "Amount of nodes in citation graph is incorrect")

    def test_citations_nodes(self):
        expected_nodes = expected_graph.nodes()
        actual_nodes = self.citations_graph.nodes()
        self.assertEqual(expected_nodes, actual_nodes, "Nodes in citation graph are incorrect")

    def test_load_citations(self):
        self.assertTrue(nx.is_isomorphic(expected_graph, self.citations_graph), "Graph of citations is incorrect")

    def test_cocitations_df_rows(self):
        expected_rows = cocitations_df.shape[0]
        actual_rows = self.cocitations_df.shape[0]
        self.assertEqual(expected_rows, actual_rows, "Number of rows in co-citations dataframe is incorrect")

    def test_cocitations_total(self):
        expected_total = cocitations_df['total']
        actual_total = self.cocitations_df['total']
        assert_series_equal(expected_total, actual_total, "Total amount of co-citations is incorrect")

    def test_load_cocitations(self):
        expected_cocit_df = cocitations_df
        actual = self.cocitations_df
        assert_frame_equal(expected_cocit_df, actual, "Co-citations dataframe is incorrect")

    @classmethod
    def tearDownClass(cls):
        query = '''
        drop table sscitations_test;
        drop table sspublications_test;'''
        with cls.loader.conn:
            cls.loader.cursor.execute(query)


if __name__ == "__main__":
    unittest.main()
