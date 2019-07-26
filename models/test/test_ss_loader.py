import logging
import re
import unittest

from models.keypaper.ss_loader import SemanticScholarLoader
from models.test.articles import required_articles, extra_articles, required_citations, cit_stats_df, \
    pub_df, extra_citations


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

        cls._insert_publications()
        cls._insert_citations()
        cls._load_publications()

    @classmethod
    def _insert_publications(cls):
        articles = ', '.join(
            map(lambda article: article.to_db_publication(), (required_articles + extra_articles)))

        query = re.sub('\$values\$', articles, '''
        insert into sspublications_test(ssid, crc32id, title, year) values $values$;
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

        query = re.sub('\$values\$', citations_str, '''
        insert into sscitations_test (id_out, crc32id_out, id_in, crc32id_in) values $values$;
        ''')

        with cls.loader.conn:
            cls.loader.cursor.execute(query)

    @classmethod
    def _load_publications(cls):
        values = ', '.join(map(lambda article: article.indexes(), required_articles))
        query = re.sub('\$VALUES\$', values, '''
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

    def test_load_citations_stats(self):
        self.loader.load_citation_stats(filter_citations=False)
        actual = self.loader.cit_stats_df_from_query
        actual_sorted = actual.sort_values(by=['id', 'year']).reset_index(drop=True)
        expected_sorted = cit_stats_df.sort_values(by=['id', 'year']).reset_index(
            drop=True).astype(dtype=object)
        assert actual_sorted.equals(expected_sorted), "Citations statistics is incorrect"

    # def test_load_citations(self):
    #     self.loader.load_citations()
    #     actual = self.loader.G
    #     assert nx.is_isomorphic(actual, expected_graph), "Graph of citations is incorrect"

    # def test_load_cocitations(self):
    #     self.loader.load_cocitations()
    #     actual = self.loader.CG
    #     em = iso.numerical_edge_match('weight', 1)
    #     assert nx.is_isomorphic(actual, expected_cgraph,
    #                             edge_match=em), "Graph of co-citations is incorrect"

    @classmethod
    def tearDownClass(cls):
        query = '''
        drop table sscitations_test;
        drop table sspublications_test;'''
        with cls.loader.conn:
            cls.loader.cursor.execute(query)


if __name__ == "__main__":
    unittest.main()
