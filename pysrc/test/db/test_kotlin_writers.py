import os
import subprocess
import unittest

from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.pm_neo4j_loader import PubmedNeo4jLoader
from pysrc.papers.db.pm_postgres_loader import PubmedPostgresLoader
from pysrc.papers.db.ss_neo4j_loader import SemanticScholarNeo4jLoader
from pysrc.papers.db.ss_postgres_loader import SemanticScholarPostgresLoader

PUBTRENDS_JAR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build/libs/pubtrends-dev.jar'))


class TestKotlinWriters(unittest.TestCase):
    PUBMED_ARTICLE = \
        {'abstract': 'Test abstract 2',
         'authors': 'Genius1',
         'aux': {'authors': [{'affiliation': [], 'name': 'Genius1'}],
                 'databanks': [],
                 'journal': {'name': 'Pravda'},
                 'language': ''},
         'doi': '',
         'id': '1',
         'journal': 'Pravda',
         'keywords': 'Keyword1,Keyword2',
         'mesh': 'Term1,Term2,Term3',
         'title': 'Test title 1',
         'type': 'TechnicalReport',
         'year': 1986}

    SEMANTIC_SCHOLAR_ARTICLE = \
        {'abstract': 'Test abstract 2',
         'authors': 'Genius',
         'aux': {'authors': [{'name': 'Genius'}],
                 'journal': {'name': 'Nature Aging', 'pages': '1-6', 'volume': '1'},
                 'links': {'pdfUrls': ['https://doi.org/10.1101/2020.05.10.087023'],
                           's2PdfUrl': '',
                           's2Url': ''},
                 'venue': 'Nature'},
         'crc32id': 1648497316,
         'doi': '10.1101/2020.05.10.087023',
         'id': '03029e4427cfe66c3da6257979dc2d5b6eb3a0e4',
         'journal': 'Nature Aging',
         'pmid': '2252909',
         'title': 'Test title 1',
         'type': 'Article',
         'year': 2020}

    def test_kotlin_pubmed_neo4j_writer(self):
        print(PUBTRENDS_JAR)
        self.assertTrue(os.path.exists(PUBTRENDS_JAR), f'File not found: {PUBTRENDS_JAR}')
        subprocess.run(['java', '-cp', PUBTRENDS_JAR, 'org.jetbrains.bio.pubtrends.DBWriter', 'PubmedNeo4JWriter'])
        loader = PubmedNeo4jLoader(PubtrendsConfig(True))
        try:
            pub_df = loader.load_publications(['1'])
            actual = dict(zip(pub_df.columns, next(pub_df.iterrows())[1]))
            self.assertEqual(actual, self.PUBMED_ARTICLE)
        finally:
            loader.close_connection()

    def test_kotlin_pubmed_postgres_writer(self):
        print(PUBTRENDS_JAR)
        self.assertTrue(os.path.exists(PUBTRENDS_JAR), f'File not found: {PUBTRENDS_JAR}')
        subprocess.run(['java', '-cp', PUBTRENDS_JAR, 'org.jetbrains.bio.pubtrends.DBWriter', 'PubmedPostgresWriter'])
        loader = PubmedPostgresLoader(PubtrendsConfig(True))
        try:
            pub_df = loader.load_publications(['1'])
            actual = dict(zip(pub_df.columns, next(pub_df.iterrows())[1]))
            self.assertEqual(actual, self.PUBMED_ARTICLE)
        finally:
            loader.close_connection()

    def test_kotlin_semantic_scholar_neo4j_writer(self):
        print(PUBTRENDS_JAR)
        self.assertTrue(os.path.exists(PUBTRENDS_JAR), f'File not found: {PUBTRENDS_JAR}')
        subprocess.run(
            ['java', '-cp', PUBTRENDS_JAR, 'org.jetbrains.bio.pubtrends.DBWriter', 'SemanticScholarNeo4JWriter'])
        loader = SemanticScholarNeo4jLoader(PubtrendsConfig(True))
        try:
            pub_df = loader.load_publications(['03029e4427cfe66c3da6257979dc2d5b6eb3a0e4'])
            actual = dict(zip(pub_df.columns, next(pub_df.iterrows())[1]))
            expected = self.SEMANTIC_SCHOLAR_ARTICLE
            expected.update(dict(keywords='', mesh=''))
            self.assertEqual(actual, expected)
        finally:
            loader.close_connection()

    def test_kotlin_semantic_scholar_postgres_writer(self):
        print(PUBTRENDS_JAR)
        self.assertTrue(os.path.exists(PUBTRENDS_JAR), f'File not found: {PUBTRENDS_JAR}')
        subprocess.run(
            ['java', '-cp', PUBTRENDS_JAR, 'org.jetbrains.bio.pubtrends.DBWriter', 'SemanticScholarPostgresWriter'])
        loader = SemanticScholarPostgresLoader(PubtrendsConfig(True))
        try:
            pub_df = loader.load_publications(['03029e4427cfe66c3da6257979dc2d5b6eb3a0e4'])
            actual = dict(zip(pub_df.columns, next(pub_df.iterrows())[1]))
            expected = self.SEMANTIC_SCHOLAR_ARTICLE
            expected.update(dict(keywords='', mesh=''))
            self.assertEqual(actual, expected)
        finally:
            loader.close_connection()


if __name__ == "__main__":
    unittest.main()
