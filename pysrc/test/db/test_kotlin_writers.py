import logging
import os
import subprocess
import unittest

from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.pm_neo4j_loader import PubmedNeo4jLoader
from pysrc.papers.db.pm_postgres_loader import PubmedPostgresLoader
from pysrc.papers.db.ss_neo4j_loader import SemanticScholarNeo4jLoader
from pysrc.papers.db.ss_postgres_loader import SemanticScholarPostgresLoader

PUBTRENDS_JAR = os.path.abspath(os.path.relpath('../../../build/libs/pubtrends-dev.jar', os.path.dirname(__file__)))
print(PUBTRENDS_JAR)


class TestKotlinWriters(unittest.TestCase):
    PUBMED_ARTICLE_NEO4J = \
        {'id': '1', 'title': 'Test title 1', 'abstract': 'Test abstract 2',
         'year': '1986', 'type': 'TechnicalReport',
         'keywords': 'Keyword1,Keyword2', 'mesh': 'Term1,Term2,Term3', 'doi': '',
         'aux': "{'authors': [{'name': 'Genius1', 'affiliation': []}], \
'databanks': [], 'journal': {'name': 'Pravda'}, 'language': ''}",
         'authors': 'Genius1', 'journal': 'Pravda'}

    # Slightly different order in aux
    PUBMED_ARTICLE_POSTGRES = \
        {'id': '1', 'title': 'Test title 1', 'abstract': 'Test abstract 2',
         'year': '1986', 'type': 'TechnicalReport',
         'keywords': 'Keyword1,Keyword2', 'mesh': 'Term1,Term2,Term3', 'doi': '',
         'aux': "{'authors': [{'name': 'Genius1', 'affiliation': []}], \
'journal': {'name': 'Pravda'}, 'language': '', 'databanks': []}",
         'authors': 'Genius1', 'journal': 'Pravda'}

    SEMANTIC_SCHOLAR_ARTICLE_NEO4J = \
        {'id': '03029e4427cfe66c3da6257979dc2d5b6eb3a0e4', 'crc32id': '1648497316', 'pmid': '2252909',
         'title': 'Primary Debulking Surgery Versus Neoadjuvant Chemotherapy in Stage IV Ovarian Cancer',
         'abstract': '', 'year': '2011', 'type': 'Article', 'doi': '10.1245/s10434-011-2100-x',
         'aux': "{'authors': [{'name': 'Jose Alejandro Rauh-Hain'}], "
                "'journal': {'name': 'Annals of Surgical Oncology', 'volume': '19', 'pages': '959-965'}, "
                "'links': {'s2Url': 'https://semanticscholar.org/paper/4cd223df721b722b1c40689caa52932a41fcc223',"
                " 's2PdfUrl': '', 'pdfUrls': ['https://doi.org/10.1093/llc/fqu052']}, "
                "'venue': 'Annals of Surgical Oncology'}",
         'authors': 'Jose Alejandro Rauh-Hain', 'journal': 'Annals of Surgical Oncology'}

    # Slightly different order in Aux
    SEMANTIC_SCHOLAR_ARTICLE_POSTGRES = \
        {'id': '03029e4427cfe66c3da6257979dc2d5b6eb3a0e4', 'crc32id': '1648497316', 'pmid': '2252909',
         'title': 'Primary Debulking Surgery Versus Neoadjuvant Chemotherapy in Stage IV Ovarian Cancer',
         'abstract': '', 'year': '2011', 'type': 'Article', 'doi': '10.1245/s10434-011-2100-x',
         'aux': "{'links': {'s2Url': 'https://semanticscholar.org/paper/4cd223df721b722b1c40689caa52932a41fcc223', "
                "'pdfUrls': ['https://doi.org/10.1093/llc/fqu052'], 's2PdfUrl': ''}, "
                "'venue': 'Annals of Surgical Oncology', 'authors': [{'name': 'Jose Alejandro Rauh-Hain'}], "
                "'journal': {'name': 'Annals of Surgical Oncology', 'pages': '959-965', 'volume': '19'}}",
         'authors': 'Jose Alejandro Rauh-Hain', 'journal': 'Annals of Surgical Oncology'}

    def test_kotlin_pubmed_neo4j_writer(self):
        subprocess.run(['java', '-cp', PUBTRENDS_JAR, 'org.jetbrains.bio.pubtrends.DBWriter', 'PubmedNeo4JWriter'])
        loader = PubmedNeo4jLoader(PubtrendsConfig(True))
        loader.set_progress(logging.getLogger(__name__))
        pub_df = loader.load_publications(['1'])
        actual = dict(zip(pub_df.columns, [str(v) for v in next(pub_df.iterrows())[1]]))
        self.assertEquals(actual, self.PUBMED_ARTICLE_NEO4J)
        loader.close_connection()

    def test_kotlin_pubmed_postgres_writer(self):
        subprocess.run(['java', '-cp', PUBTRENDS_JAR, 'org.jetbrains.bio.pubtrends.DBWriter', 'PubmedPostgresWriter'])
        loader = PubmedPostgresLoader(PubtrendsConfig(True))
        loader.set_progress(logging.getLogger(__name__))
        pub_df = loader.load_publications(['1'])
        actual = dict(zip(pub_df.columns, [str(v) for v in next(pub_df.iterrows())[1]]))
        self.assertEquals(actual, self.PUBMED_ARTICLE_POSTGRES)
        loader.close_connection()

    def test_kotlin_semantic_scholar_neo4j_writer(self):
        subprocess.run(
            ['java', '-cp', PUBTRENDS_JAR, 'org.jetbrains.bio.pubtrends.DBWriter', 'SemanticScholarNeo4JWriter'])
        loader = SemanticScholarNeo4jLoader(PubtrendsConfig(True))
        loader.set_progress(logging.getLogger(__name__))
        pub_df = loader.load_publications(['03029e4427cfe66c3da6257979dc2d5b6eb3a0e4'])
        actual = dict(zip(pub_df.columns, [str(v) for v in next(pub_df.iterrows())[1]]))
        print(actual)
        self.assertEquals(actual, self.SEMANTIC_SCHOLAR_ARTICLE_NEO4J)
        loader.close_connection()

    def test_kotlin_semantic_scholar_postgres_writer(self):
        subprocess.run(
            ['java', '-cp', PUBTRENDS_JAR, 'org.jetbrains.bio.pubtrends.DBWriter', 'SemanticScholarPostgresWriter'])
        loader = SemanticScholarPostgresLoader(PubtrendsConfig(True))
        loader.set_progress(logging.getLogger(__name__))
        pub_df = loader.load_publications(['03029e4427cfe66c3da6257979dc2d5b6eb3a0e4'])
        actual = dict(zip(pub_df.columns, [str(v) for v in next(pub_df.iterrows())[1]]))
        print(actual)
        self.assertEquals(actual, self.SEMANTIC_SCHOLAR_ARTICLE_POSTGRES)
        loader.close_connection()


if __name__ == "__main__":
    unittest.main()
