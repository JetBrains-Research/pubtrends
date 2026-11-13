import shutil
import tempfile
import unittest
from types import SimpleNamespace
from uuid import uuid4

import psycopg2

from pysrc.papers.db.lucene_searcher import LuceneIndexManager
from pysrc.papers.db.mesh_terms import MeshRecord
from pysrc.papers.db.pm_postgres_loader import PubmedPostgresLoader
from pysrc.papers.db.ss_postgres_loader import SemanticScholarPostgresLoader
from pysrc.papers.utils import SORT_MOST_CITED
from pysrc.preprocess.update_mesh import replace_mesh_vocabulary
from pysrc.test.conftest import get_test_config


class FakeCursor:

    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, query, params=None):
        self.connection.statements.append((query, params))

    def executemany(self, query, rows):
        self.connection.statements.append((query, list(rows)))


class FakeConnection:

    def __init__(self):
        self.statements = []
        self.committed = False
        self.rolled_back = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.committed = exc_type is None
        self.rolled_back = exc_type is not None
        return False

    def cursor(self):
        return FakeCursor(self)


class TestMeshVocabularyReplacement(unittest.TestCase):

    def test_empty_input_does_not_replace_existing_vocabulary(self):
        connection = FakeConnection()

        with self.assertRaises(ValueError):
            replace_mesh_vocabulary(connection, [], release_year=2026)

        self.assertTrue(connection.rolled_back)
        self.assertFalse(any("TRUNCATE" in statement for statement, _ in connection.statements))

    def test_nonempty_input_replaces_tables_after_staging_rows(self):
        connection = FakeConnection()
        record = MeshRecord("D000001", "Example Heading", ["Example Term"], ["A01.001"])

        count = replace_mesh_vocabulary(connection, [record], release_year=2026)

        self.assertEqual(1, count)
        self.assertTrue(connection.committed)
        sql = [statement for statement, _ in connection.statements]
        truncate = next(i for i, statement in enumerate(sql) if "TRUNCATE" in statement)
        stage_inserts = [i for i, statement in enumerate(sql)
                         if "_mesh_stage" in statement and "VALUES" in statement]
        self.assertTrue(stage_inserts)
        self.assertLess(max(stage_inserts), truncate)
        self.assertTrue(any("pmmeshmetadata" in statement for statement in sql))


class TestMeshVocabularyPostgres(unittest.TestCase):

    def test_postgres_search_expands_pubmed_but_not_semantic_scholar(self):
        config = get_test_config()
        config.mesh_query_expansion = True
        connection = psycopg2.connect(
            host=config.postgres_host,
            port=config.postgres_port,
            dbname=config.postgres_database,
            user=config.postgres_username,
            password=config.postgres_password,
        )
        schema = f"mesh_test_{uuid4().hex}"
        temp_dir = tempfile.mkdtemp()
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"CREATE SCHEMA {schema}")
                cursor.execute(f"SET search_path TO {schema}")
            connection.commit()

            records = [
                MeshRecord("D900001", "Cardiac Event", ["Heart Attack", "Myocardial Infarction"], ["A01"]),
                MeshRecord("D900002", "Cardiac Injury", [], ["A01.001"]),
            ]
            self.assertEqual(2, replace_mesh_vocabulary(connection, records, release_year=2026))

            with connection.cursor() as cursor:
                cursor.execute("CREATE TABLE SSPublications (ssid TEXT, crc32id BIGINT, year INT)")
                cursor.execute("CREATE TABLE matview_sscitations (ssid TEXT, crc32id BIGINT, count INT)")
                cursor.executemany(
                    "INSERT INTO SSPublications (ssid, crc32id, year) VALUES (%s, %s, %s)",
                    [("1", 1, 2020), ("2", 2, 2021), ("3", 3, 2022)],
                )
            connection.commit()

            articles = [
                SimpleNamespace(pmid="1", title="Myocardial infarction", abstract="", keywords=[], mesh=[],
                               year=2020, type="Article", doi=""),
                SimpleNamespace(pmid="2", title="Cardiac injury", abstract="", keywords=[], mesh=[],
                               year=2021, type="Article", doi=""),
                SimpleNamespace(pmid="3", title="Heart attack", abstract="", keywords=[], mesh=[],
                               year=2022, type="Article", doi=""),
            ]
            pubmed_index = LuceneIndexManager(f"{temp_dir}/pubmed")
            pubmed_index.reset()
            pubmed_index.insert_pubmed_articles(articles)
            semantic_scholar_index = LuceneIndexManager(f"{temp_dir}/semantic_scholar")
            semantic_scholar_index.reset()
            semantic_scholar_index.insert_pubmed_articles(articles)

            pubmed_loader = PubmedPostgresLoader(config)
            pubmed_loader.postgres_connection = connection
            pubmed_loader.initialized = True
            pubmed_loader.lucene_searcher = pubmed_index

            semantic_scholar_loader = SemanticScholarPostgresLoader(config)
            semantic_scholar_loader.postgres_connection = connection
            semantic_scholar_loader.initialized = True
            semantic_scholar_loader.lucene_searcher = semantic_scholar_index

            self.assertCountEqual(["1", "2", "3"], pubmed_loader.search("heart attack", noreviews=False))
            self.assertEqual(["3"], semantic_scholar_loader.search("heart attack", noreviews=False))
            self.assertEqual(["3"], semantic_scholar_loader.search(
                "heart attack", sort=SORT_MOST_CITED, noreviews=False))
        finally:
            connection.close()
            cleanup = psycopg2.connect(
                host=config.postgres_host,
                port=config.postgres_port,
                dbname=config.postgres_database,
                user=config.postgres_username,
                password=config.postgres_password,
            )
            try:
                with cleanup.cursor() as cursor:
                    cursor.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            finally:
                cleanup.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
