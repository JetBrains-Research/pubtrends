import unittest

import psycopg2
from psycopg2.extensions import TRANSACTION_STATUS_IDLE, TRANSACTION_STATUS_INTRANS

from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.test.conftest import get_test_config


class TestPostgresConnector(unittest.TestCase):

    def test_readonly_query_ends_transaction(self):
        connector = PostgresConnector(get_test_config())
        self.addCleanup(connector.close_connection)
        connector.check_connection()

        with connector.postgres_connection.cursor() as cursor:
            cursor.execute('SELECT 1')

        self.assertEqual(TRANSACTION_STATUS_IDLE, connector.postgres_connection.get_transaction_status())

    def test_readonly_connection_rejects_writes(self):
        connector = PostgresConnector(get_test_config())
        self.addCleanup(connector.close_connection)
        connector.check_connection()

        with connector.postgres_connection.cursor() as cursor:
            with self.assertRaises(psycopg2.errors.ReadOnlySqlTransaction):
                cursor.execute('CREATE TEMP TABLE connector_readonly_test (value integer)')

    def test_writer_connection_keeps_transactional_semantics(self):
        connector = PostgresConnector(get_test_config(), readonly=False)
        self.addCleanup(connector.close_connection)
        connector.check_connection()

        with connector.postgres_connection.cursor() as cursor:
            cursor.execute('CREATE TEMP TABLE connector_transaction_test (value integer)')
            cursor.execute('INSERT INTO connector_transaction_test VALUES (1)')

        self.assertEqual(TRANSACTION_STATUS_INTRANS, connector.postgres_connection.get_transaction_status())
        connector.postgres_connection.rollback()
        self.assertEqual(TRANSACTION_STATUS_IDLE, connector.postgres_connection.get_transaction_status())

        with connector.postgres_connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('pg_temp.connector_transaction_test') IS NULL")
            self.assertTrue(cursor.fetchone()[0])


if __name__ == "__main__":
    unittest.main()