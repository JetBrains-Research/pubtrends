import argparse
import logging
import os
import psycopg2
from pysrc.config import PubtrendsConfig
from pysrc.papers.db.lucene_searcher import LuceneIndexManager

logger = logging.getLogger(__name__)


def update_lucene_index(min_year=None, reset=False, batch_size=10000):
    config = PubtrendsConfig(test=False)
    manager = LuceneIndexManager(config.pm_lucene_index_path, config.bm25_k1, config.bm25_b)
    if reset:
        print(f"Resetting Lucene index at {config.pm_lucene_index_path}")
        manager.reset()

    ix = manager.get_index()
    connection_string = f"host={config.postgres_host} port={config.postgres_port} dbname={config.postgres_database} user={config.postgres_username} password={config.postgres_password}".strip()

    print(f"Connecting to PostgreSQL and syncing Lucene index at {config.pm_lucene_index_path}...")
    with psycopg2.connect(connection_string) as connection:
        connection.set_session(readonly=True)
        with connection.cursor(name="pm_lucene_cursor") as cursor:
            if min_year is not None:
                query = "SELECT pmid, title, year, abstract, type, keywords, mesh, doi FROM PMPublications WHERE year >= %s ORDER BY pmid;"
                cursor.execute(query, (min_year,))
            else:
                query = "SELECT pmid, title, year, abstract, type, keywords, mesh, doi FROM PMPublications ORDER BY pmid;"
                cursor.execute(query)

            total = 0
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                writer = ix.writer()
                for pmid, title, year, abstract, ptype, keywords, mesh, doi in rows:
                    writer.update_document(
                        id=str(pmid),
                        title=str(title or ''),
                        abstract=str(abstract or ''),
                        keywords=str(keywords or ''),
                        mesh=str(mesh or ''),
                        year=int(year) if year is not None else None,
                        type=str(ptype or ''),
                        doi=str(doi or '')
                    )
                writer.commit()
                total += len(rows)
                print(f"Indexed {total} documents...", end='\r')
            print(f"\nSuccessfully indexed {total} documents in Lucene.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update Lucene keyword index from PostgreSQL')
    parser.add_argument('--min-year', type=int, default=None, help='Minimum year to process')
    parser.add_argument('--reset', action='store_true', help='Reset the index before updating')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size for database fetching')
    args = parser.parse_args()

    update_lucene_index(min_year=args.min_year, reset=args.reset, batch_size=args.batch_size)
