import argparse
import logging
import os
import re
from contextlib import closing
from datetime import datetime

import psycopg2
import requests

from pysrc.config import PubtrendsConfig
from pysrc.papers.db.mesh_terms import parse_mesh_ascii

logger = logging.getLogger(__name__)

MESH_BASE_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh"
MESH_BATCH_SIZE = 1000

SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS pmmeshdescriptors (
           ui VARCHAR(16) PRIMARY KEY,
           heading VARCHAR(96) NOT NULL
       )""",
    """CREATE TABLE IF NOT EXISTS pmmeshterms (
           ui VARCHAR(16) NOT NULL,
           term VARCHAR(96) NOT NULL,
           is_heading BOOLEAN NOT NULL,
           term_key TEXT GENERATED ALWAYS AS
               (phraseto_tsquery('english', term)::text) STORED,
           PRIMARY KEY (ui, term)
       )""",
    """CREATE TABLE IF NOT EXISTS pmmeshtree (
           ui VARCHAR(16) NOT NULL,
           tree_number VARCHAR(128) NOT NULL,
           PRIMARY KEY (ui, tree_number)
       )""",
    """CREATE TABLE IF NOT EXISTS pmmeshmetadata (
           key VARCHAR(32) PRIMARY KEY,
           value TEXT NOT NULL
       )""",
    "ALTER TABLE pmmeshterms ADD COLUMN IF NOT EXISTS term_key TEXT "
    "GENERATED ALWAYS AS (phraseto_tsquery('english', term)::text) STORED",
    "CREATE INDEX IF NOT EXISTS pmmeshterms_term_key_idx ON pmmeshterms (term_key)",
    "CREATE INDEX IF NOT EXISTS pmmeshterms_ui_idx ON pmmeshterms (ui)",
    "CREATE INDEX IF NOT EXISTS pmmeshtree_tree_number_idx "
    "ON pmmeshtree (tree_number text_pattern_ops)",
)


def descriptor_url(name: str, year: int) -> str:
    return f"{MESH_BASE_URL}/{year}/asciimesh/{name}.bin"


def _release_exists(url: str) -> bool:
    try:
        response = requests.head(url, allow_redirects=False, timeout=30)
        try:
            return response.status_code == requests.codes.ok
        finally:
            response.close()
    except requests.RequestException:
        return False


def latest_available_year(current_year=None) -> int:
    current_year = current_year or datetime.now().year
    for year in range(current_year + 1, current_year - 3, -1):
        if _release_exists(descriptor_url(f"d{year}", year)):
            return year
    raise RuntimeError(f"No MeSH release found under {MESH_BASE_URL}")


def _records_from_url(url: str):
    response = requests.get(url, allow_redirects=False, stream=True, timeout=(30, 60))
    if response.status_code != requests.codes.ok:
        response.close()
        raise RuntimeError(f"{url} returned HTTP {response.status_code}")
    response.encoding = "utf-8"
    try:
        yield from parse_mesh_ascii(response.iter_lines(decode_unicode=True))
    finally:
        response.close()


def _records_from_file(path: str):
    with open(os.path.expanduser(path), encoding="utf-8") as source:
        yield from parse_mesh_ascii(source)


def _require_records(records, source: str):
    count = 0
    for record in records:
        count += 1
        yield record
    if count == 0:
        raise ValueError(f"No MeSH records parsed from {source}")


def replace_mesh_vocabulary(connection, records, release_year=None, supplementary=False) -> int:
    """Stage a complete release, then atomically replace the active MeSH vocabulary."""
    with connection:
        with connection.cursor() as cursor:
            for statement in SCHEMA_STATEMENTS:
                cursor.execute(statement)

            cursor.execute(
                """CREATE TEMP TABLE _mesh_stage_descriptors (
                       ui VARCHAR(16) PRIMARY KEY,
                       heading VARCHAR(96) NOT NULL
                   ) ON COMMIT DROP"""
            )
            cursor.execute(
                """CREATE TEMP TABLE _mesh_stage_terms (
                       ui VARCHAR(16) NOT NULL,
                       term VARCHAR(96) NOT NULL,
                       is_heading BOOLEAN NOT NULL,
                       PRIMARY KEY (ui, term)
                   ) ON COMMIT DROP"""
            )
            cursor.execute(
                """CREATE TEMP TABLE _mesh_stage_tree (
                       ui VARCHAR(16) NOT NULL,
                       tree_number VARCHAR(128) NOT NULL,
                       PRIMARY KEY (ui, tree_number)
                   ) ON COMMIT DROP"""
            )

            descriptors = []
            terms = []
            tree_numbers = []
            descriptor_count = 0

            def flush():
                if descriptors:
                    cursor.executemany(
                        "INSERT INTO _mesh_stage_descriptors (ui, heading) VALUES (%s, %s) "
                        "ON CONFLICT DO NOTHING",
                        descriptors,
                    )
                    descriptors.clear()
                if terms:
                    cursor.executemany(
                        "INSERT INTO _mesh_stage_terms (ui, term, is_heading) VALUES (%s, %s, %s) "
                        "ON CONFLICT DO NOTHING",
                        terms,
                    )
                    terms.clear()
                if tree_numbers:
                    cursor.executemany(
                        "INSERT INTO _mesh_stage_tree (ui, tree_number) VALUES (%s, %s) "
                        "ON CONFLICT DO NOTHING",
                        tree_numbers,
                    )
                    tree_numbers.clear()

            for record in records:
                descriptor_count += 1
                descriptors.append((record.ui, record.heading))
                terms.append((record.ui, record.heading, True))
                terms.extend((record.ui, term, False) for term in record.entry_terms)
                tree_numbers.extend((record.ui, number) for number in record.tree_numbers)
                if len(descriptors) >= MESH_BATCH_SIZE:
                    flush()

            flush()
            if descriptor_count == 0:
                raise ValueError("No MeSH records to load; preserving the existing vocabulary")

            cursor.execute("TRUNCATE pmmeshterms, pmmeshtree, pmmeshdescriptors")
            cursor.execute(
                "INSERT INTO pmmeshdescriptors (ui, heading) "
                "SELECT ui, heading FROM _mesh_stage_descriptors"
            )
            cursor.execute(
                "INSERT INTO pmmeshterms (ui, term, is_heading) "
                "SELECT ui, term, is_heading FROM _mesh_stage_terms"
            )
            cursor.execute(
                "INSERT INTO pmmeshtree (ui, tree_number) "
                "SELECT ui, tree_number FROM _mesh_stage_tree"
            )
            metadata = {
                "release_year": str(release_year) if release_year is not None else "",
                "supplementary": str(bool(supplementary)).lower(),
            }
            cursor.executemany(
                "INSERT INTO pmmeshmetadata (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                list(metadata.items()),
            )

    logger.info("Replaced MeSH vocabulary with %d records from release %s",
                descriptor_count, release_year or "local file")
    return descriptor_count


def _database_connection(config):
    return psycopg2.connect(
        host=config.postgres_host,
        port=config.postgres_port,
        dbname=config.postgres_database,
        user=config.postgres_username,
        password=config.postgres_password,
    )


def _loaded_release(connection):
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT key, value FROM pmmeshmetadata")
            metadata = dict(cursor.fetchall())
        year = int(metadata["release_year"]) if metadata.get("release_year") else None
        supplementary = metadata.get("supplementary", "false").lower() == "true"
        return year, supplementary
    except Exception:
        connection.rollback()
        return None, False


def _inferred_year(path):
    if not path:
        return None
    match = re.search(r"d(\d{4})\.bin$", os.path.basename(path), re.IGNORECASE)
    return int(match.group(1)) if match else None


def update_mesh_vocabulary(config=None, year=None, file=None, supplementary=False,
                           supplementary_file=None, force=False) -> int:
    config = config or PubtrendsConfig(test=False)
    if not config.postgres_host:
        raise RuntimeError("PostgreSQL is not configured")

    release_year = year or _inferred_year(file)
    if release_year is None and not file:
        release_year = latest_available_year()
    load_supplementary = supplementary or supplementary_file is not None
    if load_supplementary and not supplementary_file and release_year is None:
        raise ValueError("A release year is required to download supplementary concepts")

    with closing(_database_connection(config)) as connection:
        loaded_year, loaded_supplementary = _loaded_release(connection)
    already_loaded = False
    if not force and release_year is not None and loaded_year is not None:
        already_loaded = release_year <= loaded_year
    if already_loaded and load_supplementary:
        already_loaded = loaded_supplementary
    if already_loaded:
        logger.info("MeSH release %s is already loaded; skipping", release_year)
        return 0

    def records():
        if file:
            yield from _require_records(_records_from_file(file), file)
        else:
            source = descriptor_url(f"d{release_year}", release_year)
            yield from _require_records(_records_from_url(source), source)

        if load_supplementary:
            if supplementary_file:
                yield from _require_records(_records_from_file(supplementary_file), supplementary_file)
            else:
                source = descriptor_url(f"c{release_year}", release_year)
                yield from _require_records(_records_from_url(source), source)

    connection = _database_connection(config)
    try:
        return replace_mesh_vocabulary(connection, records(), release_year, load_supplementary)
    finally:
        connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Update the PostgreSQL MeSH vocabulary")
    parser.add_argument("--year", type=int, help="MeSH release year; autodetected when omitted")
    parser.add_argument("--file", help="Load a local d<year>.bin descriptor file")
    parser.add_argument("--supplementary", action="store_true",
                        help="Also load supplementary c<year>.bin concept records")
    parser.add_argument("--supplementary-file", help="Load a local supplementary concept file")
    parser.add_argument("--force", action="store_true", help="Reload an already-loaded release")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    update_mesh_vocabulary(
        year=args.year,
        file=args.file,
        supplementary=args.supplementary,
        supplementary_file=args.supplementary_file,
        force=args.force,
    )


if __name__ == "__main__":
    main()
