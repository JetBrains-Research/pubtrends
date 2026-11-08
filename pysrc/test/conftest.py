import shutil
import tempfile
import pytest
from testcontainers.postgres import PostgresContainer

from pysrc.config import PubtrendsConfig

_test_config = None


@pytest.fixture(scope="session", autouse=True)
def postgres_test_container():
    global _test_config
    pm_temp_dir = tempfile.mkdtemp()
    ss_temp_dir = tempfile.mkdtemp()
    try:
        with PostgresContainer(
            "postgres:17",
            username="biolabs",
            password="mysecretpassword",
            dbname="test_pubtrends",
        ) as pg:
            override = {
                'host': pg.get_container_host_ip(),
                'port': pg.get_exposed_port(5432),
                'username': 'biolabs',
                'password': 'mysecretpassword',
                'database': 'test_pubtrends',
                'pm_lucene_index_path': pm_temp_dir,
                'ss_lucene_index_path': ss_temp_dir,
            }
            # Set class-level default so any PubtrendsConfig() created by
            # celery tasks or other code also connects to the test container.
            PubtrendsConfig._default_postgres_override = override
            _test_config = PubtrendsConfig(postgres_override=override)
            yield
            PubtrendsConfig._default_postgres_override = None
            shutil.rmtree(pm_temp_dir, ignore_errors=True)
            shutil.rmtree(ss_temp_dir, ignore_errors=True)
    except Exception:
        override = {
            'pm_lucene_index_path': pm_temp_dir,
            'ss_lucene_index_path': ss_temp_dir,
        }
        try:
            _test_config = PubtrendsConfig(postgres_override=override)
        except Exception:
            pass
        yield
        shutil.rmtree(pm_temp_dir, ignore_errors=True)
        shutil.rmtree(ss_temp_dir, ignore_errors=True)


def get_test_config():
    """Get the PubtrendsConfig connected to the TestContainers PostgreSQL instance."""
    assert _test_config is not None, "TestContainers PostgreSQL not started. Run tests via pytest."
    return _test_config
