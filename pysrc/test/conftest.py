import pytest
from testcontainers.postgres import PostgresContainer

from pysrc.config import PubtrendsConfig

_test_config = None


@pytest.fixture(scope="session", autouse=True)
def postgres_test_container():
    global _test_config
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
        }
        # Set class-level default so any PubtrendsConfig() created by
        # celery tasks or other code also connects to the test container.
        PubtrendsConfig._default_postgres_override = override
        _test_config = PubtrendsConfig(postgres_override=override)
        yield
        PubtrendsConfig._default_postgres_override = None


def get_test_config():
    """Get the PubtrendsConfig connected to the TestContainers PostgreSQL instance."""
    assert _test_config is not None, "TestContainers PostgreSQL not started. Run tests via pytest."
    return _test_config
