from pysrc.papers.db.pm_postgres_loader import PubmedPostgresLoader
from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.papers.db.ss_postgres_loader import SemanticScholarPostgresLoader
from pysrc.papers.utils import PUBMED_ARTICLE_BASE_URL, SEMANTIC_SCHOLAR_BASE_URL


class Loaders:
    @staticmethod
    def source(loader, test=False):
        # Determine source to provide correct URLs to articles,
        # see #get_loader_and_url_prefix
        # TODO: Bad design, refactor
        if isinstance(loader, PubmedPostgresLoader):
            return 'Pubmed'
        elif isinstance(loader, SemanticScholarPostgresLoader):
            return 'Semantic Scholar'
        elif not test:
            raise TypeError(f'Unknown loader {loader}')

    @staticmethod
    def get_loader(source, config):
        if PostgresConnector.postgres_configured(config):
            if source == 'Pubmed':
                return PubmedPostgresLoader(config)
            elif source == 'Semantic Scholar':
                return SemanticScholarPostgresLoader(config)
            else:
                raise ValueError(f"Unknown source {source}")
        else:
            raise ValueError("No database configured")

    @staticmethod
    def get_url_prefix(source):
        if source == 'Pubmed':
            return PUBMED_ARTICLE_BASE_URL
        elif source == 'Semantic Scholar':
            return SEMANTIC_SCHOLAR_BASE_URL
        else:
            raise ValueError(f"Unknown source {source}")
