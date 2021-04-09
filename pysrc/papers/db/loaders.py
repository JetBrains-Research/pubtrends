from pysrc.papers.db.pm_postgres_loader import PubmedPostgresLoader
from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.papers.db.ss_postgres_loader import SemanticScholarPostgresLoader
from pysrc.papers.utils import PUBMED_ARTICLE_BASE_URL, SEMANTIC_SCHOLAR_BASE_URL
from pysrc.prediction.ss_arxiv_loader import SSArxivLoader
from pysrc.prediction.ss_pubmed_loader import SSPubmedLoader


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
        elif isinstance(loader, SSArxivLoader):
            return 'SSArxiv'
        elif isinstance(loader, SSPubmedLoader):
            return 'SSPubmed'
        elif not test:
            raise TypeError(f'Unknown loader {loader}')

    @staticmethod
    def get_loader_and_url_prefix(source, config):
        if PostgresConnector.postgres_configured(config):
            if source == 'Pubmed':
                return PubmedPostgresLoader(config), PUBMED_ARTICLE_BASE_URL
            elif source == 'Semantic Scholar':
                return SemanticScholarPostgresLoader(config), SEMANTIC_SCHOLAR_BASE_URL
            else:
                raise ValueError(f"Unknown source {source}")
        else:
            raise ValueError("No database configured")

    @staticmethod
    def get_loader(source, config):
        return Loaders.get_loader_and_url_prefix(source, config)[0]
