import configparser
import os
from collections import namedtuple


class PubtrendsConfig:
    """
    Main service configuration
    """

    # Deployment and development
    CONFIG_PATHS = ['/config', os.path.expanduser('~/.pubtrends')]

    def __init__(self, test=True):
        config_parser = configparser.ConfigParser()

        # Add fake section [params] for ConfigParser to accept the file
        for config_path in [os.path.join(p, 'config.properties') for p in self.CONFIG_PATHS]:
            if os.path.exists(config_path):
                with open(os.path.expanduser(config_path)) as f:
                    config_parser.read_string("[params]\n" + f.read())
                break
        else:
            raise RuntimeError(f'Configuration file not found among: {self.CONFIG_PATHS}')
        params = config_parser['params']

        self.postgres_host = params['postgres_host' if not test else 'test_postgres_host']
        self.postgres_port = params['postgres_port' if not test else 'test_postgres_port']
        self.postgres_username = params['postgres_username' if not test else 'test_postgres_username']
        self.postgres_password = params['postgres_password' if not test else 'test_postgres_password']
        self.postgres_database = params['postgres_database' if not test else 'test_postgres_database']

        self.pm_enabled = params.getboolean('pm_enabled')
        self.pm_search_example_terms = [terms.strip() for terms in params['pm_search_example_terms'].split(';')]

        self.ss_enabled = params.getboolean('ss_enabled')
        self.ss_search_example_terms = [terms.strip() for terms in params['ss_search_example_terms'].split(';')]

        self.min_search_words = params.getint('min_search_words') if not test else 0
        self.max_number_of_citations = params.getint('max_number_of_citations')
        self.max_number_of_cocitations = params.getint('max_number_of_cocitations')
        self.max_number_of_bibliographic_coupling = params.getint('max_number_of_bibliographic_coupling')
        self.max_number_to_expand = params.getint('max_number_to_expand')

        self.show_max_articles_options = [int(opt.strip()) for opt in params['show_max_articles_options'].split(',')]
        self.show_max_articles_default_value = int(params['show_max_articles_default_value'].strip())
        self.max_number_of_articles = max(self.show_max_articles_options)
        self.max_graph_size = params.getint('max_graph_size')

        # Max allowed pending tasks
        self.celery_max_pending_tasks = params.getint('celery_max_pending_tasks')
        # Seconds, pending task will be revoked after no polling activity
        self.celery_pending_tasks_timeout = params.getint('celery_pending_tasks_timeout')

        # TODO Admin password - should be a better way
        self.admin_password = params['admin_password']

        # Additional modules configuration
        self.feature_authors_enabled = params.getboolean('feature_authors_enabled')
        self.feature_journals_enabled = params.getboolean('feature_journals_enabled')
        self.feature_numbers_enabled = params.getboolean('feature_numbers_enabled')
        self.feature_evolution_enabled = params.getboolean('feature_evolution_enabled')
        self.feature_review_enabled = params.getboolean('feature_review_enabled')



DEFAULT_ANALYZER_SETTINGS_DICT = dict(
    SEED=20190723,

    TOP_CITED_PAPERS=50,

    # ...bibliographic coupling (BC) was the most accurate,  followed by co-citation (CC).
    # Direct citation (DC) was a distant third among the three...
    SIMILARITY_BIBLIOGRAPHIC_COUPLING=0.125,  # Limited by number of references
    SIMILARITY_COCITATION=1,  # Limiter by number of co-citations
    SIMILARITY_CITATION=0.125,  # Limited by 1 citation
    SIMILARITY_TEXT_CITATION=1,  # Limited by cosine similarity <= 1

    SIMILARITY_TEXT_CITATION_MIN=0.3,  # Minimal cosine similarity for potential text citation
    SIMILARITY_TEXT_CITATION_N=50,  # Max number of potential text citations for paper

    # Reduce number of edges in smallest communities, i.e. topics
    STRUCTURE_LOW_LEVEL_SPARSITY=0.5,
    # Limit number of edges between different topics to min number of inner edges * scale factor,
    # ignoring similarities less than average within groups
    STRUCTURE_BETWEEN_TOPICS_SPARSITY=0.2,

    TOPIC_MIN_SIZE=10,
    TOPICS_MAX_NUMBER=100,
    TOPIC_PAPERS_TFIDF=50,
    TOPIC_WORDS=20,

    VECTOR_WORDS=10000,
    VECTOR_NGRAMS=1,
    VECTOR_MIN_DF=0.01,
    VECTOR_MAX_DF=0.5,

    TOP_JOURNALS=50,
    TOP_AUTHORS=50,

    EXPAND_STEPS=2,
    # Limit citations count of expanded papers to avoid prevalence of related methods
    EXPAND_CITATIONS_SIGMA=3,
    # Take up to fraction of top similarity
    EXPAND_SIMILARITY_THRESHOLD=0.2,
    EXPAND_ZOOM_OUT=100,

    EVOLUTION_MIN_PAPERS=100,
    EVOLUTION_STEP=10,
)

AnalyzerSettings = namedtuple('AnalyzerSettings', DEFAULT_ANALYZER_SETTINGS_DICT.keys(),
                              defaults=DEFAULT_ANALYZER_SETTINGS_DICT.values())
DEFAULT_ANALYZER_SETTINGS = AnalyzerSettings()
