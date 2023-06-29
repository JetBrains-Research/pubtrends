import configparser
import os

TOP_CITED_PAPERS = 50

# Features are originally taken from paper:
# 1) Which type of citation analysis generates the most accurate taxonomy of
#   scientific and technical knowledge? (https://arxiv.org/pdf/1511.05078.pdf)
#   ...bibliographic coupling (BC) was the most accurate,  followed by co-citation (CC).
#   Direct citation (DC) was a distant third among the three...

SIMILARITY_COCITATION = 5  # Limiter by number of co-citations, applied to log
SIMILARITY_BIBLIOGRAPHIC_COUPLING = 3  # Limited by number of references, applied to log
SIMILARITY_CITATION = 1  # Limited by 1 citation

# Reduce number of edges in papers graph
EMBEDDINGS_SPARSE_GRAPH_EDGES_TO_NODES = 50
VISUALIZATION_SPARSE_GRAPH_EDGES_TO_NODES = 10

# Minimal number of common references, used to reduce papers graph edges count
# Value > 1 is especially useful while analysing single paper,
# removes meaningless connections by construction
SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN = 2

# Minimal number of common references, used to reduce papers graph edges count
SIMILARITY_COCITATION_MIN = 2

# Papers embeddings is a concatenation of graph and text embeddings times corresponding factors
# Graph embeddings produce more clear topics separation, so it goes with bigger coefficient
GRAPH_EMBEDDINGS_FACTOR = 3
TEXT_EMBEDDINGS_FACTOR = 1

# Global vectorization max vocabulary size
VECTOR_WORDS = 10_000
# Terms with lower frequency will be ignored, remove rare words
VECTOR_MIN_DF = 0.001
# Terms with higher frequency will be ignored, remove abundant words
VECTOR_MAX_DF = 0.8

# Reduce embeddings dimensionality before tSNE projection
PCA_COMPONENTS = 30

# Configure number and size of topics
TOPICS_NUMBER_SMALL = dict(max_number=10, min_size=50)
TOPICS_NUMBER_MEDIUM = dict(max_number=20, min_size=20)
TOPICS_NUMBER_LARGE = dict(max_number=50, min_size=10)

# Number of top cited papers in topic picked for description computation
TOPIC_MOST_CITED_PAPERS = 50
# Number of words for topic description
TOPIC_DESCRIPTION_WORDS = 10

POPULAR_JOURNALS = 50
POPULAR_AUTHORS = 50

# Expand limit by references before filtration by citations and keywords
EXPAND_LIMIT = 5000
# Control citations count
EXPAND_CITATIONS_Q_LOW = 5
EXPAND_CITATIONS_Q_HIGH = 95
EXPAND_CITATIONS_SIGMA = 3
# Take up to fraction of top similarity
EXPAND_SIMILARITY_THRESHOLD = 0.5

# Impact of single paper when analyzing citations and mesh terms when analysing paper
SINGLE_PAPER_IMPACT = 20

EVOLUTION_MIN_PAPERS = 100
EVOLUTION_STEP = 10


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
        self.entrez_email = params['entrez_email'].strip()

        self.ss_enabled = params.getboolean('ss_enabled')
        self.ss_search_example_terms = [terms.strip() for terms in params['ss_search_example_terms'].split(';')]

        self.save_to_files_enabled = params.getboolean('save_to_files_enabled')
        self.min_search_words = params.getint('min_search_words') if not test else 0
        self.max_number_of_citations = params.getint('max_number_of_citations')
        self.max_number_of_cocitations = params.getint('max_number_of_cocitations')
        self.max_number_of_bibliographic_coupling = params.getint('max_number_of_bibliographic_coupling')
        self.max_number_to_expand = params.getint('max_number_to_expand')

        self.show_max_articles_options = [int(opt.strip()) for opt in params['show_max_articles_options'].split(',')]
        self.show_max_articles_default_value = int(params['show_max_articles_default_value'].strip())
        self.max_number_of_articles = max(self.show_max_articles_options)
        self.max_graph_size = params.getint('max_graph_size')

        # TODO Admin password - should be a better way
        self.admin_password = params['admin_password']

        # Additional modules configuration
        self.feature_authors_enabled = params.getboolean('feature_authors_enabled')
        self.feature_journals_enabled = params.getboolean('feature_journals_enabled')
        self.feature_numbers_enabled = params.getboolean('feature_numbers_enabled')
        self.feature_evolution_enabled = params.getboolean('feature_evolution_enabled')
        self.feature_review_enabled = params.getboolean('feature_review_enabled')
