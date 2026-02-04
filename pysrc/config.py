import configparser
import os
import re

###################
# Search settings #
###################
SHOW_MAX_ARTICLES_OPTIONS = [200, 1000, 2000]
SHOW_MAX_ARTICLES_DEFAULT = 1000

# Configure the number and size of topics
SHOW_TOPICS_OPTIONS = [5, 10, 20]
SHOW_TOPICS_DEFAULT = 10

# Number of steps in expanding
PAPER_EXPAND_STEPS = 2

#####################
## Analysis config ##
#####################
MAX_SEARCH_TIME_SEC = 300

# Postgresql loader options
MAX_NUMBER_OF_PAPERS = 1_000_000
MAX_NUMBER_OF_CITATIONS = 100_000_000
MAX_NUMBER_OF_COCITATIONS = 10_000_000
MAX_NUMBER_OF_BIBLIOGRAPHIC_COUPLING = 10_000_000

# Features are originally taken from paper:
# 1) Which type of citation analysis generates the most accurate taxonomy of
#   scientific and technical knowledge? (https://arxiv.org/pdf/1511.05078.pdf)
#   ...bibliographic coupling (BC) was the most accurate,  followed by co-citation (CC).
#   Direct citation (DC) was a distant third among the three...

SIMILARITY_COCITATION = 10  # x number of co-citations
SIMILARITY_BIBLIOGRAPHIC_COUPLING = 3  # x number of references
SIMILARITY_CITATION = 1  # x 0 - 1 citations

SIMILARITY_TEXT = 20  # cosine similarity between texts embeddings -1 - 1

# A minimal number of common references, used to reduce papers graph edges count
# Value > 1 is especially useful while analysing a single paper,
# removes meaningless connections by construction
SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN = 2

# Minimal number of common references, used to reduce papers graph edges count
SIMILARITY_COCITATION_MIN = 2

# Maximal graph size for analysis
MAX_GRAPH_SIZE = 10_000

# Number of text similarity edges in papers graph
GRAPH_TEXT_SIMILARITY_EDGES = 100

# Number of edges in papers graph for visualization
VISUALIZATION_GRAPH_EDGES = 10

# PCA for visualizing
PCA_VARIANCE = 0.9

# Maximal dataset size for agglomerative clustering
MAX_AGGLOMERATIVE_CLUSTERING = 5000

# Global vectorization max vocabulary size
VECTOR_WORDS = 10_000

# Terms with lower frequency will be ignored, remove rare words
VECTOR_MIN_DF = 0.001

# Terms with higher frequency will be ignored, remove abundant words
VECTOR_MAX_DF = 0.8

ANALYSIS_CHUNK = 1000

#############################
## Embeddings settings #####
#############################

# Size of a chunk for global text embeddings used for clustering
EMBEDDINGS_CHUNK_SIZE = 512
EMBEDDINGS_SENTENCE_OVERLAP = 0

# Size of a chunk for precise questioning
EMBEDDINGS_QUESTIONS_CHUNK_SIZE = 64
EMBEDDINGS_QUESTIONS_SENTENCE_OVERLAP = 1

FAISS_CLUSTERS = 1024
FAISS_SUBQUANTIZERS = 32
FAISS_SUBQUANTIZER_BITS = 8
FAISS_INDEX_NPROBE = 100  # 10%
FAISS_EMBEDDIGNS_SAMPLE_PROBES = 100_000

#############################
## Expanding by references ##
#############################

# Expand the limit by references before filtration by citations and keywords
PAPER_EXPAND_LIMIT = 5000
# Fraction of papers to expand by similar papers embeddings
PAPER_EXPAND_SEMANTIC = 0.25

# Control citation count while expanding
EXPAND_CITATIONS_Q_LOW = 10
EXPAND_CITATIONS_Q_HIGH = 90
EXPAND_CITATIONS_SIGMA = 5

# Impact of a single paper when analyzing citations and mesh terms when analysing paper
EXPAND_SINGLE_PAPER_IMPACT = 50

#####################
## Node2vec config ##
#####################
NODE2VEC_EMBEDDINGS_VECTOR_LENGTH = 64
NODE2VEC_P = 5.0
NODE2VEC_Q = 2.0

# Increasing the number of walks increases node2vec representation accuracy
NODE2VEC_GRAPH_EDGES = 50
NODE2VEC_WALKS_PER_NODE = 100
NODE2VEC_WALK_LENGTH = 30
NODE2VEC_WORD2VEC_WINDOW = 5
NODE2VEC_WORD2VEC_EPOCHS = 3

#####################
## Word2vec config ##
#####################

WORD2VEC_EMBEDDINGS_LENGTH = 128
WORD2VEC_WINDOW = 5
WORD2VEC_EPOCHS = 3

#################
## Plot config ##
#################

# Number of top-cited papers to show
TOP_CITED_PAPERS = 50

# Number of words for topic description
TOPIC_DESCRIPTION_WORDS = 5

MAX_AUTHOR_LENGTH = 100
MAX_JOURNAL_LENGTH = 100

MAX_LINEAR_AXIS = 100

PLOT_WIDTH = 870
PAPERS_PLOT_WIDTH = 670

SHORT_PLOT_HEIGHT = 300
TALL_PLOT_HEIGHT = 600
PLOT_HEIGHT = 375

WORD_CLOUD_WIDTH = 250
WORD_CLOUD_HEIGHT = 300

WORD_CLOUD_KEYWORDS = 15

############
# Features #
############

POPULAR_AUTHORS = 200
POPULAR_JOURNALS = 200

QUESTIONS_RELEVANCE_THRESHOLD = 0.5
QUESTIONS_ANSWERS_TOP_N = 50



class PubtrendsConfig:
    """
    Main service configuration
    """

    # Deployment and development
    CONFIG_PATHS = ['/config', os.path.expanduser('~/.pubtrends')]

    def __init__(self, test=True):
        config_parser = configparser.ConfigParser()

        # Add a fake section [params] for ConfigParser to accept the file
        for config_path in [os.path.join(p, 'config.properties') for p in self.CONFIG_PATHS]:
            if os.path.exists(config_path):
                with open(os.path.expanduser(config_path)) as f:
                    config_parser.read_string("[params]\n" + f.read())
                break
        else:
            raise RuntimeError(f'Configuration file not found among: {self.CONFIG_PATHS}')
        params = config_parser['params']

        # DB config
        self.postgres_host = params['postgres_host' if not test else 'test_postgres_host']
        self.postgres_port = params['postgres_port' if not test else 'test_postgres_port']
        self.postgres_username = params['postgres_username' if not test else 'test_postgres_username']
        self.postgres_password = params['postgres_password' if not test else 'test_postgres_password']
        self.postgres_database = params['postgres_database' if not test else 'test_postgres_database']

        # Embeddings DB config
        self.embeddings_postgres_host = params['embeddings_postgres_host']
        self.embeddings_postgres_port = params['embeddings_postgres_port']
        self.embeddings_postgres_username = params['embeddings_postgres_username']
        self.embeddings_postgres_password = params['embeddings_postgres_password']
        self.embeddings_postgres_database = params['embeddings_postgres_database']

        # Source config
        self.pm_enabled = params.getboolean('pm_enabled')
        self.pm_search_example_terms = [terms.strip() for terms in params['pm_search_example_terms'].split(';')]
        self.pm_paper_examples = [terms.strip().split('=') for terms in params['pm_paper_examples'].split(';')]
        self.entrez_email = params['entrez_email'].strip()

        self.ss_enabled = params.getboolean('ss_enabled')
        self.ss_search_example_terms = [terms.strip() for terms in params['ss_search_example_terms'].split(';')]

        # Model name used for embeddings
        self.sentence_transformer_model = params['sentence_transformer_model']
        self.embeddings_model_name = re.sub('[^a-zA-Z0-9]+', '_', self.sentence_transformer_model)
        self.embeddings_dimension = params.getint('embeddings_dimension')

        # Additional features configuration
        self.feature_semantic_search_enabled = params.getboolean('feature_semantic_search_enabled')
        self.feature_authors_enabled = params.getboolean('feature_authors_enabled')
        self.feature_journals_enabled = params.getboolean('feature_journals_enabled')
        self.feature_numbers_enabled = params.getboolean('feature_numbers_enabled')
        self.feature_questions_enabled = params.getboolean('feature_questions_enabled')

        # Admin bootstrap credentials
        self.admin_email = self._read_secret('ADMIN_EMAIL', params.get('admin_email'))
        self.admin_password = self._read_secret('ADMIN_PASSWORD', params.get('admin_password'))


    # Prefer environment variables (or Docker secrets via *_FILE),
    # fallback to config.properties for backward compatibility
    @staticmethod
    def _read_secret(name: str, default: str = None):
            # Support *_FILE convention for Docker Swarm/K8s secrets
            file_var = os.getenv(f"{name}_FILE")
            if file_var and os.path.exists(file_var):
                try:
                    with open(file_var, 'r') as fh:
                        return fh.read().strip()
                except Exception:
                    # do not raise to avoid breaking startup; fall back to env/params
                    pass
            return os.getenv(name, default)
