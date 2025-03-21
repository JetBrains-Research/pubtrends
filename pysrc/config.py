import configparser
import os

#####################
## Analysis config ##
#####################

# Features are originally taken from paper:
# 1) Which type of citation analysis generates the most accurate taxonomy of
#   scientific and technical knowledge? (https://arxiv.org/pdf/1511.05078.pdf)
#   ...bibliographic coupling (BC) was the most accurate,  followed by co-citation (CC).
#   Direct citation (DC) was a distant third among the three...

SIMILARITY_COCITATION = 10  # x number of co-citations
SIMILARITY_BIBLIOGRAPHIC_COUPLING = 3  # x number of references
SIMILARITY_CITATION = 1  # x 0-1 citation

# Reduce number of edges in papers graph for node2vec embeddings computations
EMBEDDINGS_GRAPH_EDGES = 100

# Reduce number of edges in papers graph for visualization
GRAPH_BIBLIOGRAPHIC_EDGES = 20

# Add artificial text similarity nodes to sparse graph for visualisations
GRAPH_TEXT_SIMILARITY_EDGES = 5

# Minimal number of common references, used to reduce papers graph edges count
# Value > 1 is especially useful while analysing single paper,
# removes meaningless connections by construction
SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN = 5

# Minimal number of common references, used to reduce papers graph edges count
SIMILARITY_COCITATION_MIN = 2

# Papers embeddings is a weighted average of graph and text embeddings
EMBEDDINGS_VECTOR_LENGTH = 200
GRAPH_EMBEDDINGS_FACTOR = 1
TEXT_EMBEDDINGS_FACTOR = 1

PCA_COMPONENTS = 15

# Global vectorization max vocabulary size
VECTOR_WORDS = 10_000

# Terms with lower frequency will be ignored, remove rare words
VECTOR_MIN_DF = 0.001

# Terms with higher frequency will be ignored, remove abundant words
VECTOR_MAX_DF = 0.8

# Control citations count
EXPAND_CITATIONS_Q_LOW = 10
EXPAND_CITATIONS_Q_HIGH = 90
EXPAND_CITATIONS_SIGMA = 3

# Take up to fraction of top similarity
EXPAND_MESH_SIMILARITY = 0.5

# Impact of single paper when analyzing citations and mesh terms when analysing paper
EXPAND_SINGLE_PAPER_IMPACT = 50

#####################
## Node2vec config ##
#####################

NODE2VEC_P = 5.0
NODE2VEC_Q = 2.0

# Increasing number of walks increases node2vec representation accuracy
NODE2VEC_WALKS_PER_NODE = 100
NODE2VEC_WALK_LENGTH = 50
NODE2VEC_WORD2VEC_WINDOW = 5
NODE2VEC_WORD2VEC_EPOCHS = 5

#####################
## Word2vec config ##
#####################

WORD2VEC_WINDOW = 5
WORD2VEC_EPOCHS = 5

#################
## Plot config ##
#################

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

        self.show_topics_options = [int(opt.strip()) for opt in params['show_topics_options'].split(',')]
        self.show_topics_default_value = int(params['show_topics_default_value'].strip())

        self.max_graph_size = params.getint('max_graph_size')

        self.top_cited_papers = params.getint('top_cited_papers')
        self.topic_description_words = params.getint('topic_description_words')

        self.popular_journals = params.getint('popular_journals')
        self.popular_authors = params.getint('popular_authors')

        self.paper_expands_steps = params.getint('paper_expands_steps')
        self.paper_expand_limit = params.getint('paper_expand_limit')

        # TODO Admin password - should be a better way
        self.admin_email = params['admin_email']
        self.admin_password = params['admin_password']

        # Additional modules configuration
        self.feature_authors_enabled = params.getboolean('feature_authors_enabled')
        self.feature_journals_enabled = params.getboolean('feature_journals_enabled')
        self.feature_numbers_enabled = params.getboolean('feature_numbers_enabled')
        self.feature_review_enabled = params.getboolean('feature_review_enabled')
