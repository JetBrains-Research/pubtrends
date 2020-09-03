import configparser
import os

import numpy as np


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

        self.neo4j_host = params['neo4j_host' if not test else 'test_neo4j_host']
        self.neo4j_port = params['neo4j_port' if not test else 'test_neo4j_port']
        self.neo4j_username = params['neo4j_username' if not test else 'test_neo4j_username']
        self.neo4j_password = params['neo4j_password' if not test else 'test_neo4j_password']

        self.postgres_host = params['postgres_host' if not test else 'test_postgres_host']
        self.postgres_port = params['postgres_port' if not test else 'test_postgres_port']
        self.postgres_username = params['postgres_username' if not test else 'test_postgres_username']
        self.postgres_password = params['postgres_password' if not test else 'test_postgres_password']
        self.postgres_database = params['postgres_database' if not test else 'test_postgres_database']

        self.experimental = params.getboolean('experimental')

        self.pm_enabled = params.getboolean('pm_enabled')
        self.pm_search_example_terms = [terms.strip() for terms in params['pm_search_example_terms'].split(';')]

        self.ss_enabled = params.getboolean('ss_enabled')
        self.ss_search_example_terms = [terms.strip() for terms in params['ss_search_example_terms'].split(';')]

        self.min_search_words = params.getint('min_search_words') if not test else 0
        self.max_number_of_citations = params.getint('max_number_of_citations')
        self.max_number_of_cocitations = params.getint('max_number_of_cocitations')
        self.max_number_of_bibliographic_coupling = params.getint('max_number_of_bibliographic_coupling')
        self.max_number_to_expand = params.getint('max_number_to_expand')

        self.show_max_articles_options = [opt.strip() for opt in params['show_max_articles_options'].split(',')]
        self.show_max_articles_default_value = params['show_max_articles_default_value'].strip()
        self.max_number_of_articles = np.max(self.show_max_articles_options)

        # Max allowed pending tasks
        self.celery_max_pending_tasks = params.getint('celery_max_pending_tasks')
        # Seconds, pending task will be revoked after no polling activity
        self.celery_pending_tasks_timeout = params.getint('celery_pending_tasks_timeout')

        # TODO Admin password - should be a better way
        self.admin_password = params['admin_password']
