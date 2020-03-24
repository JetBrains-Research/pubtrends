import configparser
import os


class PubtrendsConfig:
    """
    Main service configuration loaded from ~/.pubtrends/config.properties
    """

    def __init__(self, config_path='~/.pubtrends/config.properties', test=True):
        config_parser = configparser.ConfigParser()

        # Add fake section [params] for ConfigParser to accept the file
        with open(os.path.expanduser(config_path)) as config_properties:
            config_parser.read_string("[params]\n" + config_properties.read())
        params = config_parser['params']

        self.neo4jhost = params['neo4jhost' if not test else 'test_neo4jhost']
        self.neo4jport = params['neo4jport' if not test else 'test_neo4jport']
        self.neo4juser = params['neo4jusername' if not test else 'test_neo4jusername']
        self.neo4jpassword = params['neo4jpassword' if not test else 'test_neo4jpassword']

        self.experimental = params.getboolean('experimental')
        self.development = params.getboolean('development')

        self.pm_enabled = params.getboolean('pm_enabled')
        self.pm_search_example_terms = [terms.strip() for terms in params['pm_search_example_terms'].split(';')]

        self.ss_enabled = params.getboolean('ss_enabled')
        self.ss_search_example_terms = [terms.strip() for terms in params['ss_search_example_terms'].split(';')]

        self.min_search_words = params.getint('min_search_words') if not test else 0
        self.max_number_of_articles = params.getint('max_number_of_articles')
        self.max_number_of_citations = params.getint('max_number_of_citations')
        self.max_number_of_cocitations = params.getint('max_number_of_cocitations')
        self.max_number_of_bibliographic_coupling = params.getint('max_number_of_bibliographic_coupling')
        self.max_number_to_expand = params.getint('max_number_to_expand')

        self.show_max_articles_options = [opt.strip() for opt in params['show_max_articles_options'].split(',')]
        self.show_max_articles_default_value = params['show_max_articles_default_value'].strip()

        # Max allowed pending tasks
        self.celery_max_pending_tasks = params.getint('celery_max_pending_tasks')
        # Seconds, pending task will be revoked after no polling activity
        self.celery_pending_tasks_timeout = params.getint('celery_pending_tasks_timeout')
        # Max completed tasks to store
        self.celery_max_completed_tasks = params.getint('celery_max_completed_tasks')
