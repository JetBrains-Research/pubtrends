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

        self.version = params['version']

        self.host = params['url' if not test else 'test_url']
        self.port = params['port' if not test else 'test_port']
        self.dbname = params['database' if not test else 'test_database']
        self.user = params['username' if not test else 'test_username']
        self.password = params['password' if not test else 'test_password']

        self.pm_entrez_email = params['pm_entrez_email']
        self.run_experimental = params.getboolean('run_experimental')

        self.max_number_of_articles = params.getint('max_number_of_articles')
        self.max_number_of_citations = params.getint('max_number_of_citations')
        self.max_number_of_cocitations = params.getint('max_number_of_cocitations')

        self.show_max_articles_options = [opt.strip() for opt in
                                          params['show_max_articles_options'].split(',')]
        self.show_max_articles_default_value = params['show_max_articles_default_value'].strip()
