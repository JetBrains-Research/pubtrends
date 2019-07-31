import configparser as configparser
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

        self.version = config_parser['params']['version']

        self.host = config_parser['params']['url' if not test else 'test_url']
        self.port = config_parser['params']['port' if not test else 'test_port']
        self.dbname = config_parser['params']['database' if not test else 'test_database']
        self.user = config_parser['params']['username' if not test else 'test_username']
        self.password = config_parser['params']['password' if not test else 'test_password']

        self.pm_entrez_email = config_parser['params']['pm_entrez_email']
