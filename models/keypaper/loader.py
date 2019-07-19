import configparser as configparser
import os

import psycopg2 as pg_driver


class DBConfig:
    def __init__(self, test=True):
        # Find 'config.properties' file
        config = configparser.ConfigParser()
        home_dir = os.path.expanduser('~')

        # Add fake section [params] for ConfigParser to accept the file
        with open(f'{home_dir}/.pubtrends/config.properties') as config_properties:
            config.read_string("[params]\n" + config_properties.read())

        self.host = config['params']['url' if not test else 'test_url']
        self.port = config['params']['port' if not test else 'test_port']
        self.dbname = config['params']['database' if not test else 'test_database']
        self.user = config['params']['username' if not test else 'test_username']
        self.password = config['params']['password' if not test else 'test_password']


class Loader:
    def __init__(self, test=True):
        db_config = DBConfig(test)
        connection_string = f"""
dbname={db_config.dbname} user={db_config.user} password={db_config.password} host={db_config.host} port={db_config.port}
        """
        self.conn = pg_driver.connect(connection_string)
        self.cursor = self.conn.cursor()

    def set_logger(self, logger):
        self.logger = logger
