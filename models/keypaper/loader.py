import configparser as configparser
import os

import psycopg2 as pg_driver
from Bio import Entrez


class Loader:
    def __init__(self,
                 host='localhost', port='5432', dbname='pubmed',
                 user='biolabs', password='pubtrends',
                 email='nikolay.kapralov@gmail.com'):

        # Find 'config.properties' file for parser
        config = configparser.ConfigParser()
        home_dir = os.path.expanduser('~')

        # Add fake section [params] for ConfigParser to accept the file
        with open(f'{home_dir}/.pubtrends/config.properties') as config_properties:
            config.read_string("[params]\n" + config_properties.read())

        if host is None:
            host = config['params']['url']

        if port is None:
            port = config['params']['port']

        if dbname is None:
            dbname = config['params']['database']

        if user is None:
            user = config['params']['username']

        if password is None:
            password = config['params']['password']

        Entrez.email = email
        connection_string = f"""
        dbname={dbname} user={user} password={password} host={host} port={port}
        """

        self.conn = pg_driver.connect(connection_string)
        self.cursor = self.conn.cursor()

    def set_logger(self, logger):
        self.logger = logger
