import html
import json
from abc import abstractmethod, ABCMeta

import numpy as np

from models.keypaper.connector import Connector
from models.keypaper.utils import extract_authors


class Loader(Connector, metaclass=ABCMeta):

    def __init__(self, pubtrends_config, connect=True):
        super(Loader, self).__init__(pubtrends_config, connect)
        self.pubtrends_config = pubtrends_config
        self.max_number_of_articles = pubtrends_config.max_number_of_articles
        self.max_number_of_citations = pubtrends_config.max_number_of_citations
        self.max_number_of_cocitations = pubtrends_config.max_number_of_cocitations
        self.progress = None

    def set_progress_logger(self, pl):
        self.progress = pl

    @abstractmethod
    def find(self, key, value, current=0, task=None):
        """
        Searches single or multiple paper(s) for give search key, value.
        :return: list of ids, i.e. list[String].
        """
        pass

    @abstractmethod
    def search(self, query, limit=None, sort=None, current=0, task=None):
        """
        Searches publications by given query.
        :return: list of ids, i.e. list[String].
        """
        pass

    @abstractmethod
    def load_publications(self, ids, current=0, task=None):
        """
        Loads publications for given ids.
        :return: dataframe[id, title, abstract, year, type, aux]
        """

    @abstractmethod
    def load_citation_stats(self, ids, current=0, task=None):
        """
        Loads all the citations stats for each of given ids.
        :return: dataframe[id, year, count]
        """
        pass

    @abstractmethod
    def load_citations(self, ids, current=0, task=None):
        """
        Loading INNER citations graph, where all the nodes are inside query of interest.
        :return: dataframe[id_out, id_in]
        """
        pass

    @abstractmethod
    def load_cocitations(self, ids, current=0, task=None):
        """
        Loading co-citations graph.
        :return: dataframe[citing, cited_1, cited_2, year]
        """

    @abstractmethod
    def expand(self, ids, current=0, task=None):
        """
        Expands list of ids doing one or two steps of breadth first search along citations graph.
        :return: list of ids, i.e. list[String].
        """

    @staticmethod
    def process_publications_dataframe(publications_df):
        # Semantic Scholar stores aux in jsonb format, no json parsing required
        publications_df['aux'] = publications_df['aux'].apply(
            lambda aux: json.loads(aux) if type(aux) is str else aux
        )
        publications_df = publications_df.fillna(value={'abstract': ''})
        publications_df['year'] = publications_df['year'].apply(
            lambda year: int(year) if year and np.isfinite(year) else np.nan
        )
        publications_df['authors'] = publications_df['aux'].apply(lambda aux: extract_authors(aux['authors']))
        publications_df['journal'] = publications_df['aux'].apply(lambda aux: html.unescape(aux['journal']['name']))
        publications_df['title'] = publications_df['title'].apply(lambda title: html.unescape(title))

        # Semantic Scholar specific hack
        if 'crc32id' in publications_df:
            publications_df['crc32id'] = publications_df['crc32id'].apply(int)
        return publications_df
