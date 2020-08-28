import html
import json
from abc import abstractmethod, ABCMeta

import numpy as np

from pysrc.papers.utils import extract_authors


class Loader(metaclass=ABCMeta):

    def __init__(self, ):
        self.progress = None

    def set_progress(self, pl):
        self.progress = pl

    @abstractmethod
    def find(self, key, value, current=1, task=None):
        """
        Searches single or multiple paper(s) for give search key, value.
        :return: list of ids, i.e. list(String).
        """

    @abstractmethod
    def search(self, query, limit=None, sort=None, current=1, task=None):
        """
        Searches publications by given query.
        :return: list of ids, i.e. list(String).
        """

    @abstractmethod
    def load_publications(self, ids, current=1, task=None):
        """
        Loads publications for given ids.
        :return: dataframe[id(String), title, abstract, year, type, aux]
        """

    @abstractmethod
    def load_citation_stats(self, ids, current=1, task=None):
        """
        Loads all the citations stats for each of given ids.
        :return: dataframe[id(String), year, count]
        """

    @abstractmethod
    def load_citations(self, ids, current=1, task=None):
        """
        Loading INNER citations graph, where all the nodes are inside query of interest.
        :return: dataframe[id_out(String), id_in(String)]
        """

    @abstractmethod
    def load_cocitations(self, ids, current=1, task=None):
        """
        Loading co-citations graph.
        :return: dataframe[citing(String), cited_1(String), cited_2(String), year]
        """

    @abstractmethod
    def load_bibliographic_coupling(self, ids, current=1, task=None):
        """
        Loading bibliographic coupling graph.
        :return: dataframe[citing_1(String), citing_2(String), total]
        """

    @abstractmethod
    def expand(self, ids, limit, current=1, task=None):
        """
        Expands list of ids after one or BFS step along citations graph sorted by citations count.
        :return: list of ids, i.e. list(String).
        """

    @staticmethod
    def process_publications_dataframe(pub_df):
        # Switch to string ids
        pub_df['id'] = pub_df['id'].apply(str)
        # Semantic Scholar stores aux in jsonb format, no json parsing required
        pub_df['aux'] = pub_df['aux'].apply(
            lambda aux: json.loads(aux) if type(aux) is str else aux
        )
        pub_df = pub_df.fillna(value={'abstract': ''})
        pub_df['year'] = pub_df['year'].apply(
            lambda year: int(year) if year and np.isfinite(year) else np.nan
        )
        pub_df['authors'] = pub_df['aux'].apply(lambda aux: extract_authors(aux['authors']))
        pub_df['journal'] = pub_df['aux'].apply(lambda aux: html.unescape(aux['journal']['name']))
        pub_df['title'] = pub_df['title'].apply(lambda title: html.unescape(title))

        # Semantic Scholar specific hack
        if 'crc32id' in pub_df:
            pub_df['crc32id'] = pub_df['crc32id'].apply(int)
        return pub_df
