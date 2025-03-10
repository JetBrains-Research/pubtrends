import json
import numpy as np
from abc import abstractmethod, ABCMeta

from pysrc.papers.utils import extract_authors, reorder_publications


class Loader(metaclass=ABCMeta):
    # Max years of publications SPAN to prevent papers from 1860 to spoil all visualizations
    MAX_YEARS_SPAN = 50

    @abstractmethod
    def find(self, key, value):
        """
        Searches single or multiple paper(s) for give search key, value.
        :return: list of ids, i.e. list(String).
        """

    @abstractmethod
    def search(self, query, limit=None, sort=None, noreviews=True):
        """
        Searches publications by given query, ignoring reviews
        :return: list of ids, i.e. list(String).
        """

    @abstractmethod
    def load_publications(self, ids):
        """
        Loads publications for given ids.
        :return: dataframe[id(String), title, abstract, year, type, keywords, mesh, doi, aux]
        """

    @abstractmethod
    def load_citations_by_year(self, ids):
        """
        Loads all the citations stats for each of given ids.
        :return: dataframe[id(String), year, count]
        """

    @abstractmethod
    def load_references(self, pid, limit):
        """
        Returns paper references, limited by number.
        :return: list[String]
        """

    @abstractmethod
    def load_citations_counts(self, ids):
        """
        Returns total citations counts for each paper id.
        :return: list[Int]
        """

    @abstractmethod
    def load_citations(self, ids):
        """
        Loading citations graph, between ids.
        :return: dataframe[id_out(String), id_in(String)]
        """

    @abstractmethod
    def load_cocitations(self, ids):
        """
        Loading co-citations graph.
        :return: dataframe[citing(String), cited_1(String), cited_2(String), year]
        """

    @abstractmethod
    def load_bibliographic_coupling(self, ids):
        """
        Loading bibliographic coupling graph.
        :return: dataframe[citing_1(String), citing_2(String), total]
        """

    @abstractmethod
    def expand(self, ids, limit):
        """
        Expands list of ids after one or BFS step along citations graph.
        :return: dataframe[id(String), total]
        """

    @staticmethod
    def process_publications_dataframe(ids, pub_df):
        # Switch to string ids
        pub_df['id'] = pub_df['id'].apply(str)
        # Semantic Scholar stores aux in jsonb format, no json parsing required
        pub_df['aux'] = pub_df['aux'].apply(
            lambda aux: json.loads(aux) if type(aux) is str else aux
        )
        pub_df = pub_df.fillna(value={'abstract': ''})
        pub_df['year'] = pub_df['year'].apply(int)
        # Keep max year span
        pub_df['year'] = pub_df['year'].clip(lower=pub_df['year'].max() - Loader.MAX_YEARS_SPAN)
        pub_df['authors'] = pub_df['aux'].apply(lambda aux: extract_authors(aux['authors']))
        pub_df['journal'] = pub_df['aux'].apply(lambda aux: aux['journal']['name'])
        pub_df['title'] = pub_df['title'].apply(lambda title: title)

        # Semantic Scholar specific hack
        if 'crc32id' in pub_df:
            pub_df['crc32id'] = pub_df['crc32id'].apply(int)

        # Reorder dataframe
        return reorder_publications(ids, pub_df)

