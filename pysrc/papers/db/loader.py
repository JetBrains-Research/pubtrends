import html
import json
from abc import abstractmethod, ABCMeta

import numpy as np
import pandas as pd

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
        Expands list of ids doing one or two steps of breadth first search along citations graph.
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

    @staticmethod
    def process_cocitations_postgres(cursor):
        data = []
        lines = 0
        for row in cursor:
            lines += 1
            citing, year, cited_list = row
            cited_list.sort()
            for i in range(len(cited_list)):
                for j in range(i + 1, len(cited_list)):
                    data.append((str(citing), str(cited_list[i]), str(cited_list[j]), year))
        df = pd.DataFrame(data, columns=['citing', 'cited_1', 'cited_2', 'year'], dtype=object)
        df['year'] = df['year'].apply(lambda x: int(x) if x else np.nan)
        return df, lines

    @staticmethod
    def process_bibliographic_coupling_postgres(cursor):
        data = []
        lines = 0
        for row in cursor:
            lines += 1
            _, citing_list = row
            citing_list.sort()
            for i in range(len(citing_list)):
                for j in range(i + 1, len(citing_list)):
                    data.append((str(citing_list[i]), str(citing_list[j]), 1))
        df = pd.DataFrame(data, columns=['citing_1', 'citing_2', 'total'], dtype=object)
        df = df.groupby(['citing_1', 'citing_2']).sum().reset_index()
        return df, lines
