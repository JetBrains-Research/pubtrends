import html

import numpy as np
import pandas as pd

from .ss_loader import SemanticScholarLoader
from .utils import extract_authors


class ArxivLoader(SemanticScholarLoader):
    def __init__(self, pubtrends_config):
        super(ArxivLoader, self).__init__(pubtrends_config)

    def search(self, *terms, current=0, task=None):
        query = f'''
        SELECT DISTINCT ON(ssid) ssid, crc32id, title, abstract, year, aux FROM SSPublications P
        WHERE source = 'Arxiv' limit {self.max_number_of_articles};
        '''

        with self.conn.cursor() as cursor:
            cursor.execute(query)
            self.pub_df = pd.DataFrame(cursor.fetchall(),
                                       columns=['id', 'crc32id', 'title', 'abstract', 'year', 'aux'],
                                       dtype=object)

        if np.any(self.pub_df[['id', 'crc32id', 'title']].isna()):
            raise ValueError('Paper must have ID and title')
        self.pub_df = self.pub_df.fillna(value={'abstract': ''})

        self.pub_df['year'] = self.pub_df['year'].apply(lambda year: int(year) if year else np.nan)
        self.pub_df['authors'] = self.pub_df['aux'].apply(lambda aux: extract_authors(aux['authors']))
        self.pub_df['journal'] = self.pub_df['aux'].apply(lambda aux: html.unescape(aux['journal']['name']))
        self.pub_df['title'] = self.pub_df['title'].apply(lambda title: html.unescape(title))

        self.logger.info(f'Found {len(self.pub_df)} publications in the local database', current=current,
                         task=task)

        self.ids = self.pub_df['id']
        crc32ids = self.pub_df['crc32id']
        self.values = ', '.join([f'({i}, \'{j}\')' for (i, j) in zip(crc32ids, self.ids)])
        return self.ids
