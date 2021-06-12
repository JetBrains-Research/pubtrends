import logging

import pandas as pd

from pysrc.papers.db.ss_postgres_loader import SemanticScholarPostgresLoader
from pysrc.papers.utils import SORT_MOST_CITED, SORT_MOST_RECENT

logger = logging.getLogger(__name__)


class SSArxivLoader(SemanticScholarPostgresLoader):
    def __init__(self, config):
        super(SSArxivLoader, self).__init__(config)

    def search(self, query, limit=None, sort=None, noreviews=True):
        raise Exception('Use search_arxiv')

    def search_arxiv(self, limit, sort='random'):
        # TODO: implement arxiv filtration here!!!
        if sort == SORT_MOST_CITED:
            query = f'''
                SELECT P.ssid
                FROM SSPublications P 
                LEFT JOIN matview_sscitations C
                ON C.ssid = P.ssid AND C.crc32id = P.crc32id
                WHERE P.pmid IS NOT NULL
                GROUP BY P.ssid, P.crc32id
                ORDER BY COUNT(*) DESC NULLS LAST
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_RECENT:
            query = f'''
                SELECT ssid
                SSPublications P
                WHERE P.pmid IS NOT NULL
                ORDER BY year DESC NULLS LAST
                LIMIT {limit};
                '''
        else:
            raise ValueError(f'Illegal sort method: {sort}')

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            pub_df = pd.DataFrame(cursor.fetchall(), columns=['id'])

        # Duplicate rows may occur if crawler was stopped while parsing Semantic Scholar archive
        pub_df.drop_duplicates(subset='id', inplace=True)

        return list(pub_df['id'].values)
