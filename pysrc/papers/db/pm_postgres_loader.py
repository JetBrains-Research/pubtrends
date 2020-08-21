import html
import logging
from collections import Iterable

import numpy as np
import pandas as pd

from pysrc.papers.db.loader import Loader
from pysrc.papers.db.postgres_connector import PostgresConnector
from pysrc.papers.utils import SORT_MOST_RELEVANT, SORT_MOST_CITED, SORT_MOST_RECENT, preprocess_doi, \
    preprocess_pubmed_search_title

logger = logging.getLogger(__name__)


class PubmedPostgresLoader(PostgresConnector, Loader):
    def __init__(self, pubtrends_config):
        super(PubmedPostgresLoader, self).__init__(pubtrends_config)

    @staticmethod
    def ids_to_vals(ids):
        return ','.join([f'({i})' for i in ids])

    def find(self, key, value, current=1, task=None):
        value = value.strip()
        self.progress.info(f"Searching for a publication with {key} '{value}'", current=current, task=task)

        if key == 'id':
            key = 'pmid'

            # We use integer PMIDs in neo4j, if value is not a valid integer -> no match
            try:
                value = int(value)
            except ValueError:
                raise Exception("PMID should be an integer")

        # Preprocess DOI
        if key == 'doi':
            value = preprocess_doi(value)

        # Use dedicated text index to search title.
        if key == 'title':
            value = preprocess_pubmed_search_title(value)
            query = f'''
                SELECT pmid
                FROM websearch_to_tsquery('english', '{value}') query, PMPublications P 
                WHERE tsv @@ query AND title = '{value}';
            '''
        else:
            query = f'''
                SELECT pmid
                FROM PMPublications P 
                WHERE {key} = {repr(value)};
            '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)

        self.progress.info(f'Found {len(df)} publications in the local database', current=current,
                           task=task)

        return list(df['pmid'].astype(str))

    def search(self, query, limit=None, sort=None, current=0, task=None):
        query_str = '\'' + query + '\''
        if not limit:
            limit_message = ''
            limit = self.config.max_number_of_articles
        else:
            limit_message = f'{limit} '

        self.progress.info(html.escape(f'Searching {limit_message}{sort.lower()} publications matching {query}'),
                           current=current, task=task)

        if sort == SORT_MOST_RELEVANT:
            query = f'''
                SELECT pmid
                FROM websearch_to_tsquery('english', {query_str}) query, PMPublications P 
                WHERE tsv @@ query
                ORDER BY ts_rank_cd(P.tsv, query) DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_CITED:
            query = f'''
                SELECT P.pmid as pmid
                FROM websearch_to_tsquery('english', {query_str}) query, PMPublications P
                    LEFT JOIN PMCitations C
                        ON C.pmid_in = P.pmid
                WHERE tsv @@ query
                GROUP BY pmid
                ORDER BY COUNT(*) DESC NULLS LAST
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_RECENT:
            query = f'''
                SELECT pmid
                FROM websearch_to_tsquery('english', {query_str}) query, PMPublications P
                WHERE tsv @@ query
                ORDER BY date DESC NULLS LAST
                LIMIT {limit};
                '''
        else:
            raise ValueError(f'Illegal sort method: {sort}')

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)

        self.progress.info(f'Found {len(df)} publications in the local database', current=current,
                           task=task)

        return list(df['pmid'].astype(str))

    def load_publications(self, ids, current=0, task=None):
        self.progress.info('Loading publication data', current=current, task=task)
        vals = self.ids_to_vals(ids)

        query = f'''
                SELECT P.pmid as id, title, abstract, date_part('year', date) as year, type, aux
                FROM PMPublications P
                WHERE P.pmid IN (VALUES {vals});
                '''
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(),
                              columns=['id', 'title', 'abstract', 'year', 'type', 'aux'],
                              dtype=object)
        if np.any(df[['id', 'title']].isna()):
            raise ValueError('Paper must have ID and title')
        df['id'] = df['id'].apply(str)
        logger.debug(f'Loaded {len(df)} papers')
        return Loader.process_publications_dataframe(df)

    def load_citation_stats(self, ids, current=0, task=None):
        self.progress.info('Loading citations statistics',
                           current=current, task=task)
        vals = self.ids_to_vals(ids)
        query = f'''
           SELECT C.pmid_in AS id, date_part('year', date) as year, COUNT(1) AS count
                FROM PMCitations C
                JOIN PMPublications P
                  ON C.pmid_out = P.pmid
                WHERE C.pmid_in IN (VALUES {vals}) 
                GROUP BY C.pmid_in, year;
            '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(),
                              columns=['id', 'year', 'count'], dtype=object)

        if np.any(df.isna()):
            raise ValueError('NaN values are not allowed in citation stats DataFrame')

        df['id'] = df['id'].apply(str)
        df['year'] = df['year'].apply(int)
        df['count'] = df['count'].apply(int)

        self.progress.info(f'Found {df.shape[0]} records of citations by year',
                           current=current, task=task)

        return df

    def load_citations(self, ids, current=0, task=None):
        self.progress.info('Loading citations data', current=current, task=task)
        vals = self.ids_to_vals(ids)
        query = f'''SELECT pmid_out as id_out, pmid_in as id_in
                    FROM PMCitations C
                    WHERE pmid_in IN (VALUES {vals}) AND pmid_out IN (VALUES {vals}); 
                    '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(),
                              columns=['id_out', 'id_in'], dtype=object)

        if np.any(df.isna()):
            raise ValueError('Citation must have id_out and id_in')
        df['id_in'] = df['id_in'].apply(str)
        df['id_out'] = df['id_out'].apply(str)

        self.progress.info(f'Found {len(df)} citations', current=current, task=task)
        return df

    def load_cocitations(self, ids, current=0, task=None):
        self.progress.info('Calculating co-citations for selected papers', current=current, task=task)
        vals = self.ids_to_vals(ids)
        query = f'''with X AS (SELECT pmid_out, pmid_in
                        FROM PMCitations
                        WHERE pmid_in IN (VALUES {vals})), 
                        
                        Y AS (SELECT pmid_out, ARRAY_AGG(pmid_in) as cited_list
                        FROM X
                        GROUP BY pmid_out
                        HAVING COUNT(*) >= 2)
                        
                        SELECT Y.pmid_out, date_part('year', P.date) as year, Y.cited_list
                        FROM Y
                        JOIN PMPublications P
                        ON pmid_out = P.pmid and pmid_out = P.pmid;
                    '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)

            data = []
            lines = 0
            for row in cursor:
                lines += 1
                citing, year, cited = row
                for i in range(len(cited)):
                    for j in range(i + 1, len(cited)):
                        data.append((str(citing), str(cited[i]), str(cited[j]), year))
            df = pd.DataFrame(data, columns=['citing', 'cited_1', 'cited_2', 'year'], dtype=object)

        if np.any(df[['citing', 'cited_1', 'cited_2']].isna()):
            raise ValueError('NaN values are not allowed in ids of co-citation DataFrame')
        df['year'] = df['year'].apply(lambda x: int(x) if x else np.nan)

        logger.debug(f'Loaded {lines} lines of citing info')
        self.progress.info(f'Found {len(df)} co-cited pairs of papers', current=current, task=task)

        return df

    def expand(self, ids, limit, current=1, task=None):
        max_to_expand = (limit - len(ids)) / 2
        vals = self.ids_to_vals(ids)
        if isinstance(ids, Iterable):
            self.progress.info('Expanding current topic', current=current, task=task)
            query = f'''
                SELECT C.pmid_out AS pmid_inner, ARRAY_AGG(C.pmid_in) AS pmids_outter 
                FROM PMCitations C
                WHERE C.pmid_out IN (VALUES {vals})
                GROUP BY C.pmid_out
                UNION
                SELECT C.pmid_in AS pmid_inner, ARRAY_AGG(C.pmid_out) AS pmids_outter
                FROM PMCitations C
                WHERE C.pmid_in IN (VALUES {vals})
                GROUP BY C.pmid_in
                LIMIT {max_to_expand};
                '''
        elif isinstance(ids, int):
            query = f'''
                SELECT C.pmid_out AS pmid_inner, ARRAY_AGG(C.pmid_in) AS pmids_outter
                FROM PMCitations C
                WHERE C.pmid_out = {ids}
                GROUP BY C.pmid_out
                UNION
                SELECT C.pmid_in AS pmid_inner, ARRAY_AGG(C.pmid_out) AS pmids_outter
                FROM PMCitations C
                WHERE C.pmid_in = {ids}
                GROUP BY C.pmid_in
                LIMIT {max_to_expand};
                '''
        else:
            raise TypeError('ids should be either int or Iterable')

        expanded = set()
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)

            for row in cursor.fetchall():
                inner, outer = row
                expanded.add(inner)
                expanded |= set(outer)

        self.progress.info(f'Found {len(expanded)} papers', current=current, task=task)
        return expanded

    def load_bibliographic_coupling(self, ids, current=1, task=None):
        self.progress.info('Processing bibliographic coupling for selected papers', current=current, task=task)
        vals = self.ids_to_vals(ids)
        query = f'''WITH X AS (SELECT pmid_out, pmid_in
                        FROM PMCitations
                        WHERE pmid_out IN (VALUES {vals})) 
                        
                        SELECT pmid_in, ARRAY_AGG(pmid_out) as citing_list
                        FROM X
                        GROUP BY pmid_in
                        HAVING COUNT(*) >= 2;
                    '''

        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query)

            data = []
            lines = 0
            for row in cursor:
                lines += 1
                _, citing = row
                for i in range(len(citing)):
                    for j in range(i + 1, len(citing)):
                        data.append((str(citing[i]), str(citing[j]), 1))
            df = pd.DataFrame(data, columns=['citing_1', 'citing_2', 'total'], dtype=object)
            df = df.groupby(['citing_1', 'citing_2']).sum().reset_index()

        logger.debug(f'Loaded {lines} lines of bibliographic coupling info')
        self.progress.info(f'Found {len(df)} bibliographic coupling pairs of papers',
                           current=current, task=task)

        return df
