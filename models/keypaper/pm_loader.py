import html
import re

import numpy as np
import pandas as pd
from Bio import Entrez

from models.keypaper.utils import extract_authors
from .loader import Loader


class PubmedLoader(Loader):
    def __init__(self, pubtrends_config):
        super(PubmedLoader, self).__init__(pubtrends_config)
        Entrez.email = pubtrends_config.pm_entrez_email

    def search(self, terms, current=0, task=None):
        self.logger.debug(f'TODO: handle queries which return more than {self.max_number_of_articles} items',
                          current=current, task=task)
        handle = Entrez.esearch(db='pubmed', retmax=str(self.max_number_of_articles),
                                retmode='xml', term=terms)
        self.ids = Entrez.read(handle)['IdList']
        self.logger.info(f'Found {len(self.ids)} publications matching {terms}', current=current, task=task)
        self.values = ', '.join(['({})'.format(i) for i in sorted(self.ids)])
        return self.ids

    def load_publications(self, current=0, task=None):
        self.logger.info('Loading publication data', current=current, task=task)

        query = re.sub(Loader.VALUES_REGEX, self.values, '''
        DROP TABLE IF EXISTS TEMP_PMIDS;
        WITH vals(pmid) AS (VALUES $VALUES$)
        SELECT pmid INTO temporary table TEMP_PMIDS FROM vals;
        DROP INDEX IF EXISTS temp_pmids_unique_index;
        CREATE UNIQUE INDEX temp_pmids_unique_index ON TEMP_PMIDS USING btree (pmid);

        SELECT CAST(P.pmid AS TEXT), P.title, P.aux, P.abstract, date_part('year', P.date) AS year
        FROM PMPublications P
        JOIN TEMP_PMIDS AS T ON (P.pmid = T.pmid);
        ''')
        self.logger.debug('Creating pmids table for request with index.', current=current, task=task)

        with self.conn.cursor() as cursor:
            cursor.execute(query)
            pub_df = pd.DataFrame(cursor.fetchall(),
                                  columns=['id', 'title', 'aux', 'abstract', 'year'],
                                  dtype=object)

        if np.any(pub_df[['id', 'title']].isna()):
            raise ValueError('Paper must have PMID and title')
        pub_df = pub_df.fillna(value={'abstract': ''})

        pub_df['year'] = pub_df['year'].apply(lambda year: int(year) if year else np.nan)
        pub_df['authors'] = pub_df['aux'].apply(lambda aux: extract_authors(aux['authors']))
        pub_df['journal'] = pub_df['aux'].apply(lambda aux: html.unescape(aux['journal']['name']))

        self.logger.debug(f'Found {len(pub_df)} publications in the local database\n', current=current, task=task)
        return pub_df

    def load_citation_stats(self, current=0, task=None):
        self.logger.info('Loading citations statistics: searching for correct citations over 168 million of citations',
                         current=current, task=task)

        query = re.sub(Loader.VALUES_REGEX, self.values, '''
        SELECT CAST(C.pmid_in AS TEXT) AS pmid, date_part('year', P.date) AS year, COUNT(1) AS count
        FROM PMCitations C
        JOIN (VALUES $VALUES$) AS CT(pmid) ON (C.pmid_in = CT.pmid)
        JOIN PMPublications P
        ON C.pmid_out = P.pmid
        WHERE date IS NOT NULL
        GROUP BY C.pmid_in, date_part('year', P.date);
        ''')

        with self.conn.cursor() as cursor:
            cursor.execute(query)
            self.logger.debug('Done loading citation stats', current=current, task=task)
            cit_stats_df_from_query = pd.DataFrame(cursor.fetchall(),
                                                   columns=['id', 'year', 'count'])

        if np.any(cit_stats_df_from_query.isna()):
            raise ValueError('NaN values are not allowed in citation stats DataFrame')

        cit_stats_df_from_query['year'] = cit_stats_df_from_query['year'].apply(int)
        cit_stats_df_from_query['count'] = cit_stats_df_from_query['count'].apply(int)

        return cit_stats_df_from_query

    def load_citations(self, current=0, task=None):
        self.logger.info('Started loading citations', current=current, task=task)

        query = re.sub(Loader.VALUES_REGEX, self.values, '''
        SELECT CAST(C.pmid_out AS TEXT), CAST(C.pmid_in AS TEXT)
        FROM PMCitations C
        JOIN (VALUES $VALUES$) AS CT(pmid) ON (C.pmid_in = CT.pmid)
        JOIN (VALUES $VALUES$) AS CT2(pmid) ON (C.pmid_out = CT2.pmid);
        ''')

        with self.conn.cursor() as cursor:
            cursor.execute(query)

            cit_df = pd.DataFrame(cursor.fetchall(), columns=['id_out', 'id_in'])

        if np.any(cit_df.isna()):
            raise ValueError('Citation must have id_out and id_in')

        self.logger.debug(f'Found {len(cit_df)} citations', current=current, task=task)

        return cit_df

    def load_cocitations(self, current=0, task=None):
        self.logger.info('Calculating co-citations for selected papers', current=current, task=task)

        # Use unfolding to pairs on the client side instead of DataBase
        query = '''
        with Z as (select pmid_out, CAST(pmid_in AS TEXT)
            from PMCitations
            -- Hack to make Postgres use index!
            where pmid_in
            between (select min(pmid) from TEMP_PMIDS) and (select max(pmid) from TEMP_PMIDS)
            and pmid_in in (select pmid from TEMP_PMIDS)),
        X as (select pmid_out, array_agg(pmid_in) as cited_list
            from Z
            group by pmid_out
            having count(*) >= 2)
        select CAST(X.pmid_out AS TEXT), date_part('year', P.date) AS year, X.cited_list from
            X join PMPublications P
            on pmid_out = P.pmid;
        '''

        with self.conn.cursor() as cursor:
            cursor.execute(query)

            cocit_data = []
            lines = 0
            for row in cursor:
                lines += 1
                citing, year, cited = row
                for i in range(len(cited)):
                    for j in range(i + 1, len(cited)):
                        cocit_data.append((citing, cited[i], cited[j], year))

        cocit_df = pd.DataFrame(cocit_data, columns=['citing', 'cited_1', 'cited_2', 'year'], dtype=object)

        if np.any(cocit_df.isna()):
            raise ValueError('NaN values are not allowed in co-citation DataFrame')
        cocit_df['year'] = cocit_df['year'].apply(int)

        self.logger.debug(f'Loaded {lines} lines of citing info', current=current, task=task)
        self.logger.debug(f'Found {len(cocit_df)} co-cited pairs of papers', current=current, task=task)

        return cocit_df
