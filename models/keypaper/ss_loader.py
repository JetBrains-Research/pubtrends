import html
import re
from collections import Iterable

import numpy as np
import pandas as pd

from models.keypaper.utils import SORT_MOST_CITED, SORT_MOST_RECENT, SORT_MOST_RELEVANT
from .loader import Loader
from .utils import crc32, preprocess_search_query


class SemanticScholarLoader(Loader):

    def __init__(self, pubtrends_config):
        super(SemanticScholarLoader, self).__init__(pubtrends_config)

    def find(self, key, value, current=1, task=None):
        self.progress.info(f"Searching for a publication with {key} '{value}'", current=current, task=task)
        if key == 'id':
            key = 'ssid'
        # Use dedicated text index to search title.
        if key == 'title':
            query = f'''
                CALL db.index.fulltext.queryNodes("ssTitlesAndAbstracts", '"{re.sub('"', '', value.strip())}"')
                YIELD node
                MATCH (p:SSPublication)
                WHERE p.ssid = node.ssid AND p.title = '{value}'
                RETURN p.ssid AS ssid;
            '''
        else:
            query = f'''
                MATCH (p:SSPublication)
                WHERE p.{key} = {repr(value)}
                RETURN p.ssid AS ssid;
            '''

        with self.neo4jdriver.session() as session:
            return [str(r['ssid']) for r in session.run(query)]

    def search(self, query, limit=None, sort=None, current=1, task=None):
        query_str = preprocess_search_query(query, self.pubtrends_config.min_search_words)

        if not limit:
            limit_message = ''
            limit = self.max_number_of_articles
        else:
            limit_message = f'{limit} '

        self.progress.info(html.escape(f'Searching {limit_message}{sort.lower()} publications matching <{query}>'),
                           current=current, task=task)

        if sort == SORT_MOST_RELEVANT:
            query = f'''
                CALL db.index.fulltext.queryNodes("ssTitlesAndAbstracts", {query_str})
                YIELD node, score
                RETURN node.ssid as ssid
                ORDER BY score DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_CITED:
            query = f'''
                CALL db.index.fulltext.queryNodes("ssTitlesAndAbstracts", {query_str}) YIELD node
                MATCH ()-[r:SSReferenced]->(in:SSPublication)
                WHERE in.crc32id = node.crc32id AND in.ssid = node.ssid
                WITH node, COUNT(r) AS cnt
                RETURN node.ssid as ssid
                ORDER BY cnt DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_RECENT:
            query = f'''
                CALL db.index.fulltext.queryNodes("ssTitlesAndAbstracts", {query_str}) YIELD node
                RETURN node.ssid as ssid
                ORDER BY node.date DESC
                LIMIT {limit};
                '''
        else:
            raise ValueError(f'Illegal sort method: {sort}')

        with self.neo4jdriver.session() as session:
            ids = [str(r['ssid']) for r in session.run(query)]

        self.progress.info(f'Found {len(ids)} publications in the local database', current=current,
                           task=task)
        return ids

    def load_publications(self, ids, current=1, task=None):
        self.progress.info('Loading publication data', current=current, task=task)

        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join([f'"{id}"' for id in ids])}] AS ssids,
                 [{','.join([str(crc32(id)) for id in ids])}] AS crc32ids
            MATCH (p:SSPublication)
            WHERE p.crc32id in crc32ids AND p.ssid IN ssids
            RETURN p.ssid as id, p.crc32id as crc32id, p.pmid as pmid, p.title as title, p.abstract as abstract,
                p.date.year as year, p.type as type, p.aux as aux
            ORDER BY id
        '''

        with self.neo4jdriver.session() as session:
            pub_df = pd.DataFrame(session.run(query).data())
            if len(pub_df) == 0:
                self.progress.debug(f'Failed to load publications.', current=current, task=task)

        if np.any(pub_df[['id', 'title']].isna()):
            raise ValueError('Paper must have PMID and title')

        pub_df = Loader.process_publications_dataframe(pub_df)

        self.progress.debug(f'Found {len(pub_df)} publications in the local database', current=current, task=task)
        # Hack for missing type in SS, see https://github.com/JetBrains-Research/pubtrends/issues/200
        pub_df['type'] = 'Article'
        return pub_df

    def load_citation_stats(self, ids, current=1, task=None):
        self.progress.info('Loading citations statistics', current=current, task=task)

        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join([f'"{id}"' for id in ids])}] AS ssids,
                 [{','.join([str(crc32(id)) for id in ids])}] AS crc32ids
            MATCH (out:SSPublication)-[:SSReferenced]->(in:SSPublication)
            WHERE in.crc32id in crc32ids AND in.ssid IN ssids AND out.date.year >= in.date.year
            RETURN in.ssid AS id, out.date.year AS year, COUNT(*) AS count;
        '''

        with self.neo4jdriver.session() as session:
            cit_stats_df = pd.DataFrame(session.run(query).data())
            if len(cit_stats_df) == 0:
                self.progress.debug(f'Failed to load citations statistics.', current=current, task=task)

        self.progress.debug('Done loading citation stats', current=current, task=task)

        if np.any(cit_stats_df.isna()):
            raise ValueError('NaN values are not allowed in citation stats DataFrame')

        cit_stats_df['count'] = cit_stats_df['count'].apply(int)
        cit_stats_df['id'] = cit_stats_df['id'].apply(str)
        cit_stats_df['year'] = cit_stats_df['year'].apply(int)

        self.progress.info(f'Found {cit_stats_df.shape[0]} records of citations by year',
                           current=current, task=task)

        return cit_stats_df

    def load_citations(self, ids, current=1, task=None):
        """ Loading INNER citations graph, where all the nodes are inside query of interest """
        self.progress.info('Started loading citations', current=current, task=task)

        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join([f'"{id}"' for id in ids])}] AS ssids,
                 [{','.join([str(crc32(id)) for id in ids])}] AS crc32ids
            MATCH (out:SSPublication)-[:SSReferenced]->(in:SSPublication)
            WHERE in.crc32id in crc32ids AND in.ssid IN ssids AND
                  out.crc32id in crc32ids AND out.ssid IN ssids
            RETURN out.ssid AS id_out, in.ssid AS id_in
            ORDER BY id_out, id_in;
        '''

        with self.neo4jdriver.session() as session:
            cit_df = pd.DataFrame(session.run(query).data())
            if len(cit_df) == 0:
                self.progress.debug(f'Failed to load citations.', current=current, task=task)

        if np.any(cit_df.isna()):
            raise ValueError('Citation must have id_out and id_in')

        self.progress.info(f'Found {len(cit_df)} citations', current=current, task=task)

        cit_df['id_in'] = cit_df['id_in'].apply(str)
        cit_df['id_out'] = cit_df['id_out'].apply(str)
        return cit_df

    def load_cocitations(self, ids, current=1, task=None):
        self.progress.info('Calculating co-citations for papers', current=current, task=task)

        # Use unfolding to pairs on the client side instead of DataBase
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join([f'"{id}"' for id in ids])}] AS ssids,
                 [{','.join([str(crc32(id)) for id in ids])}] AS crc32ids
            MATCH (out:SSPublication)-[:SSReferenced]->(in:SSPublication)
            WHERE in.crc32id in crc32ids AND in.ssid IN ssids
            RETURN out.ssid AS citing, COLLECT(in.ssid) AS cited, out.date.year AS year;
        '''

        with self.neo4jdriver.session() as session:
            cocit_data = []
            lines = 0
            for r in session.run(query):
                lines += 1
                citing, year, cited = r['citing'], r['year'], sorted(r['cited'])
                for i in range(len(cited)):
                    for j in range(i + 1, len(cited)):
                        cocit_data.append((citing, cited[i], cited[j], year))

        cocit_df = pd.DataFrame(cocit_data, columns=['citing', 'cited_1', 'cited_2', 'year'])

        if np.any(cocit_df[['citing', 'cited_1', 'cited_2']].isna()):
            raise ValueError('NaN values are not allowed in co-citation DataFrame')

        cocit_df['citing'] = cocit_df['citing'].apply(str)
        cocit_df['cited_1'] = cocit_df['cited_1'].apply(str)
        cocit_df['cited_2'] = cocit_df['cited_2'].apply(str)
        cocit_df['year'] = cocit_df['year'].apply(lambda x: int(x) if x else np.nan)

        self.progress.debug(f'Loaded {lines} lines of citing info', current=current, task=task)
        self.progress.info(f'Found {len(cocit_df)} co-cited pairs of papers', current=current, task=task)

        return cocit_df

    def expand(self, ids, current=1, task=None):
        expanded = set(ids)
        if isinstance(ids, Iterable):
            self.progress.info('Expanding current topic', current=current, task=task)

            # TODO[shpynov] transferring huge list of ids can be a problem
            query = f'''
            WITH [{','.join([f'"{id}"' for id in ids])}] AS ssids,
                 [{','.join([str(crc32(id)) for id in ids])}] AS crc32ids
                MATCH (out:SSPublication)-[:SSReferenced]->(in:SSPublication)
                WHERE in.crc32id IN crc32ids AND in.ssid in ssids
                RETURN COLLECT(out.ssid) AS expanded;
            '''
            with self.neo4jdriver.session() as session:
                for r in session.run(query):
                    expanded |= set(r['expanded'])

            query = f'''
            WITH [{','.join([f'"{id}"' for id in ids])}] AS ssids,
                 [{','.join([str(crc32(id)) for id in ids])}] AS crc32ids
                MATCH (out:SSPublication)-[:SSReferenced]->(in:SSPublication)
                WHERE out.crc32id IN crc32ids AND out.ssid in ssids
                RETURN COLLECT(in.ssid) AS expanded;
            '''
            with self.neo4jdriver.session() as session:
                for r in session.run(query):
                    expanded |= set(r['expanded'])
        else:
            raise TypeError('ids should be Iterable')

        self.progress.debug(f'Found {len(expanded)} papers', current=current, task=task)
        return expanded
