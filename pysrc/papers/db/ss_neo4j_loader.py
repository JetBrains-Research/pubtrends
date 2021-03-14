import logging
import re

import numpy as np
import pandas as pd

from pysrc.papers.utils import SORT_MOST_CITED, SORT_MOST_RECENT, preprocess_doi, crc32
from .loader import Loader
from .neo4j_connector import Neo4jConnector
from .neo4j_utils import preprocess_search_query_for_neo4j

logger = logging.getLogger(__name__)


class SemanticScholarNeo4jLoader(Neo4jConnector, Loader):

    def __init__(self, config):
        super(SemanticScholarNeo4jLoader, self).__init__(config)

    def find(self, key, value):
        self.check_connection()
        value = value.strip()

        if key == 'id':
            key = 'ssid'

        # Preprocess DOI
        if key == 'doi':
            value = preprocess_doi(value)

        # Use dedicated text index to search title.
        if key == 'title':
            query = f'''
                CALL db.index.fulltext.queryNodes("ssTitlesAndAbstracts", '"{re.sub('"', '', value.strip())}"')
                YIELD node
                MATCH (p:SSPublication)
                WHERE p.crc32id = node.crc32id AND p.ssid = node.ssid AND toLower(p.title) = '{value.lower()}'
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

    def search(self, query, limit=None, sort=None, noreviews=True):
        self.check_connection()
        if noreviews:
            logger.debug('Type is not supported for Semantic Scholar')

        query_str = preprocess_search_query_for_neo4j(query, self.config.min_search_words)

        if sort == SORT_MOST_CITED:
            neo4j_query = f'''
                CALL db.index.fulltext.queryNodes("ssTitlesAndAbstracts", '{query_str}') YIELD node
                MATCH ()-[r:SSReferenced]->(in:SSPublication)
                WHERE in.crc32id = node.crc32id AND in.ssid = node.ssid
                WITH node, COUNT(r) AS cnt
                RETURN node.ssid as ssid
                ORDER BY cnt DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_RECENT:
            neo4j_query = f'''
                CALL db.index.fulltext.queryNodes("ssTitlesAndAbstracts", '{query_str}') YIELD node
                RETURN node.ssid as ssid
                ORDER BY node.date DESC
                LIMIT {limit};
                '''
        else:
            raise ValueError(f'Illegal sort method: {sort}')

        with self.neo4jdriver.session() as session:
            ids = [str(r['ssid']) for r in session.run(neo4j_query)]

        if sort == SORT_MOST_CITED and len(ids) < limit:
            # Papers with any citations may be missing
            additional_ids = self.search(query, limit=limit, sort=SORT_MOST_RECENT, noreviews=noreviews)
            sids = set(ids)
            for ai in additional_ids:
                if ai in sids:
                    continue
                ids.append(ai)
                if len(ids) == limit:
                    return ids

        return ids

    def load_publications(self, ids):
        self.check_connection()
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join(f'"{id}"' for id in ids)}] AS ssids,
                 [{','.join(str(crc32(id)) for id in ids)}] AS crc32ids
            MATCH (p:SSPublication)
            WHERE p.crc32id in crc32ids AND p.ssid IN ssids
            RETURN p.ssid as id, p.crc32id as crc32id, p.pmid as pmid, p.title as title, p.abstract as abstract,
                p.date.year as year, p.doi as doi, p.aux as aux
            ORDER BY id
        '''

        with self.neo4jdriver.session() as session:
            pub_df = pd.DataFrame(session.run(query).data())
        if len(pub_df) == 0:
            logger.debug('Failed to load publications.')
            pub_df = pd.DataFrame(columns=['id', 'crc32id', 'ssid', 'title', 'abstract', 'year', 'doi', 'aux'])
        else:
            logger.debug(f'Found {len(pub_df)} publications in the local database')
            if np.any(pub_df[['id', 'title']].isna()):
                logger.debug('Detected paper(s) without ID or title')
                pub_df.dropna(subset=['id', 'title'], inplace=True)
                logger.debug(f'Correct publications {len(pub_df)}')
            pub_df = Loader.process_publications_dataframe(ids, pub_df)
            # Hack for missing type in SS, see https://github.com/JetBrains-Research/pubtrends/issues/200
            pub_df['type'] = 'Article'
            pub_df['mesh'] = ''
            pub_df['keywords'] = ''
        return pub_df

    def load_citations_by_year(self, ids):
        self.check_connection()
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join(f'"{id}"' for id in ids)}] AS ssids,
                 [{','.join(str(crc32(id)) for id in ids)}] AS crc32ids
            MATCH (out:SSPublication)-[:SSReferenced]->(in:SSPublication)
            WHERE in.crc32id in crc32ids AND in.ssid IN ssids
            RETURN in.ssid AS id, out.date.year AS year, COUNT(*) AS count
            LIMIT {self.config.max_number_of_citations};
        '''

        with self.neo4jdriver.session() as session:
            cit_stats_df = pd.DataFrame(session.run(query).data())

        if len(cit_stats_df) == 0:
            logger.debug('Failed to load citations statistics.')
            cit_stats_df = pd.DataFrame(columns=['id', 'year', 'count'])
        else:
            if np.any(cit_stats_df.isna()):
                raise ValueError('NaN values are not allowed in citation stats DataFrame')

            cit_stats_df['count'] = cit_stats_df['count'].apply(int)
            cit_stats_df['id'] = cit_stats_df['id'].apply(str)
            cit_stats_df['year'] = cit_stats_df['year'].apply(int)

        return cit_stats_df

    def load_references(self, pid, limit):
        raise Exception('Not implemented yet')

    def estimate_citations(self, ids):
        raise Exception('Not implemented yet')

    def load_citations(self, ids):
        self.check_connection()
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join(f'"{id}"' for id in ids)}] AS ssids,
                 [{','.join(str(crc32(id)) for id in ids)}] AS crc32ids
            MATCH (out:SSPublication)-[:SSReferenced]->(in:SSPublication)
            WHERE in.crc32id in crc32ids AND in.ssid IN ssids AND
                  out.crc32id in crc32ids AND out.ssid IN ssids
            RETURN out.ssid AS id_out, in.ssid AS id_in
            ORDER BY id_out, id_in
            LIMIT {self.config.max_number_of_citations};
        '''

        with self.neo4jdriver.session() as session:
            cit_df = pd.DataFrame(session.run(query).data())

        if len(cit_df) == 0:
            logger.debug('Failed to load citations.')
            cit_df = pd.DataFrame(columns=['id_in', 'id_out'])
        else:
            if np.any(cit_df.isna()):
                raise ValueError('Citation must have id_out and id_in')

            cit_df['id_in'] = cit_df['id_in'].apply(str)
            cit_df['id_out'] = cit_df['id_out'].apply(str)
        return cit_df

    def load_cocitations(self, ids):
        self.check_connection()
        # Use unfolding to pairs on the client side instead of DataBase
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join(f'"{id}"' for id in ids)}] AS ssids,
                 [{','.join(str(crc32(id)) for id in ids)}] AS crc32ids
            MATCH (out:SSPublication)-[:SSReferenced]->(in:SSPublication)
            WHERE in.crc32id in crc32ids AND in.ssid IN ssids
            RETURN out.ssid AS citing, COLLECT(in.ssid) AS cited, out.date.year AS year
            LIMIT {self.config.max_number_of_cocitations};
        '''

        with self.neo4jdriver.session() as session:
            cocit_data = []
            lines = 0
            for r in session.run(query):
                lines += 1
                citing, year, cited_list = r['citing'], r['year'], sorted(r['cited'])
                for i in range(len(cited_list)):
                    for j in range(i + 1, len(cited_list)):
                        cocit_data.append((citing, cited_list[i], cited_list[j], year))

        logger.debug(f'Loaded {lines} lines of co-citing info')
        cocit_df = pd.DataFrame(cocit_data, columns=['citing', 'cited_1', 'cited_2', 'year'])
        if len(cocit_data) == 0:
            logger.debug('Failed to load cocitations.')
        else:
            if np.any(cocit_df[['citing', 'cited_1', 'cited_2']].isna()):
                raise ValueError('NaN values are not allowed in co-citation DataFrame')

            cocit_df['citing'] = cocit_df['citing'].apply(str)
            cocit_df['cited_1'] = cocit_df['cited_1'].apply(str)
            cocit_df['cited_2'] = cocit_df['cited_2'].apply(str)
            cocit_df['year'] = cocit_df['year'].apply(lambda x: int(x) if x else np.nan)

        return cocit_df

    def load_bibliographic_coupling(self, ids):
        self.check_connection()
        # Use unfolding to pairs on the client side instead of DataBase
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join(f'"{id}"' for id in ids)}] AS ssids,
                 [{','.join(str(crc32(id)) for id in ids)}] AS crc32ids
            MATCH (out1:SSPublication),(out2:SSPublication)
            WHERE NOT (out1.crc32id = out2.crc32id AND out1.ssid = out2.ssid) AND
                out1.crc32id in crc32ids AND out1.ssid IN ssids AND
                out2.crc32id in crc32ids AND out2.ssid IN ssids
            MATCH (out1:SSPublication)-[:SSReferenced]->(in:SSPublication),
                  (out2:SSPublication)-[:SSReferenced]->(in:SSPublication)
            RETURN out1.ssid AS citing_1, out2.ssid AS citing_2, COUNT(in) AS total
            LIMIT {self.config.max_number_of_bibliographic_coupling};
        '''

        with self.neo4jdriver.session() as session:
            bibliographic_coupling_df = pd.DataFrame(session.run(query).data())

        if len(bibliographic_coupling_df) == 0:
            logger.debug('Failed to load bibliographic coupling.')
            bibliographic_coupling_df = pd.DataFrame(columns=['citing_1', 'citing_2', 'total'])

        bibliographic_coupling_df['total'] = bibliographic_coupling_df['total'].apply(int)
        return bibliographic_coupling_df

    def expand(self, ids, limit):
        self.check_connection()
        max_to_expand = limit
        # Cypher doesn't support any operations on unions, process two separate queries
        expanded_dfs = []

        # TODO[shpynov] transferring huge list of ids can be a problem
        # Join query sorted by citations and without any
        query_expand_out = f'''
                WITH [{','.join(f'"{id}"' for id in ids)}] AS ssids,
                    [{','.join(str(crc32(id)) for id in ids)}] AS crc32ids
                MATCH (out:SSPublication)-[:SSReferenced]->(in1:SSPublication)
                WHERE in1.ssid IN ssids AND in1.crc32id IN crc32ids
                MATCH ()-[r:SSReferenced]->(in2:SSPublication)
                WHERE in2.ssid = out.ssid AND in2.crc32id = out.crc32id
                WITH out, COUNT(r) AS cnt
                RETURN out.ssid as id, cnt as total
                ORDER BY cnt DESC
                LIMIT {max_to_expand}
                UNION
                WITH [{','.join(f'"{id}"' for id in ids)}] AS ssids,
                    [{','.join(str(crc32(id)) for id in ids)}] AS crc32ids
                MATCH (out:SSPublication)-[:SSReferenced]->(in:SSPublication)
                WHERE in.ssid IN ssids AND in.crc32id in crc32ids
                RETURN out.ssid as id, 0 as total
                LIMIT {max_to_expand};
            '''

        with self.neo4jdriver.session() as session:
            expanded_dfs.append(pd.DataFrame(session.run(query_expand_out).data()))

        # Join query sorted by citations and without any
        query_expand_in = f'''
                WITH [{','.join(f'"{id}"' for id in ids)}] AS ssids,
                    [{','.join(str(crc32(id)) for id in ids)}] AS crc32ids
                MATCH (out:SSPublication)-[:SSReferenced]->(in1:SSPublication)
                WHERE out.ssid IN ssids AND out.crc32id in crc32ids
                MATCH ()-[r:SSReferenced]->(in2:SSPublication)
                WHERE in2.ssid = in1.ssid AND in2.crc32id = in1.crc32id
                WITH in1, COUNT(r) AS cnt
                RETURN in1.ssid as id, cnt as total
                ORDER BY cnt DESC
                LIMIT {max_to_expand}
                UNION
                WITH [{','.join(f'"{id}"' for id in ids)}] AS ssids,
                    [{','.join(str(crc32(id)) for id in ids)}] AS crc32ids
                MATCH (out:SSPublication)-[:SSReferenced]->(in:SSPublication)
                WHERE out.ssid IN ssids AND out.crc32id IN crc32ids
                RETURN in.ssid as id, 0 as total
                LIMIT {max_to_expand};
            '''

        with self.neo4jdriver.session() as session:
            expanded_dfs.append(pd.DataFrame(session.run(query_expand_in).data()))

        expanded_df = pd.concat(expanded_dfs)
        expanded_df.sort_values(by=['total'], ascending=False, inplace=True)
        expanded_df['id'] = expanded_df['id'].astype(str)
        return expanded_df.iloc[:max_to_expand, :]
