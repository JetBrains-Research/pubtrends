import logging
import re

import numpy as np
import pandas as pd

from pysrc.papers.db.loader import Loader
from pysrc.papers.db.neo4j_connector import Neo4jConnector
from pysrc.papers.db.neo4j_utils import preprocess_search_query_for_neo4j
from pysrc.papers.utils import SORT_MOST_CITED, SORT_MOST_RECENT, SORT_MOST_RELEVANT
from pysrc.papers.utils import preprocess_doi, preprocess_search_title

logger = logging.getLogger(__name__)


class PubmedNeo4jLoader(Neo4jConnector, Loader):
    def __init__(self, config):
        super(PubmedNeo4jLoader, self).__init__(config)

    def find(self, key, value):
        value = value.strip()

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
            value = preprocess_search_title(value)
            query = f'''
                CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", '"{re.sub('"', '', value.strip())}"')
                YIELD node
                MATCH (p:PMPublication)
                WHERE p.pmid = node.pmid AND toLower(p.title) = '{value.lower()}'
                RETURN p.pmid AS pmid;
            '''
        else:
            query = f'''
                MATCH (p:PMPublication)
                WHERE p.{key} = {repr(value)}
                RETURN p.pmid AS pmid;
            '''

        with self.neo4jdriver.session() as session:
            return [str(r['pmid']) for r in session.run(query)]

    def search(self, query, limit=None, sort=None, noreviews=True):
        query_str = preprocess_search_query_for_neo4j(query, self.config.min_search_words)
        noreviews_filter = 'WHERE node.type <> "Review"' if noreviews else ''

        if sort == SORT_MOST_RELEVANT:
            neo4j_query = f'''
                CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", '{query_str}') YIELD node, score
                {noreviews_filter}
                RETURN node.pmid as pmid
                ORDER BY score DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_CITED:
            neo4j_query = f'''
                CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", '{query_str}') YIELD node
                MATCH ()-[r:PMReferenced]->(in:PMPublication)
                WHERE in.pmid = node.pmid
                WITH node, COUNT(r) AS cnt
                {noreviews_filter}
                RETURN node.pmid as pmid
                ORDER BY cnt DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_RECENT:
            neo4j_query = f'''
                CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", '{query_str}') YIELD node
                {noreviews_filter}
                RETURN node.pmid as pmid
                ORDER BY node.date DESC
                LIMIT {limit};
                '''
        else:
            raise ValueError(f'Illegal sort method: {sort}')

        with self.neo4jdriver.session() as session:
            ids = [str(r['pmid']) for r in session.run(neo4j_query)]

        if sort == SORT_MOST_CITED and len(ids) < limit:
            # Papers with any citations may be missing, append papers by relevance
            additional_ids = self.search(query, limit=limit, sort=SORT_MOST_RELEVANT, noreviews=noreviews)
            sids = set(ids)
            for ai in additional_ids:
                if ai in sids:
                    continue
                ids.append(ai)
                if len(ids) == limit:
                    return ids

        return ids

    def load_publications(self, ids):
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join(str(id) for id in ids)}] AS pmids
            MATCH (p:PMPublication)
            WHERE p.pmid IN pmids
            RETURN p.pmid as id, p.title as title, p.abstract as abstract,
                p.date.year as year, p.type as type, p.keywords as keywords, p.mesh as mesh, p.doi as doi, p.aux as aux
            ORDER BY id;
        '''

        with self.neo4jdriver.session() as session:
            pub_df = pd.DataFrame(session.run(query).data())
        if len(pub_df) == 0:
            logger.debug('Failed to load publications.')
            pub_df = pd.DataFrame(columns=['id', 'title', 'abstract', 'year', 'type', 'keywords', 'mesh', 'doi', 'aux'])
        else:
            logger.debug(f'Found {len(pub_df)} publications in the local database')
            if np.any(pub_df[['id', 'title']].isna()):
                logger.debug('Detected paper(s) without ID or title')
                pub_df.dropna(subset=['id', 'title'], inplace=True)
                logger.debug(f'Correct publications {len(pub_df)}')
            pub_df = Loader.process_publications_dataframe(pub_df)

        return pub_df

    def load_citations_by_year(self, ids):
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join(str(id) for id in ids)}] AS pmids
            MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication)
            WHERE in.pmid IN pmids
            RETURN in.pmid AS id, out.date.year AS year, COUNT(*) AS count
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
            cit_stats_df['id'] = cit_stats_df['id'].apply(str)
            cit_stats_df['year'] = cit_stats_df['year'].apply(int)
            cit_stats_df['count'] = cit_stats_df['count'].apply(int)

        return cit_stats_df

    def estimate_citations(self, ids):
        raise Exception('Not implemented yet')

    def load_citations(self, ids):
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join(str(id) for id in ids)}] AS pmids
            MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication)
            WHERE in.pmid IN pmids AND out.pmid IN pmids
            RETURN out.pmid AS id_out, in.pmid AS id_in
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
            cit_df['id_out'] = cit_df['id_out'].apply(str)
            cit_df['id_in'] = cit_df['id_in'].apply(str)

        return cit_df

    def load_cocitations(self, ids):
        # Use unfolding to pairs on the client side instead of DataBase
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join(str(id) for id in ids)}] AS pmids
            MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication)
            WHERE in.pmid IN pmids
            RETURN out.pmid AS citing, COLLECT(in.pmid) AS cited, out.date.year AS year
            LIMIT {self.config.max_number_of_cocitations};
        '''

        with self.neo4jdriver.session() as session:
            data = []
            lines = 0
            for r in session.run(query):
                lines += 1
                citing, year, cited_list = r['citing'], r['year'], sorted(r['cited'])
                for i in range(len(cited_list)):
                    for j in range(i + 1, len(cited_list)):
                        data.append((citing, cited_list[i], cited_list[j], year))
        cocit_df = pd.DataFrame(data, columns=['citing', 'cited_1', 'cited_2', 'year'])

        logger.debug(f'Loaded {lines} lines of co-citing info')

        if len(data) == 0:
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
        # Use unfolding to pairs on the client side instead of DataBase
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join(str(id) for id in ids)}] AS pmids
            MATCH (out1:PMPublication),(out2:PMPublication)
            WHERE out1.pmid <> out2.pmid AND out1.pmid IN pmids AND out2.pmid IN pmids
            MATCH (out1:PMPublication)-[:PMReferenced]->(in:PMPublication),
                  (out2:PMPublication)-[:PMReferenced]->(in:PMPublication)
            RETURN out1.pmid AS citing_1, out2.pmid AS citing_2, COUNT(in) AS total
            LIMIT {self.config.max_number_of_bibliographic_coupling};
        '''

        with self.neo4jdriver.session() as session:
            bibliographic_coupling_df = pd.DataFrame(session.run(query).data())

        if len(bibliographic_coupling_df) == 0:
            logger.debug('Failed to load bibliographic coupling.')
            bibliographic_coupling_df = pd.DataFrame(columns=['citing_1', 'citing_2', 'total'])

        bibliographic_coupling_df['citing_1'] = bibliographic_coupling_df['citing_1'].apply(str)
        bibliographic_coupling_df['citing_2'] = bibliographic_coupling_df['citing_2'].apply(str)
        bibliographic_coupling_df['total'] = bibliographic_coupling_df['total'].apply(int)
        return bibliographic_coupling_df

    def expand(self, ids, limit):
        max_to_expand = limit
        # Cypher doesn't support any operations on unions, process two separate queries
        expanded_dfs = []

        # TODO[shpynov] transferring huge list of ids can be a problem
        # Join query sorted by citations and without any
        query_expand_out = f'''
                WITH [{','.join(str(id) for id in ids)}] AS pmids
                MATCH (out:PMPublication)-[:PMReferenced]->(in1:PMPublication)
                WHERE in1.pmid IN pmids
                MATCH ()-[r:PMReferenced]->(in2:PMPublication)
                WHERE in2.pmid = out.pmid
                WITH out, COUNT(r) AS cnt
                RETURN out.pmid as id, cnt as total
                ORDER BY cnt DESC
                LIMIT {max_to_expand}
                UNION
                WITH [{','.join(str(id) for id in ids)}] AS pmids
                MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication)
                WHERE in.pmid IN pmids
                RETURN out.pmid as id, 0 as total
                LIMIT {max_to_expand};
            '''

        with self.neo4jdriver.session() as session:
            expanded_dfs.append(pd.DataFrame(session.run(query_expand_out).data()))

        # Join query sorted by citations and without any
        query_expand_in = f'''
                WITH [{','.join(str(id) for id in ids)}] AS pmids
                MATCH (out:PMPublication)-[:PMReferenced]->(in1:PMPublication)
                WHERE out.pmid IN pmids
                MATCH ()-[r:PMReferenced]->(in2:PMPublication)
                WHERE in2.pmid = in1.pmid
                WITH in1, COUNT(r) AS cnt
                RETURN in1.pmid as id, cnt as total
                ORDER BY cnt DESC
                LIMIT {max_to_expand}
                UNION
                WITH [{','.join(str(id) for id in ids)}] AS pmids
                MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication)
                WHERE out.pmid IN pmids
                RETURN in.pmid as id, 0 as total
                LIMIT {max_to_expand};
            '''

        with self.neo4jdriver.session() as session:
            expanded_dfs.append(pd.DataFrame(session.run(query_expand_in).data()))

        expanded_df = pd.concat(expanded_dfs)
        expanded_df.sort_values(by=['total'], ascending=False, inplace=True)
        expanded_df['id'] = expanded_df['id'].astype(str)
        return expanded_df.iloc[:max_to_expand, :]
