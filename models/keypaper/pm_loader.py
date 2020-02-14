import html
import logging
import re

from collections import Iterable
import numpy as np
import pandas as pd

from models.keypaper.utils import SORT_MOST_CITED, SORT_MOST_RECENT, SORT_MOST_RELEVANT
from models.keypaper.utils import preprocess_search_query
from models.keypaper.loader import Loader

logger = logging.getLogger(__name__)


class PubmedLoader(Loader):
    def __init__(self, pubtrends_config):
        super(PubmedLoader, self).__init__(pubtrends_config)

    def find(self, key, value, current=1, task=None):
        self.progress.info(f"Searching for a publication with {key} '{value}'", current=current, task=task)

        if key == 'id':
            key = 'pmid'

        # Use dedicated text index to search title.
        if key == 'title':
            query = f'''
                CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", '"{re.sub('"', '', value.strip())}"')
                YIELD node
                MATCH (p:PMPublication)
                WHERE p.pmid = node.pmid AND p.title = '{value}'
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
                CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", {query_str})
                YIELD node, score
                RETURN node.pmid as pmid
                ORDER BY score DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_CITED:
            query = f'''
                CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", {query_str}) YIELD node
                MATCH ()-[r:PMReferenced]->(in:PMPublication)
                WHERE in.pmid = node.pmid
                WITH node, COUNT(r) AS cnt
                RETURN node.pmid as pmid
                ORDER BY cnt DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_RECENT:
            query = f'''
                CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", {query_str}) YIELD node
                RETURN node.pmid as pmid
                ORDER BY node.date DESC
                LIMIT {limit};
                '''
        else:
            raise ValueError(f'Illegal sort method: {sort}')

        with self.neo4jdriver.session() as session:
            ids = [str(r['pmid']) for r in session.run(query)]

        self.progress.info(f'Found {len(ids)} publications in the local database', current=current,
                           task=task)
        return ids

    def load_publications(self, ids, current=1, task=None):
        self.progress.info('Loading publication data', current=current, task=task)

        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join([str(id) for id in ids])}] AS pmids
            MATCH (p:PMPublication)
            WHERE p.pmid IN pmids
            RETURN p.pmid as id, p.title as title, p.abstract as abstract,
                p.date.year as year, p.type as type, p.aux as aux
            ORDER BY id
        '''

        with self.neo4jdriver.session() as session:
            pub_df = pd.DataFrame(session.run(query).data())
        if len(pub_df) == 0:
            logger.debug(f'Failed to load publications.')
            pub_df = pd.DataFrame(columns=['id', 'title', 'abstract', 'year', 'type', 'aux'])
        else:
            logger.debug(f'Found {len(pub_df)} publications in the local database')
            if np.any(pub_df[['id', 'title']].isna()):
                logger.debug('Detected paper(s) without ID or title')
                pub_df.dropna(subset=['id', 'title'], inplace=True)
                logger.debug(f'Correct publications {len(pub_df)}')
            pub_df = Loader.process_publications_dataframe(pub_df)

        return pub_df

    def load_citation_stats(self, ids, current=1, task=None):
        self.progress.info('Loading citations statistics', current=current, task=task)

        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join([str(id) for id in ids])}] AS pmids
            MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication)
            WHERE in.pmid IN pmids AND out.date.year >= in.date.year
            RETURN in.pmid AS id, out.date.year AS year, COUNT(*) AS count
            LIMIT {self.max_number_of_citations};
        '''

        with self.neo4jdriver.session() as session:
            cit_stats_df = pd.DataFrame(session.run(query).data())
        if len(cit_stats_df) == 0:
            logger.debug(f'Failed to load citations statistics.')
            cit_stats_df = pd.DataFrame(columns=['id', 'year', 'count'])
        else:
            self.progress.info(f'Found {cit_stats_df.shape[0]} records of citations by year',
                               current=current, task=task)
            if np.any(cit_stats_df.isna()):
                raise ValueError('NaN values are not allowed in citation stats DataFrame')
            cit_stats_df['id'] = cit_stats_df['id'].apply(str)
            cit_stats_df['year'] = cit_stats_df['year'].apply(int)
            cit_stats_df['count'] = cit_stats_df['count'].apply(int)

        return cit_stats_df

    def load_citations(self, ids, current=1, task=None):
        """ Loading INNER citations graph, where all the nodes are inside query of interest """
        self.progress.info('Started loading citations', current=current, task=task)

        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join([str(id) for id in ids])}] AS pmids
            MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication)
            WHERE in.pmid IN pmids AND out.pmid IN pmids
            RETURN out.pmid AS id_out, in.pmid AS id_in
            ORDER BY id_out, id_in
            LIMIT {self.max_number_of_cocitations};
        '''

        with self.neo4jdriver.session() as session:
            cit_df = pd.DataFrame(session.run(query).data())
        if len(cit_df) == 0:
            logger.debug(f'Failed to load citations.')
            cit_df = pd.DataFrame(columns=['id_in', 'id_out'])
        else:
            self.progress.info(f'Found {len(cit_df)} citations', current=current, task=task)
            if np.any(cit_df.isna()):
                raise ValueError('Citation must have id_out and id_in')
            cit_df['id_out'] = cit_df['id_out'].apply(str)
            cit_df['id_in'] = cit_df['id_in'].apply(str)

        return cit_df

    def load_cocitations(self, ids, current=1, task=None):
        self.progress.info('Calculating co-citations for papers', current=current, task=task)

        # Use unfolding to pairs on the client side instead of DataBase
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join([str(id) for id in ids])}] AS pmids
            MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication)
            WHERE in.pmid IN pmids
            RETURN out.pmid AS citing, COLLECT(in.pmid) AS cited, out.date.year AS year;
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

        logger.debug(f'Loaded {lines} lines of citing info')

        cocit_df = pd.DataFrame(cocit_data, columns=['citing', 'cited_1', 'cited_2', 'year'])
        if len(cocit_data) == 0:
            logger.debug(f'Failed to load cocitations.')
        else:
            self.progress.info(f'Found {len(cocit_df)} co-cited pairs of papers', current=current, task=task)

            if np.any(cocit_df[['citing', 'cited_1', 'cited_2']].isna()):
                raise ValueError('NaN values are not allowed in co-citation DataFrame')

            cocit_df['citing'] = cocit_df['citing'].apply(str)
            cocit_df['cited_1'] = cocit_df['cited_1'].apply(str)
            cocit_df['cited_2'] = cocit_df['cited_2'].apply(str)
            cocit_df['year'] = cocit_df['year'].apply(lambda x: int(x) if x else np.nan)

        return cocit_df

    def expand(self, ids, current=1, task=None):
        expanded = set(ids)
        if isinstance(ids, Iterable):
            self.progress.info('Expanding current topic', current=current, task=task)

            # TODO[shpynov] transferring huge list of ids can be a problem
            query = f'''
                WITH [{','.join(ids)}] AS pmids
                MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication)
                WHERE in.pmid IN pmids
                RETURN COLLECT(out.pmid) AS expanded;
            '''
            with self.neo4jdriver.session() as session:
                for r in session.run(query):
                    expanded |= set([str(i) for i in r['expanded']])

            query = f'''
                WITH [{','.join(ids)}] AS pmids
                MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication)
                WHERE out.pmid IN pmids
                RETURN COLLECT(in.pmid) AS expanded;
            '''
            with self.neo4jdriver.session() as session:
                for r in session.run(query):
                    expanded |= set([str(i) for i in r['expanded']])

        else:
            raise TypeError('ids should be Iterable')

        self.progress.info(f'Found {len(expanded)} papers', current=current, task=task)
        return expanded
