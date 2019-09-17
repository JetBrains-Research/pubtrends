import html
import re

from collections import Iterable

import numpy as np
import pandas as pd

from .loader import Loader


class PubmedLoader(Loader):
    def __init__(self, pubtrends_config):
        super(PubmedLoader, self).__init__(pubtrends_config)

    def find(self, key, value, current=0, task=None):
        self.logger.info(f"Searching for a publication with {key} '{value}'", current=current, task=task)

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
            return [r['pmid'] for r in session.run(query)]

    def preprocess_search_string(self, terms):
        terms_str = re.sub('[^0-9a-zA-Z"\\- ]', '', terms.strip())
        words = re.sub('"', '', terms_str).split(' ')
        if len(words) < self.pubtrends_config.min_search_words:
            raise Exception(f'Please use more specific query with >= {self.pubtrends_config.min_search_words} words')
        # Looking for complete phrase
        if re.match('"[^"]+"', terms_str):
            terms_str = '\'"' + re.sub('"', '', terms_str) + '"\''
        else:
            terms_str = '"' + ' AND '.join([f"'{w}'" for w in words]) + '"'
        return terms_str

    def search(self, terms, limit=None, sort=None, current=0, task=None):
        terms_str = self.preprocess_search_string(terms)

        if sort == 'relevance':
            sort_msg = 'most relevant'
        elif sort == 'citations':
            sort_msg = 'most cited'
        elif sort == 'year':
            sort_msg = 'most recent'
        else:
            raise ValueError(f'sort can be either citations, relevance or year, got {sort}')

        if not limit:
            limit_message = ''
            limit = self.max_number_of_articles
        else:
            limit_message = f'{limit} '

        self.logger.info(html.escape(f'Searching {limit_message}{sort_msg} publications matching <{terms}>'),
                         current=current, task=task)

        if sort == 'relevance':
            query = f'''
                CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", {terms_str}) 
                YIELD node, score
                RETURN node.pmid as pmid 
                ORDER BY score DESC 
                LIMIT {limit};
                '''
        elif sort == 'citations':
            query = f'''
                CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", {terms_str}) YIELD node
                MATCH ()-[r:PMReferenced]->(in:PMPublication) 
                WHERE in.pmid = node.pmid 
                WITH node, COUNT(r) AS cnt 
                RETURN node.pmid as pmid 
                ORDER BY cnt DESC 
                LIMIT {limit};
                '''
        elif sort == 'year':
            query = f'''
                CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", {terms_str}) YIELD node
                RETURN node.pmid as pmid 
                ORDER BY node.date DESC 
                LIMIT {limit};
                '''
        else:
            raise ValueError(f'sort can be either citations, relevance or year, got {sort}')

        with self.neo4jdriver.session() as session:
            # Duplicate rows may occur if crawler was stopped while parsing
            self.ids = list(set([r['pmid'] for r in session.run(query)]))

        self.logger.info(f'Found {len(self.ids)} publications in the local database', current=current,
                         task=task)
        return self.ids

    def load_publications(self, current=0, task=None):
        self.logger.info('Loading publication data', current=current, task=task)

        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join([f"'{id}'" for id in self.ids])}] AS pmids 
            MATCH (p:PMPublication) 
            WHERE p.pmid IN pmids
            RETURN p.pmid as id, p.title as title, p.abstract as abstract, p.date.year as year, p.aux as aux
            ORDER BY id
        '''

        with self.neo4jdriver.session() as session:
            pub_df = pd.DataFrame(session.run(query).data())

        # Parse aux
        Loader.parse_aux(pub_df)

        # pub_df.dropna(subset=['id', 'title'], inplace=True)
        if np.any(pub_df[['id', 'title']].isna()):
            raise ValueError('Paper must have PMID and title')

        pub_df = Loader.process_publications_dataframe(pub_df)

        self.logger.debug(f'Found {len(pub_df)} publications in the local database', current=current, task=task)
        return pub_df

    def search_with_given_ids(self, ids, current=0, task=None):
        self.ids = ids
        self.values = ', '.join(['({})'.format(i) for i in self.ids])
        return self.load_publications(current=current, task=task)

    def load_citation_stats(self, current=0, task=None):
        self.logger.info('Loading citations statistics among millions of citations',
                         current=current, task=task)

        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join([f"'{id}'" for id in self.ids])}] AS pmids 
            MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication) 
            WHERE in.pmid IN pmids 
            RETURN in.pmid AS id, out.date.year AS year, COUNT(*) AS count;
        '''

        with self.neo4jdriver.session() as session:
            cit_stats_df_from_query = pd.DataFrame(session.run(query).data())

        self.logger.debug('Done loading citation stats', current=current, task=task)

        if np.any(cit_stats_df_from_query.isna()):
            raise ValueError('NaN values are not allowed in citation stats DataFrame')

        cit_stats_df_from_query['year'] = cit_stats_df_from_query['year'].apply(int)
        cit_stats_df_from_query['count'] = cit_stats_df_from_query['count'].apply(int)

        self.logger.info(f'Found {cit_stats_df_from_query.shape[0]} lines of citations statistics',
                         current=current, task=task)

        return cit_stats_df_from_query

    def load_citations(self, current=0, task=None):
        """ Loading INNER citations graph, where all the nodes are inside query of interest """
        self.logger.info('Started loading citations', current=current, task=task)

        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join([f"'{id}'" for id in self.ids])}] AS pmids 
            MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication) 
            WHERE in.pmid IN pmids AND out.pmid IN pmids  
            RETURN out.pmid AS id_out, in.pmid AS id_in
            ORDER BY id_out, id_in;
        '''

        with self.neo4jdriver.session() as session:
            cit_df = pd.DataFrame(session.run(query).data())

        if np.any(cit_df.isna()):
            raise ValueError('Citation must have id_out and id_in')

        self.logger.info(f'Found {len(cit_df)} citations', current=current, task=task)

        return cit_df

    def load_cocitations(self, current=0, task=None):
        self.logger.info('Calculating co-citations for selected papers', current=current, task=task)

        # Use unfolding to pairs on the client side instead of DataBase
        # TODO[shpynov] transferring huge list of ids can be a problem
        query = f'''
            WITH [{','.join([f"'{id}'" for id in self.ids])}] AS pmids 
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

        cocit_df = pd.DataFrame(cocit_data, columns=['citing', 'cited_1', 'cited_2', 'year'], dtype=object)

        if np.any(cocit_df[['citing', 'cited_1', 'cited_2']].isna()):
            raise ValueError('NaN values are not allowed in co-citation DataFrame')
        cocit_df['year'] = cocit_df['year'].apply(lambda x: int(x) if x else np.nan)

        self.logger.debug(f'Loaded {lines} lines of citing info', current=current, task=task)
        self.logger.info(f'Found {len(cocit_df)} co-cited pairs of papers', current=current, task=task)

        return cocit_df

    def expand(self, ids, current=0, task=None):
        if isinstance(ids, Iterable):
            self.logger.info('Expanding current topic', current=current, task=task)

            # TODO[shpynov] transferring huge list of ids can be a problem
            query = f'''
                WITH [{','.join([f"'{id}'" for id in ids])}] AS pmids 
                MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication) 
                WHERE in.pmid IN pmids OR out.pmid IN pmids
                RETURN out.pmid AS citing, COLLECT(in.pmid) AS cited;
            '''
        elif isinstance(ids, int):
            query = f'''
                MATCH (out:PMPublication)-[:PMReferenced]->(in:PMPublication) 
                WHERE in.pmid = '{ids}' OR in.pmid = '{ids}'
                RETURN out.pmid AS citing, COLLECT(in.pmid) AS cited;
            '''
        else:
            raise TypeError('ids should be either int or Iterable')

        expanded = set()
        with self.neo4jdriver.session() as session:
            for r in session.run(query):
                citing, cited = r['citing'], r['cited']
                expanded.add(citing)
                expanded |= set(cited)

        self.logger.info(f'Found {len(expanded)} papers', current=current, task=task)
        return expanded
