import html

from models.keypaper.ss_loader import SemanticScholarLoader
from models.keypaper.utils import preprocess_search_query, SORT_MOST_RELEVANT, SORT_MOST_CITED, \
    SORT_MOST_RECENT


class ArxivLoader(SemanticScholarLoader):
    def __init__(self, pubtrends_config):
        super(ArxivLoader, self).__init__(pubtrends_config)

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
                CALL db.index.fulltext.queryNodes("ssTitlesAndAbstracts", {query_str}) YIELD node, score
                WHERE node.source = 'Arxiv'
                RETURN node.ssid as ssid
                ORDER BY score DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_CITED:
            query = f'''
                CALL db.index.fulltext.queryNodes("ssTitlesAndAbstracts", {query_str}) YIELD node
                MATCH ()-[r:SSReferenced]->(in:SSPublication)
                WHERE in.crc32id = node.crc32id AND in.ssid = node.ssid AND node.source = 'Arxiv'
                WITH node, COUNT(r) AS cnt
                RETURN node.ssid as ssid
                ORDER BY cnt DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_RECENT:
            query = f'''
                CALL db.index.fulltext.queryNodes("ssTitlesAndAbstracts", {query_str}) YIELD node
                WHERE node.source = 'Arxiv'
                RETURN node.ssid as ssid
                ORDER BY node.date DESC
                LIMIT {limit};
                '''
        else:
            raise ValueError(f'Illegal sort method: {sort}')
        self.progress.debug(f'Search query\n{query}', current=current, task=task)

        with self.neo4jdriver.session() as session:
            ids = [str(r['ssid']) for r in session.run(query)]

        self.progress.info(f'Found {len(ids)} publications in the local database', current=current,
                           task=task)
        return ids
