import html

from models.keypaper.ss_loader import SemanticScholarLoader
from models.keypaper.utils import preprocess_search_query, SORT_MOST_RELEVANT, SORT_MOST_CITED, \
    SORT_MOST_RECENT


class ArxivLoader(SemanticScholarLoader):
    def __init__(self, pubtrends_config):
        super(ArxivLoader, self).__init__(pubtrends_config)

    def search(self, query, limit=None, sort=None, current=1, task=None):
        raise Exception('Use search_arxiv')

    def search_arxiv(self, limit, sort=None, current=1, task=None):
        self.progress.info(html.escape(f'Searching {limit} {sort.lower()} publications'),
                           current=current, task=task)

        if sort == SORT_MOST_CITED:
            query = f'''
                MATCH ()-[r:SSReferenced]->(node:SSPublication)
                WHERE toLower(node.aux) CONTAINS "arxiv"
                WITH node, COUNT(r) AS cnt
                RETURN node.ssid as ssid
                ORDER BY cnt DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_RECENT:
            query = f'''
                MATCH (node:SSPublication)
                WHERE toLower(node.aux) CONTAINS "arxiv"
                RETURN node.ssid as ssid
                ORDER BY node.date DESC
                LIMIT {limit};
                '''
        else:
            query = f'''
                MATCH (node:SSPublication)
                WHERE toLower(node.aux) CONTAINS "arxiv"
                RETURN node.ssid as ssid
                LIMIT {limit};
            '''

        self.progress.debug(f'Search query\n{query}', current=current, task=task)

        with self.neo4jdriver.session() as session:
            ids = [str(r['ssid']) for r in session.run(query)]

        self.progress.info(f'Found {len(ids)} publications in the local database', current=current,
                           task=task)
        return ids
