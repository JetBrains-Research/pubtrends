import html
import logging

from pysrc.papers.ss_loader import SemanticScholarLoader
from pysrc.papers.utils import SORT_MOST_CITED, SORT_MOST_RECENT

logger = logging.getLogger(__name__)


class SSArxivLoader(SemanticScholarLoader):
    def __init__(self, config):
        super(SSArxivLoader, self).__init__(config)

    def search(self, query, limit=None, sort=None, current=1, task=None):
        raise Exception('Use search_arxiv')

    def search_arxiv(self, limit, sort='random'):
        self.progress.info(html.escape(f'Searching {limit} {sort.lower()} publications'),
                           current=1, task=None)

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

        logger.debug(f'Search query\n{query}')

        with self.neo4jdriver.session() as session:
            ids = [str(r['ssid']) for r in session.run(query)]

        self.progress.info(f'Found {len(ids)} publications in the local database', current=1,
                           task=None)
        return ids
