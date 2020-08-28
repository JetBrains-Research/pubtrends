import logging

from pysrc.papers.db.ss_neo4j_loader import SemanticScholarNeo4jLoader
from pysrc.papers.utils import SORT_MOST_CITED, SORT_MOST_RECENT

logger = logging.getLogger(__name__)


class SSPubmedLoader(SemanticScholarNeo4jLoader):
    def __init__(self, config):
        super(SSPubmedLoader, self).__init__(config)

    def search(self, query, limit=None, sort=None):
        raise Exception('Use search_pubmed')

    def search_pubmed(self, limit, sort='random'):
        if sort == SORT_MOST_CITED:
            query = f'''
                MATCH ()-[r:SSReferenced]->(node:SSPublication)
                WHERE node.pmid IS NOT NULL
                WITH node, COUNT(r) AS cnt
                RETURN node.ssid as ssid
                ORDER BY cnt DESC
                LIMIT {limit};
                '''
        elif sort == SORT_MOST_RECENT:
            query = f'''
                MATCH (node:SSPublication)
                WHERE node.pmid IS NOT NULL
                RETURN node.ssid as ssid
                ORDER BY node.date DESC
                LIMIT {limit};
                '''
        else:
            query = f'''
                MATCH (node:SSPublication)
                WHERE node.pmid IS NOT NULL
                RETURN node.ssid as ssid
                LIMIT {limit};
            '''

        logger.debug(f'Search query\n{query}')

        with self.neo4jdriver.session() as session:
            ids = [str(r['ssid']) for r in session.run(query)]

        return ids
