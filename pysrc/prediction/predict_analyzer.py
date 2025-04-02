import logging

import pandas as pd

from pysrc.papers.analysis.citations import build_cit_stats_df, merge_citation_stats
from pysrc.papers.analyzer import PapersAnalyzer

logger = logging.getLogger(__name__)

class PredictAnalyzer(PapersAnalyzer):
    def __init__(self, loader, config):
        super(PredictAnalyzer, self).__init__(loader, config)

    def search(self, limit, start_year=1970, end_year=2050, noreviews=True):
        self.loader.check_connection()
        noreviews_filter = "AND type != 'Review'" if noreviews else ''
        query = f'''
                SELECT pmid
                FROM PMPublications P
                WHERE year >= {start_year} AND year <= {end_year} {noreviews_filter}
                ORDER by random() 
                LIMIT {limit};
            '''
        logger.debug(f'find query: {query[:1000]}')
        with self.loader.postgres_connection.cursor() as cursor:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=['pmid'], dtype=object)
        return list(df['pmid'].astype(str))


    def analyze(self, ids):
        try:
            # Search articles relevant to the terms
            self.ids = ids
            self.n_papers = len(self.ids)

            # Nothing found
            if self.n_papers == 0:
                raise RuntimeError("Nothing found")

            # Load data about publications, citations and co-citations
            self.pub_df = self.loader.load_publications(self.ids)
            if len(self.pub_df) == 0:
                raise RuntimeError("Nothing found in DB")

            cit_stats_df_from_query = self.loader.load_citations_by_year(self.ids)
            self.cit_stats_df = build_cit_stats_df(cit_stats_df_from_query, self.n_papers)
            if len(self.cit_stats_df) == 0:
                raise RuntimeError("No citations of papers were found")

            self.df, self.citation_years = merge_citation_stats(ids, self.pub_df, self.cit_stats_df)
            if len(self.df) == 0:
                raise RuntimeError("Failed to merge publications and citations")
            self.min_year, self.max_year = self.df['year'].min(), self.df['year'].max()

            self.cit_df = self.loader.load_citations(self.ids)
        finally:
            self.loader.close_connection()
            self.progress.remove_handler()
