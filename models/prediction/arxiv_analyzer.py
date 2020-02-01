from models.keypaper.analysis import KeyPaperAnalyzer
from models.keypaper.utils import SORT_MOST_RELEVANT


class ArxivAnalyzer(KeyPaperAnalyzer):
    def __init__(self, loader, config):
        super(ArxivAnalyzer, self).__init__(loader, config)

    def launch(self, limit=1000, task=None):
        """:return full log"""

        try:
            # Search articles relevant to the terms
            self.ids = self.loader.search_arxiv(limit=limit, sort=SORT_MOST_RELEVANT, current=1, task=task)
            self.n_papers = len(self.ids)

            # Nothing found
            if self.n_papers == 0:
                raise RuntimeError("Nothing found")

            # Load data about publications, citations and co-citations
            self.pub_df = self.loader.load_publications(self.ids, current=2, task=task)
            if len(self.pub_df) == 0:
                raise RuntimeError("Nothing found in DB")

            cit_stats_df_from_query = self.loader.load_citation_stats(self.ids, current=3, task=task)
            self.cit_stats_df = self.build_cit_stats_df(cit_stats_df_from_query, self.n_papers,
                                                        current=4, task=task)
            if len(self.cit_stats_df) == 0:
                raise RuntimeError("No citations of papers were found")

            self.df, self.min_year, self.max_year, self.citation_years = self.merge_citation_stats(self.pub_df,
                                                                                                   self.cit_stats_df)
            if len(self.df) == 0:
                raise RuntimeError("Failed to merge publications and citations")

            self.cit_df = self.loader.load_citations(self.ids, current=5, task=task)

            return self.progress.stream.getvalue()
        finally:
            self.loader.close_connection()
            self.progress.remove_handler()
