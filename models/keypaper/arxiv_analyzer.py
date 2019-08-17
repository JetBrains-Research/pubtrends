from .analysis import KeyPaperAnalyzer


class ArxivAnalyzer(KeyPaperAnalyzer):
    def __init__(self, loader):
        super(ArxivAnalyzer, self).__init__(loader)

    def launch(self, *terms, task=None, analyze=True):
        """:return full log"""

        try:
            # Search articles relevant to the terms
            self.terms = terms
            self.ids = self.loader.search(*terms, current=1, task=task)
            self.n_papers = len(self.ids)

            # Nothing found
            if self.n_papers == 0:
                raise RuntimeError("Nothing found")

            # Load data about publications, citations and co-citations
            self.pub_df = self.loader.load_publications(current=2, task=task)
            if len(self.pub_df) == 0:
                raise RuntimeError("Nothing found in DB")

            cit_stats_df_from_query = self.loader.load_citation_stats(current=3, task=task)
            self.cit_stats_df = self.build_cit_stats_df(cit_stats_df_from_query, self.n_papers, current=4, task=task)
            if len(self.cit_stats_df) == 0:
                raise RuntimeError("No citations of papers were found")

            self.df, self.min_year, self.max_year, self.citation_years = self.merge_citation_stats(self.pub_df,
                                                                                                   self.cit_stats_df)
            if len(self.df) == 0:
                raise RuntimeError("Failed to merge publications and citations")

            self.cit_df = self.loader.load_citations(current=5, task=task)

            return self.logger.stream.getvalue()
        finally:
            self.loader.close_connection()
            self.logger.remove_handler()
