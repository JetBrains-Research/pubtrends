from pysrc.papers.analyzer import KeyPaperAnalyzer


class PredictAnalyzer(KeyPaperAnalyzer):
    def __init__(self, loader, config):
        super(PredictAnalyzer, self).__init__(loader, config)

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

            cit_stats_df_from_query = self.loader.load_citation_stats(self.ids)
            self.cit_stats_df = self.build_cit_stats_df(cit_stats_df_from_query, self.n_papers,
                                                        current=4, task=None)
            if len(self.cit_stats_df) == 0:
                raise RuntimeError("No citations of papers were found")

            self.df, self.min_year, self.max_year, self.citation_years = self.merge_citation_stats(
                self.pub_df, self.cit_stats_df
            )
            if len(self.df) == 0:
                raise RuntimeError("Failed to merge publications and citations")

            self.cit_df = self.loader.load_citations(self.ids)

            return self.progress.stream.getvalue()
        finally:
            self.loader.close_connection()
            self.progress.remove_handler()
