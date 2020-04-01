import logging

import community
import numpy as np
import pandas as pd

from models.keypaper.analyzer import KeyPaperAnalyzer
from models.keypaper.utils import get_tfidf_words

logger = logging.getLogger(__name__)


class ExperimentalAnalyzer(KeyPaperAnalyzer):

    EVOLUTION_STEP = 10

    def __init__(self, loader, config, test=False):
        super().__init__(loader, config, test)

    def total_steps(self):
        return super().total_steps() + 2

    def analyze_papers(self, ids, query, task=None):
        super().analyze_papers(ids, query, task)

        # Perform topic evolution analysis and get topic descriptions
        self.evolution_df, self.evolution_year_range = \
            self.topic_evolution_analysis(self.cocit_df, current=19, task=task)
        self.evolution_kwds = self.topic_evolution_descriptions(
            self.df, self.evolution_df, self.evolution_year_range, self.query, current=20, task=task
        )

    def topic_evolution_analysis(self, cocit_df, step=EVOLUTION_STEP, min_papers=0, current=0, task=None):
        min_year = int(cocit_df['year'].min())
        max_year = int(cocit_df['year'].max())
        year_range = list(np.arange(max_year, min_year - 1, step=-step).astype(int))

        # Cannot analyze evolution
        if len(year_range) < 2:
            self.progress.info(f'Year step is too big to analyze evolution of topics in {min_year} - {max_year}',
                               current=current, task=task)
            return None, None

        self.progress.info(f'Studying evolution of topics in {min_year} - {max_year}',
                           current=current, task=task)

        n_components_merged = {}
        similarity_graph = {}

        logger.debug(f"Topics evolution years: {', '.join([str(year) for year in year_range])}")

        # Use results of topic analysis for current year, perform analysis for other years
        years_processed = 1
        evolution_series = [pd.Series(self.partition)]
        for i, year in enumerate(year_range[1:]):
            # Use only co-citations earlier than year
            cocit_grouped_df = self.build_cocit_grouped_df(cocit_df[cocit_df['year'] <= year])
            similarity_graph[year] = self.build_similarity_graph(
                self.df, self.citations_graph, cocit_grouped_df, self.bibliographic_coupling_df,
                current=current, task=task
            )

            if len(similarity_graph[year].nodes) >= min_papers:
                p = {vertex: int(comp) for vertex, comp in
                     community.best_partition(similarity_graph[year], random_state=KeyPaperAnalyzer.SEED).items()}
                p, n_components_merged[year] = self.merge_components(p)
                evolution_series.append(pd.Series(p))
                years_processed += 1
            else:
                logger.debug(f'Total number of papers is less than {min_papers}, stopping.')
                break

        year_range = year_range[:years_processed]

        evolution_df = pd.concat(evolution_series, axis=1).rename(
            columns=dict(enumerate(year_range)))
        evolution_df['current'] = evolution_df[max_year]
        evolution_df = evolution_df[list(reversed(list(evolution_df.columns)))]

        # Assign -1 to articles that do not belong to any cluster at some step
        evolution_df = evolution_df.fillna(-1.0)

        evolution_df = evolution_df.reset_index().rename(columns={'index': 'id'})
        evolution_df['id'] = evolution_df['id'].astype(str)
        return evolution_df, year_range

    def topic_evolution_descriptions(
            self, df, evolution_df, year_range, query,
            keywords=KeyPaperAnalyzer.TOPIC_WORDS,
            current=0, task=None
    ):
        # Topic evolution failed, no need to generate keywords
        if evolution_df is None or not year_range:
            return None

        self.progress.info('Generating evolution topics description by top cited papers',
                           current=current, task=task)
        evolution_kwds = {}
        for col in evolution_df:
            if col in year_range:
                logger.debug(f'Generating topics descriptions for year {col}')
                if isinstance(col, (int, float)):
                    evolution_df[col] = evolution_df[col].apply(int)
                    comps = evolution_df.groupby(col)['id'].apply(list).to_dict()
                    evolution_kwds[col] = get_tfidf_words(df, comps, query, n_words=100, size=keywords)

        return evolution_kwds
