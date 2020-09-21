import logging
from queue import PriorityQueue

import community
import networkx as nx
import numpy as np
import pandas as pd

from pysrc.papers.analyzer import KeyPaperAnalyzer
from pysrc.papers.utils import get_evolution_topics_description

logger = logging.getLogger(__name__)


class ExperimentalAnalyzer(KeyPaperAnalyzer):
    EVOLUTION_MIN_PAPERS = 100
    EVOLUTION_STEP = 10

    def __init__(self, loader, config, test=False):
        super().__init__(loader, config, test)

    def total_steps(self):
        return super().total_steps() + 2  # One extra step for visualization

    def analyze_papers(self, ids, query, noreviews=True, task=None):
        super().analyze_papers(ids, query, noreviews, task)
        if len(self.df) < ExperimentalAnalyzer.EVOLUTION_MIN_PAPERS:
            self.evolution_df = None
            return
        # Perform topic evolution analysis and get topic descriptions
        self.evolution_df, self.evolution_year_range = \
            self.topic_evolution_analysis(self.cocit_df, current=super().total_steps(), task=task)
        self.evolution_kwds = self.topic_evolution_descriptions(
            self.df, self.evolution_df, self.evolution_year_range, current=super().total_steps() + 1, task=task
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

        logger.debug(f"Topics evolution years: {', '.join([str(year) for year in year_range])}")
        years_processed = 1
        evolution_series = [pd.Series(self.partition)]
        for i, year in enumerate(year_range[1:]):
            self.progress.info(f'Processing year {year}', current=current, task=task)
            # Get ids earlier than year
            ids_year = set(self.df.loc[self.df['year'] <= year]['id'])

            # Use only citations earlier than year
            citations_graph_year = nx.DiGraph()
            for index, row in self.cit_df.iterrows():
                v, u = row['id_out'], row['id_in']
                if v in ids_year and u in ids_year:
                    citations_graph_year.add_edge(v, u)

            # Use only co-citations earlier than year
            cocit_grouped_df_year = self.build_cocit_grouped_df(cocit_df.loc[cocit_df['year'] <= year])

            # Use bibliographic coupling earlier then year
            bibliographic_coupling_df_year = self.bibliographic_coupling_df.loc[
                np.logical_and(
                    self.bibliographic_coupling_df['citing_1'].isin(ids_year),
                    self.bibliographic_coupling_df['citing_2'].isin(ids_year)
                )
            ]

            # Use similarities for papers earlier then year
            texts_similarity_year = self.texts_similarity.copy()
            for idx in np.flatnonzero(self.df['year'].apply(int) > year):
                texts_similarity_year[idx] = PriorityQueue(maxsize=0)

            similarity_graph = self.build_similarity_graph(
                self.df, texts_similarity_year,
                citations_graph_year,
                cocit_grouped_df_year,
                bibliographic_coupling_df_year,
                process_all_papers=False,  # Dont add all the papers to the graph
                current=current, task=task
            )
            logger.debug('Compute aggregated similarity')
            for _, _, d in similarity_graph.edges(data=True):
                d['similarity'] = KeyPaperAnalyzer.get_similarity(d)

            if len(similarity_graph.nodes) >= min_papers:
                self.progress.info('Extracting topics from paper similarity graph', current=current, task=task)
                dendrogram = community.generate_dendrogram(
                    similarity_graph, weight='similarity', random_state=KeyPaperAnalyzer.SEED
                )
                # Smallest communities
                partition_louvain = dendrogram[0]
                logger.debug(f'Found {len(set(partition_louvain.values()))} components')
                # Reorder and merge small components to 'OTHER'
                p, _ = KeyPaperAnalyzer.merge_components(
                    partition_louvain, topic_min_size=self.TOPIC_MIN_SIZE, max_topics_number=self.TOPICS_MAX_NUMBER
                )
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

        # Assign -1 to articles not published yet
        evolution_df = evolution_df.fillna(-1.0)

        evolution_df = evolution_df.reset_index().rename(columns={'index': 'id'})
        evolution_df['id'] = evolution_df['id'].astype(str)
        return evolution_df, year_range

    def topic_evolution_descriptions(self, df, evolution_df, year_range, current=0, task=None):
        # Topic evolution failed, no need to generate keywords
        if evolution_df is None or not year_range:
            return None

        self.progress.info('Generating evolution topics description by top cited papers',
                           current=current, task=task)
        evolution_kwds = {}
        for col in evolution_df:
            if col in year_range:
                self.progress.info(f'Generating topics descriptions for year {col}',
                                   current=current, task=task)
                if isinstance(col, (int, float)):
                    evolution_df[col] = evolution_df[col].apply(int)
                    comps = evolution_df.groupby(col)['id'].apply(list).to_dict()
                    evolution_kwds[col] = get_evolution_topics_description(
                        df, comps, self.corpus_ngrams, self.corpus_counts, size=KeyPaperAnalyzer.TOPIC_WORDS
                    )

        return evolution_kwds
