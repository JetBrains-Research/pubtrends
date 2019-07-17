import logging

import community
import networkx as nx
import numpy as np
import pandas as pd

from .progress_logger import ProgressLogger
from .utils import get_subtopic_descriptions


class KeyPaperAnalyzer:
    def __init__(self, loader):
        self.logger = logging.getLogger(__name__)

        self.logger = ProgressLogger()

        self.loader = loader
        loader.set_logger(self.logger)
        self.index = loader.index

    def launch(self, *terms, task=None):
        """:return full log"""

        try:
            # Search articles relevant to the terms
            self.loader.search(*terms, current=1, task=task)

            # Nothing found
            if len(getattr(self.loader, self.index + 's')) == 0:
                raise RuntimeError("Nothing found")

            # Load data about publications, citations and co-citations
            self.loader.load_publications(current=2, task=task)
            if len(self.loader.pub_df) == 0:
                raise RuntimeError("Nothing found in DB")

            self.loader.load_citation_stats(current=3, task=task)
            if len(self.loader.df) == 0:
                raise RuntimeError("Citations stats not found DB")

            self.df = self.loader.df

            self.loader.load_cocitations(current=4, task=task)
            if len(self.loader.CG.nodes()) == 0:
                raise RuntimeError("Failed to build co-citations graph")

            self.cocit_df = self.loader.cocit_df
            self.CG = self.loader.CG

            # Calculate min and max year of publications
            self.update_years(current=5, task=task)
            # Perform basic analysis
            self.subtopic_analysis(current=6, task=task)

            self.find_top_cited_papers(current=7, task=task)  # run after subtopic analysis to color components

            self.find_max_gain_papers(current=8, task=task)

            self.find_max_relative_gain_papers(current=9, task=task)

            # Not visualized anyway
            # self.subtopic_evolution_analysis(current=10, task=task)
            return self.logger.stream.getvalue()
        finally:
            self.loader.close_connection()
            self.logger.remove_handler()

    def update_years(self, current=0, task=None):
        self.logger.update_state(current, task=task)
        self.years = [int(col) for col in list(self.df.columns) if isinstance(col, (int, float))]
        self.min_year, self.max_year = np.min(self.years), np.max(self.years)

    def find_top_cited_papers(self, max_papers=50, threshold=0.1, current=0, task=None):
        self.logger.info(f'Identifying top cited papers overall', current=current, task=task)
        papers_to_show = min(max_papers, round(len(self.df) * threshold))
        self.top_cited_df = self.df.sort_values(by='total', ascending=False).iloc[:papers_to_show, :]
        self.top_cited_papers = set(self.top_cited_df['id'].values)

    def find_max_gain_papers(self, current=0, task=None):
        self.logger.info('Identifying papers with max citation gain for each year', current=current, task=task)
        max_gain_data = []
        for year in self.years:
            max_gain = self.df[year].astype(int).max()
            if max_gain > 0:
                sel = self.df[self.df[year] == max_gain]
                max_gain_data.append([year, str(sel['id'].values[0]),
                                      sel['title'].values[0],
                                      sel['authors'].values[0],
                                      sel['year'].values[0], max_gain])

        self.max_gain_df = pd.DataFrame(max_gain_data,
                                        columns=['year', 'id', 'title', 'authors',
                                                 'paper_year', 'count'])
        self.max_gain_papers = set(self.max_gain_df['id'].values)

    def find_max_relative_gain_papers(self, current=0, task=None):
        self.logger.info('Identifying papers with max relative citation gain for each year\n', current=current,
                         task=task)
        current_sum = pd.Series(np.zeros(len(self.df), ))
        df_rel = self.df.loc[:, ['id', 'title', 'authors', 'year']]
        for year in self.years:
            df_rel[year] = self.df[year] / (current_sum + (current_sum == 0))
            current_sum += self.df[year]

        max_rel_gain_data = []
        for year in self.years:
            max_rel_gain = df_rel[year].max()
            if max_rel_gain > 1e-6:
                sel = df_rel[df_rel[year] == max_rel_gain]
                max_rel_gain_data.append([year, str(sel['id'].values[0]),
                                          sel['title'].values[0],
                                          sel['authors'].values[0],
                                          sel['year'].values[0], max_rel_gain])

        self.max_rel_gain_df = pd.DataFrame(max_rel_gain_data,
                                            columns=['year', 'id', 'title', 'authors',
                                                     'paper_year', 'rel_gain'])
        self.max_rel_gain_papers = set(self.max_rel_gain_df['id'].values)

    def subtopic_analysis(self, sort_components_key='size', current=0, task=None):
        # Graph clustering via Louvain algorithm
        self.logger.info(f'Analyzing subtopics: clustering co-citation graph', current=current, task=task)
        p = community.best_partition(self.CG)
        self.logger.debug(f'Found {len(set(p.values()))} components', current=current, task=task)
        self.logger.debug(f'Graph modularity: {community.modularity(p, self.CG):.3f}', current=current, task=task)

        # Merge small components to 'Other'
        pm, self.components_merged = self.merge_components(p, current=current, task=task)
        self.components = set(pm.values())
        self.pm = pm
        self.pmcomp_sizes = {com: sum([pm[node] == com for node in pm.keys()]) for com in self.components}
        for k, v in self.pmcomp_sizes.items():
            self.logger.debug(f'Cluster {k}: {v} ({int(100 * v / len(pm))}%)', current=current, task=task)

        # Added 'comp' column containing the ID of component
        df_comp = pd.Series(pm).reset_index().rename(columns={'index': self.index, 0: 'comp'})
        self.df = pd.merge(self.df.assign(id=self.df[self.index].astype(str)),
                           df_comp.assign(id=df_comp[self.index].astype(str)),
                           on='id')

        # Get n-gram descriptions for subtopics
        self.logger.debug('Getting n-gram descriptions for subtopics', current=current, task=task)
        kwds = get_subtopic_descriptions(self.df)
        for k, v in kwds.items():
            self.logger.debug(f'{k}: {v}', current=current, task=task)
        df_kwd = pd.Series(kwds).reset_index()
        df_kwd = df_kwd.rename(columns={'index': 'comp', 0: 'kwd'})
        self.df_kwd = df_kwd
        self.logger.debug('Done\n', current=current, task=task)

    def merge_components(self, p, granularity=0.05, current=0, task=None):
        self.logger.debug(f'Merging components smaller than {granularity} to "Other" component',
                          current=current, task=task)
        threshold = int(granularity * len(p))
        components = set(p.values())
        comp_sizes = {com: sum([p[node] == com for node in p.keys()]) for com in components}
        comp_to_merge = {com: comp_sizes[com] <= threshold for com in components}
        components_merged = sum(comp_to_merge.values()) > 0
        if components_merged > 0:
            self.logger.debug(f'Reassigning components', current=current, task=task)
            pm = {}
            newcomps = {}
            ci = 1  # Other component is 0.
            for k, v in p.items():
                if comp_sizes[v] <= threshold:
                    pm[k] = 0  # Other
                    continue
                if v not in newcomps:
                    newcomps[v] = ci
                    ci += 1
                pm[k] = newcomps[v]
            self.logger.debug(f'Processed {len(set(pm.values()))} components',
                              current=current, task=task)
        else:
            self.logger.debug(f'All components are bigger than {granularity}, no need to reassign',
                              current=current, task=task)
            pm = p
        return pm, components_merged

    def subtopic_evolution_analysis(self, step=2, min_papers=0, current=0, task=None):
        min_year = int(self.cocit_df['year'].min())
        max_year = int(self.cocit_df['year'].max())
        self.logger.debug(
            f'Studying evolution of subtopic clusters in {min_year} - {max_year} with step of {step} years',
            current=current, task=task)

        components_merged = {}
        evolution_series = []
        year_range = range(max_year, min_year - 1, -step)
        years_processed = 0
        for i, year in enumerate(year_range):
            cocit_grouped_df = self.cocit_df[self.cocit_df['year'] <= year].groupby(
                ['cited_1', 'cited_2', 'year']).count().reset_index()
            cocit_grouped_df = cocit_grouped_df.pivot_table(index=['cited_1', 'cited_2'],
                                                            columns=['year'], values=['citing']).reset_index()
            cocit_grouped_df = cocit_grouped_df.replace(np.nan, 0)
            cocit_grouped_df['total'] = cocit_grouped_df.iloc[:, 2:].sum(axis=1)
            cocit_grouped_df = cocit_grouped_df.sort_values(by='total', ascending=False)
            cocit_grouped_df = cocit_grouped_df.iloc[:min(100000, len(cocit_grouped_df)), :]

            CG = nx.Graph()
            # NOTE: we use nodes id as String to avoid problems str keys in jsonify during graph visualization
            for el in cocit_grouped_df[['cited_1', 'cited_2', 'total']].values.astype(int):
                CG.add_edge(str(el[0]), str(el[1]), weight=el[2])
            self.logger.debug(f'{year}: graph contains {len(CG.nodes)} nodes, {len(CG.edges)} edges', current=current,
                              task=task)

            if len(CG.nodes) >= min_papers:
                p = {int(vertex): int(comp) for vertex, comp in community.best_partition(CG).items()}
                p, components_merged[year] = self.merge_components(p)
                evolution_series.append(pd.Series(p))
                years_processed += 1
            else:
                self.logger.info(f'Total number of papers is less than {min_papers}, stopping.')
                break

        year_range = year_range[:years_processed]

        self.evolution_df = pd.concat(evolution_series, axis=1).rename(columns=dict(enumerate(year_range)))
        self.evolution_df['current'] = self.evolution_df[max_year]
        self.evolution_df = self.evolution_df[list(reversed(list(self.evolution_df.columns)))]

        # Assign -1 to articles that do not belong to any cluster at some step
        self.evolution_df = self.evolution_df.fillna(-1.0)

        return components_merged
