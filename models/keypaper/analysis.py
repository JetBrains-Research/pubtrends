import logging
from io import StringIO

import community
import networkx as nx
import numpy as np
import pandas as pd

from .utils import get_subtopic_descriptions


class KeyPaperAnalyzer:
    def __init__(self, loader):
        self.logger = logging.getLogger(__name__)

        self.loader = loader
        loader.set_logger(self.logger)
        self.index = loader.index

    def launch(self, *terms, task=None):
        """:return full log"""
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        self.logger.addHandler(handler)
        try:
            # Search articles relevant to the terms
            self.loader.search(*terms)
            if task:
                handler.flush()
                task.update_state(state='PROGRESS', meta={'current': 1, 'total': 10, 'log': stream.getvalue()})
            # Nothing found
            if len(getattr(self.loader, self.index + 's')) == 0:
                raise RuntimeError("Nothing found")

            # Load data about publications, citations and co-citations
            self.loader.load_publications()
            if task:
                handler.flush()
                task.update_state(state='PROGRESS', meta={'current': 2, 'total': 10, 'log': stream.getvalue()})
            if len(self.loader.pub_df) == 0:
                raise RuntimeError("Nothing found in DB")

            self.loader.load_citation_stats()
            if task:
                handler.flush()
                task.update_state(state='PROGRESS', meta={'current': 3, 'total': 10, 'log': stream.getvalue()})
            if len(self.loader.df) == 0:
                raise RuntimeError("Citations stats not found DB")

            self.df = self.loader.df

            self.loader.load_cocitations()
            if task:
                handler.flush()
                task.update_state(state='PROGRESS', meta={'current': 4, 'total': 10, 'log': stream.getvalue()})
            if len(self.loader.CG.nodes()) == 0:
                raise RuntimeError("Failed to build co-citations graph")

            self.cocit_df = self.loader.cocit_df
            self.CG = self.loader.CG

            # Calculate min and max year of publications
            self.update_years()
            if task:
                handler.flush()
                task.update_state(state='PROGRESS', meta={'current': 5, 'total': 10, 'log': stream.getvalue()})
            # Perform basic analysis
            self.subtopic_analysis()
            if task:
                handler.flush()
                task.update_state(state='PROGRESS', meta={'current': 6, 'total': 10, 'log': stream.getvalue()})
            self.find_top_cited_papers()  # run after subtopic analysis to color components
            if task:
                handler.flush()
                task.update_state(state='PROGRESS', meta={'current': 7, 'total': 10, 'log': stream.getvalue()})
            self.find_max_gain_papers()
            if task:
                handler.flush()
                task.update_state(state='PROGRESS', meta={'current': 8, 'total': 10, 'log': stream.getvalue()})
            self.find_max_relative_gain_papers()
            if task:
                handler.flush()
                task.update_state(state='PROGRESS', meta={'current': 9, 'total': 10, 'log': stream.getvalue()})
            # Not visualized anyway
            # self.subtopic_evolution_analysis()
            # if task:
            #     handler.flush()
            #     task.update_state(state='PROGRESS', meta={'current': 10, 'total': 10, 'log': stream.getvalue()})
            handler.flush()
            return stream.getvalue()
        finally:
            self.logger.removeHandler(handler)

    def update_years(self):
        self.years = [int(col) for col in list(self.df.columns) if isinstance(col, (int, float))]
        self.min_year, self.max_year = np.min(self.years), np.max(self.years)

    def find_top_cited_papers(self, max_papers=50, threshold=0.1):
        self.logger.info(f'Identifying top cited papers overall')
        papers_to_show = min(max_papers, round(len(self.df) * threshold))
        self.top_cited_df = self.df.sort_values(by='total', ascending=False).iloc[:papers_to_show, :]
        self.top_cited_papers = set(self.top_cited_df['id'].values)

    def find_max_gain_papers(self):
        self.logger.info('Identifying papers with max citation gain for each year')
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

    def find_max_relative_gain_papers(self):
        self.logger.info('Identifying papers with max relative citation gain for each year\n')
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

    def subtopic_analysis(self, sort_components_key='size'):
        # Graph clustering via Louvain algorithm
        self.logger.info(f'Louvain community clustering of co-citation graph')
        p = community.best_partition(self.CG)
        self.components = set(p.values())
        self.logger.info(f'Found {len(self.components)} components')
        self.logger.info(f'Graph modularity: {community.modularity(p, self.CG):.3f}')

        # Merge small components to 'Other'
        GRANULARITY = 0.05
        self.logger.info(f'Merging components smaller than {GRANULARITY} to "Other" component')
        threshold = int(GRANULARITY * len(p))
        comp_sizes = {com: sum([p[node] == com for node in p.keys()]) for com in self.components}
        comp_to_merge = {com: comp_sizes[com] <= threshold for com in self.components}
        self.components_merged = sum(comp_to_merge.values()) > 0
        if self.components_merged > 0:
            self.logger.info(f'Reassigning components')
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
            self.logger.info(f'Processed {len(set(pm.values()))} components')
        else:
            self.logger.info(f'All components are bigger than {GRANULARITY}, no need to reassign')
            pm = p
        self.components = set(pm.values())
        self.pm = pm
        pmcomp_sizes = {com: sum([pm[node] == com for node in pm.keys()]) for com in self.components}
        for k, v in pmcomp_sizes.items():
            self.logger.info(f'Cluster {k}: {v} ({int(100 * v / len(pm))}%)')

        # Added 'comp' column containing the ID of component
        df_comp = pd.Series(pm).reset_index().rename(columns={'index': self.index, 0: 'comp'})
        self.df = pd.merge(self.df.assign(id=self.df[self.index].astype(str)),
                           df_comp.assign(id=df_comp[self.index].astype(str)),
                           on='id')

        # Get n-gram descriptions for subtopics
        self.logger.info('Getting n-gram descriptions for subtopics')
        kwds = get_subtopic_descriptions(self.df)
        for k, v in kwds.items():
            self.logger.info(f'{k}: {v}')
        df_kwd = pd.Series(kwds).reset_index()
        df_kwd = df_kwd.rename(columns={'index': 'comp', 0: 'kwd'})
        self.df_kwd = df_kwd
        self.logger.info('Done\n')

    def subtopic_evolution_analysis(self, step=2):
        min_year = self.cocit_df['year'].min().astype(int)
        max_year = self.cocit_df['year'].max().astype(int)
        self.logger.info(
            f'Studying evolution of subtopic clusters in {min_year} - {max_year} with step of {step} years')

        evolution_series = []
        year_range = range(max_year, min_year - 1, -step)
        self.logger.info('Filtering top 100000 co-citations')
        for year in year_range:
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
            self.logger.info(f'{year}: graph contains {len(CG.nodes)} nodes, {len(CG.edges)} edges')

            p = {int(vertex): int(comp) for vertex, comp in community.best_partition(CG).items()}
            evolution_series.append(pd.Series(p))

        SHIFT = True  # use random shift to see trace of separate articles
        FILLNA = True  # NaN values sometimes cause KeyError while plotting, but sometimes not (?!)

        self.evolution_df = pd.concat(evolution_series, axis=1).rename(columns=dict(enumerate(year_range)))
        self.evolution_df['current'] = self.evolution_df[max_year]
        self.evolution_df = self.evolution_df[list(reversed(list(self.evolution_df.columns)))]

        if SHIFT:
            shift = np.random.uniform(0.25, 0.75, size=(len(self.evolution_df),))
            for year in year_range:
                self.evolution_df[year] += shift

        if FILLNA:
            self.evolution_df = self.evolution_df.fillna(-1.0)
