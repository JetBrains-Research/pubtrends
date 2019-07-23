import logging
from io import StringIO

import community
import networkx as nx
import numpy as np
import pandas as pd

from .utils import get_subtopic_descriptions


class KeyPaperAnalyzer:
    SEED = 20190723

    def __init__(self, loader):
        self.logger = logging.getLogger(__name__)

        self.loader = loader
        loader.set_logger(self.logger)

        self.index = loader.index

        # Data containers
        self.terms = None
        self.df = None

        # Graphs
        self.CG = None

    def launch(self, *terms, task=None):
        """:return full log"""
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        self.logger.addHandler(handler)
        try:
            # Search articles relevant to the terms
            self.terms = terms
            self.loader.search(*terms)
            if task:
                handler.flush()
                task.update_state(state='PROGRESS',
                                  meta={'current': 1, 'total': 10, 'log': stream.getvalue()})
            # Nothing found
            if len(getattr(self.loader, self.index + 's')) == 0:
                raise RuntimeError("Nothing found")

            # Load data about publications, citations and co-citations
            self.loader.load_publications()
            if task:
                handler.flush()
                task.update_state(state='PROGRESS',
                                  meta={'current': 2, 'total': 10, 'log': stream.getvalue()})
            if len(self.loader.pub_df) == 0:
                raise RuntimeError("Nothing found in DB")

            self.loader.load_citation_stats()
            self.df = pd.merge(self.loader.pub_df, self.loader.cit_df, on='id', how='outer')
            if task:
                handler.flush()
                task.update_state(state='PROGRESS',
                                  meta={'current': 3, 'total': 10, 'log': stream.getvalue()})
            if len(self.loader.cit_df) == 0:
                raise RuntimeError("Citations stats not found DB")

            self.loader.load_cocitations()
            self.build_cocitation_graph()
            if task:
                handler.flush()
                task.update_state(state='PROGRESS',
                                  meta={'current': 4, 'total': 10, 'log': stream.getvalue()})
            if len(self.CG.nodes()) == 0:
                raise RuntimeError("Failed to build co-citations graph")

            self.cocit_df = self.loader.cocit_df

            # Calculate min and max year of publications
            self.update_years()
            if task:
                handler.flush()
                task.update_state(state='PROGRESS',
                                  meta={'current': 5, 'total': 10, 'log': stream.getvalue()})
            # Perform basic analysis
            self.subtopic_analysis()
            if task:
                handler.flush()
                task.update_state(state='PROGRESS',
                                  meta={'current': 6, 'total': 10, 'log': stream.getvalue()})
            self.find_top_cited_papers()  # run after subtopic analysis to color components
            if task:
                handler.flush()
                task.update_state(state='PROGRESS',
                                  meta={'current': 7, 'total': 10, 'log': stream.getvalue()})
            self.find_max_gain_papers()
            if task:
                handler.flush()
                task.update_state(state='PROGRESS',
                                  meta={'current': 8, 'total': 10, 'log': stream.getvalue()})
            self.find_max_relative_gain_papers()
            if task:
                handler.flush()
                task.update_state(state='PROGRESS',
                                  meta={'current': 9, 'total': 10, 'log': stream.getvalue()})
            # Not visualized anyway
            # self.subtopic_evolution_analysis()
            # if task:
            #     handler.flush()
            #     task.update_state(state='PROGRESS',
            #                       meta={'current': 10, 'total': 10, 'log': stream.getvalue()})
            handler.flush()
            return stream.getvalue()
        finally:
            self.logger.removeHandler(handler)

    def build_cocitation_graph(self):
        self.logger.info(f'Building co-citations graph')
        self.CG = nx.Graph()

        # NOTE: we use nodes id as String to avoid problems str keys in jsonify
        # during graph visualization
        for el in self.loader.cocit_grouped_df[['cited_1', 'cited_2', 'total']].values:
            start, end, weight = el
            self.CG.add_edge(str(start), str(end), weight=int(weight))
        self.logger.info(
            f'Co-citations graph nodes {len(self.CG.nodes())} edges {len(self.CG.edges())}\n')

    def update_years(self):
        self.years = [int(col) for col in list(self.df.columns) if isinstance(col, (int, float))]
        self.min_year, self.max_year = np.min(self.years), np.max(self.years)

    def subtopic_analysis(self, sort_components_key='size'):
        # Graph clustering via Louvain algorithm
        self.logger.info(f'Louvain community clustering of co-citation graph')
        p = community.best_partition(self.CG, random_state=KeyPaperAnalyzer.SEED)
        self.logger.info(f'Found {len(set(p.values()))} components')
        self.logger.info(f'Graph modularity: {community.modularity(p, self.CG):.3f}')

        # Merge small components to 'Other'
        pm, self.components_merged = self.merge_components(p)
        self.components = set(pm.values())
        self.pm = pm
        pmcomp_sizes = {com: sum([pm[node] == com for node in pm.keys()]) for com in
                        self.components}
        for k, v in pmcomp_sizes.items():
            self.logger.info(f'Cluster {k}: {v} ({int(100 * v / len(pm))}%)')

        # Added 'comp' column containing the ID of component
        df_comp = pd.Series(pm).reset_index().rename(columns={'index': 'id', 0: 'comp'})
        self.df = pd.merge(self.df.assign(id=self.df['id'].astype(str)),
                           df_comp.assign(id=df_comp['id'].astype(str)),
                           on='id')

        # Get n-gram descriptions for subtopics
        self.logger.info('Getting n-gram descriptions for subtopics')
        comps = self.df.groupby('comp')['id'].apply(list).to_dict()
        kwds = get_subtopic_descriptions(self.df, comps)
        for k, v in kwds.items():
            self.logger.info(f'{k}: {v}')
        df_kwd = pd.Series(kwds).reset_index()
        df_kwd = df_kwd.rename(columns={'index': 'comp', 0: 'kwd'})
        self.df_kwd = df_kwd
        self.logger.info('Done\n')

    def find_top_cited_papers(self, max_papers=50, threshold=0.1):
        self.logger.info(f'Identifying top cited papers overall')
        papers_to_show = min(max_papers, round(len(self.df) * threshold))
        self.top_cited_df = self.df.sort_values(by='total',
                                                ascending=False).iloc[:papers_to_show, :]
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

    def subtopic_evolution_analysis(self, step=2, min_papers=0):
        min_year = int(self.cocit_df['year'].min())
        max_year = int(self.cocit_df['year'].max())
        self.logger.info(f'Studying evolution of subtopic clusters in '
                         f'{min_year} - {max_year} with step of {step} years')

        components_merged = {}
        CG = {}
        evolution_series = []
        year_range = range(max_year, min_year - 1, -step)
        years_processed = 0
        for i, year in enumerate(year_range):
            cocit_grouped_df = self.cocit_df[self.cocit_df['year'] <= year].groupby(
                ['cited_1', 'cited_2', 'year']).count().reset_index()
            cocit_grouped_df = cocit_grouped_df.pivot_table(index=['cited_1', 'cited_2'],
                                                            columns=['year'],
                                                            values=['citing']).reset_index()
            cocit_grouped_df = cocit_grouped_df.replace(np.nan, 0)
            cocit_grouped_df['total'] = cocit_grouped_df.iloc[:, 2:].sum(axis=1)
            cocit_grouped_df = cocit_grouped_df.sort_values(by='total', ascending=False)
            cocit_grouped_df = cocit_grouped_df.iloc[:min(100000, len(cocit_grouped_df)), :]

            CG[year] = nx.Graph()
            # NOTE: we use nodes id as String to avoid problems str keys in jsonify
            # during graph visualization
            for el in cocit_grouped_df[['cited_1', 'cited_2', 'total']].values.astype(int):
                CG[year].add_edge(str(el[0]), str(el[1]), weight=el[2])
            self.logger.info(
                f'{year}: graph contains {len(CG[year].nodes)} nodes, {len(CG[year].edges)} edges')

            if len(CG[year].nodes) >= min_papers:
                p = {int(vertex): int(comp) for vertex, comp in
                     community.best_partition(CG[year], random_state=KeyPaperAnalyzer.SEED).items()}
                p, components_merged[year] = self.merge_components(p)
                evolution_series.append(pd.Series(p))
                years_processed += 1
            else:
                self.logger.info(f'Total number of papers is less than {min_papers}, stopping.')
                break

        year_range = year_range[:years_processed]

        self.evolution_df = pd.concat(evolution_series, axis=1).rename(
            columns=dict(enumerate(year_range)))
        self.evolution_df['current'] = self.evolution_df[max_year]
        self.evolution_df = self.evolution_df[list(reversed(list(self.evolution_df.columns)))]

        # Assign -1 to articles that do not belong to any cluster at some step
        self.evolution_df = self.evolution_df.fillna(-1.0)

        return CG, components_merged

    def merge_components(self, p, granularity=0.05):
        self.logger.info(f'Merging components smaller than {granularity} to "Other" component')
        threshold = int(granularity * len(p))
        components = set(p.values())
        comp_sizes = {com: sum([p[node] == com for node in p.keys()]) for com in components}
        comp_to_merge = {com: comp_sizes[com] <= threshold for com in components}
        components_merged = sum(comp_to_merge.values()) > 0
        if components_merged > 0:
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
            self.logger.info(f'All components are bigger than {granularity}, no need to reassign')
            pm = p
        return pm, components_merged
