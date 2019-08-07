import community
import networkx as nx
import numpy as np
import pandas as pd

from models.test.test_loader import TestLoader
from .pm_loader import PubmedLoader
from .progress_logger import ProgressLogger
from .ss_loader import SemanticScholarLoader
from .utils import get_subtopic_descriptions, get_tfidf_words


class KeyPaperAnalyzer:
    SEED = 20190723

    def __init__(self, loader):
        self.logger = ProgressLogger()

        self.loader = loader
        loader.set_logger(self.logger)

        # Determine source to provide correct URLs to articles
        if isinstance(self.loader, PubmedLoader):
            self.source = 'pubmed'
        elif isinstance(self.loader, SemanticScholarLoader):
            self.source = 'semantic'
        elif isinstance(self.loader, TestLoader):
            self.source = 'test'
        else:
            raise TypeError("loader should be either PubmedLoader or SemanticScholarLoader (or TestLoader)")

        # Data containers
        self.terms = None
        self.ids = None
        self.df = None
        self.pub_df = None
        self.cit_df = None
        self.cocit_df = None

        # Graphs
        self.CG = None

    def launch(self, *terms, task=None):
        """:return full log"""

        try:
            # Search articles relevant to the terms
            self.terms = terms
            self.ids = self.loader.search(*terms, current=1, task=task)
            self.articles_found = len(self.ids)

            # Nothing found
            if self.articles_found == 0:
                raise RuntimeError("Nothing found")

            # Load data about publications, citations and co-citations
            self.pub_df = self.loader.load_publications(current=2, task=task)
            if len(self.pub_df) == 0:
                raise RuntimeError("Nothing found in DB")

            cit_stats_df_from_query = self.loader.load_citation_stats(current=3, task=task)
            self.cit_df = self.build_cit_df(cit_stats_df_from_query, current=4, task=task)
            if len(self.cit_df) == 0:
                raise RuntimeError("Citations stats not found DB")

            self.df = pd.merge(self.pub_df, self.cit_df, on='id', how='outer')
            if len(self.df) == 0:
                raise RuntimeError("Failed to merge publications and citations")

            self.G = self.loader.load_citations(current=5, task=task)

            cocit_grouped_df = self.loader.load_cocitations(current=6, task=task)
            self.CG = self.build_cocitation_graph(cocit_grouped_df, current=7, task=task, add_citation_edges=True)
            if len(self.CG.nodes()) == 0:
                raise RuntimeError("Failed to build co-citations graph")

            self.cocit_df = self.loader.cocit_df

            # Calculate min and max year of publications
            self.update_years(current=8, task=task)
            # Perform basic analysis
            self.subtopic_analysis(current=9, task=task)

            self.find_top_cited_papers(current=10, task=task)  # run after subtopic analysis to color components

            self.find_max_gain_papers(current=11, task=task)

            self.find_max_relative_gain_papers(current=12, task=task)

            self.subtopic_evolution_analysis(current=13, task=task)

            self.journal_stats = self.popular_journals(current=14, task=task)
            self.author_stats = self.popular_authors(current=15, task=task)

            return self.logger.stream.getvalue()
        finally:
            self.loader.close_connection()
            self.logger.remove_handler()

    def build_cit_df(self, cit_stats_df_from_query, current=None, task=None):
        cit_df = cit_stats_df_from_query.pivot(index='id', columns='year',
                                               values='count').reset_index().replace(np.nan, 0)
        cit_df['total'] = cit_df.iloc[:, 1:].sum(axis=1)
        cit_df = cit_df.sort_values(by='total', ascending=False)
        self.logger.debug(f"Loaded citation stats for {len(cit_df)} of {self.articles_found} articles.\n" +
                          "Others may either have zero citations or be absent in the local database.", current=current,
                          task=task)

        return cit_df

    def build_cocitation_graph(self, cocit_grouped_df, current=0, task=None, add_citation_edges=False,
                               citation_weight=0.3):
        self.logger.info(f'Building co-citations graph', current=current, task=task)
        CG = nx.Graph()

        # NOTE: we use nodes id as String to avoid problems str keys in jsonify
        # during graph visualization
        for el in cocit_grouped_df[['cited_1', 'cited_2', 'total']].values:
            start, end, weight = el
            CG.add_edge(str(start), str(end), weight=int(weight))

        if add_citation_edges:
            for u, v in self.G.edges:
                if CG.has_edge(u, v):
                    CG.add_edge(u, v, weight=CG[u][v]['weight'] + citation_weight)
                else:
                    CG.add_edge(u, v, weight=citation_weight)

        self.logger.debug(f'Co-citations graph nodes {len(CG.nodes())} edges {len(CG.edges())}\n',
                          current=current, task=task)
        return CG

    def update_years(self, current=0, task=None):
        self.logger.update_state(current, task=task)
        self.years = [int(col) for col in list(self.df.columns) if isinstance(col, (int, float))]
        self.min_year, self.max_year = int(self.df['year'].min()), int(self.df['year'].max())

    def subtopic_analysis(self, current=0, task=None):
        # Graph clustering via Louvain algorithm
        self.logger.info(f'Louvain community clustering of co-citation graph', current=current, task=task)
        self.logger.debug(f'Co-citation graph has {nx.number_connected_components(self.CG)} connected components',
                          current=current, task=task)
        p = community.best_partition(self.CG, random_state=KeyPaperAnalyzer.SEED)
        self.logger.debug(f'Found {len(set(p.values()))} components', current=current, task=task)
        self.logger.debug(f'Graph modularity: {community.modularity(p, self.CG):.3f}', current=current, task=task)

        # Merge small components to 'Other'
        pm, components_merged = self.merge_components(p)
        pm, self.comp_other = self.sort_components(pm, components_merged)
        self.components = set(pm.values())
        self.pm = pm
        self.pmcomp_sizes = {com: sum([pm[node] == com for node in pm.keys()]) for com in
                             self.components}
        for k, v in self.pmcomp_sizes.items():
            self.logger.debug(f'Cluster {k}: {v} ({int(100 * v / len(pm))}%)', current=current, task=task)

        # Added 'comp' column containing the ID of component
        df_comp = pd.Series(pm).reset_index().rename(columns={'index': 'id', 0: 'comp'})
        self.df = pd.merge(self.df.assign(id=self.df['id'].astype(str)),
                           df_comp.assign(id=df_comp['id'].astype(str)),
                           on='id', how='outer').fillna(-1)
        self.df['comp'] = self.df['comp'].apply(int)

        # Get n-gram descriptions for subtopics
        self.logger.debug('Getting n-gram descriptions for subtopics', current=current, task=task)
        comps = self.df.groupby('comp')['id'].apply(list).to_dict()
        kwds = get_subtopic_descriptions(self.df, comps)
        for k, v in kwds.items():
            self.logger.debug(f'{k}: {v}', current=current, task=task)
        df_kwd = pd.Series(kwds).reset_index()
        df_kwd = df_kwd.rename(columns={'index': 'comp', 0: 'kwd'})
        self.df_kwd = df_kwd
        self.logger.debug('Done\n', current=current, task=task)

    def find_top_cited_papers(self, max_papers=50, threshold=0.1, current=0, task=None):
        self.logger.info(f'Identifying top cited papers overall', current=current, task=task)
        papers_to_show = min(max_papers, round(len(self.df) * threshold))
        self.top_cited_df = self.df.sort_values(by='total',
                                                ascending=False).iloc[:papers_to_show, :]
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
        self.logger.info('Identifying papers with max relative citation gain for each year', current=current,
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

    def subtopic_evolution_analysis(self, step=5, keywords=15, min_papers=0, current=0, task=None):
        min_year = int(self.cocit_df['year'].min())
        max_year = int(self.cocit_df['year'].max())
        self.logger.info(
            f'Studying evolution of subtopic clusters in {min_year} - {max_year} with a step of {step} years',
            current=current, task=task)

        components_merged = {}
        cg = {}
        evolution_series = []
        year_range = list(np.arange(max_year, min_year - 1, step=-step).astype(int))
        self.logger.debug(f"Years when subtopics are studied: {', '.join([str(year) for year in year_range])}",
                          current=current, task=task)

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

            cg[year] = nx.Graph()
            # NOTE: we use nodes id as String to avoid problems str keys in jsonify
            # during graph visualization
            for el in cocit_grouped_df[['cited_1', 'cited_2', 'total']].values:
                cg[year].add_edge(str(el[0]), str(el[1]), weight=el[2])
            self.logger.debug(f'{year}: graph contains {len(cg[year].nodes)} nodes, {len(cg[year].edges)} edges',
                              current=current, task=task)

            if len(cg[year].nodes) >= min_papers:
                p = {vertex: int(comp) for vertex, comp in
                     community.best_partition(cg[year], random_state=KeyPaperAnalyzer.SEED).items()}
                p, components_merged[year] = self.merge_components(p)
                evolution_series.append(pd.Series(p))
                years_processed += 1
            else:
                self.logger.debug(f'Total number of papers is less than {min_papers}, stopping.',
                                  current=current, task=task)
                break

        year_range = year_range[:years_processed]

        self.evolution_df = pd.concat(evolution_series, axis=1).rename(
            columns=dict(enumerate(year_range)))
        self.evolution_df['current'] = self.evolution_df[max_year]
        self.evolution_df = self.evolution_df[list(reversed(list(self.evolution_df.columns)))]

        # Assign -1 to articles that do not belong to any cluster at some step
        self.evolution_df = self.evolution_df.fillna(-1.0)

        self.evolution_df = self.evolution_df.reset_index().rename(columns={'index': 'id'})
        self.evolution_df['id'] = self.evolution_df['id'].astype(str)

        self.evolution_kwds = {}
        for col in self.evolution_df:
            if col in year_range:
                self.logger.debug(f'Generating TF-IDF descriptions for year {col}',
                                  current=current, task=task)
                if isinstance(col, (int, float)):
                    self.evolution_df[col] = self.evolution_df[col].apply(int)
                    comps = dict(self.evolution_df.groupby(col)['id'].apply(list))
                    self.evolution_kwds[col] = get_tfidf_words(self.df, comps, self.terms, size=keywords)

        return cg, components_merged

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
            self.logger.debug(f'Processed {len(set(pm.values()))} components', current=current, task=task)
        else:
            self.logger.debug(f'All components are bigger than {granularity}, no need to reassign',
                              current=current, task=task)
            pm = p
        return pm, components_merged

    def sort_components(self, pm, components_merged):
        components = set(pm.values())
        comp_sizes = {com: sum([pm[node] == com for node in pm.keys()]) for com in components}

        argsort = lambda seq: sorted(range(len(seq)), key=seq.__getitem__, reverse=True)
        sorted_comps = list(argsort(list(comp_sizes.values())))
        mapping = dict(zip(sorted_comps, range(len(components))))
        sorted_pm = {node: mapping[c] for node, c in pm.items()}

        if components_merged:
            other = sorted_comps.index(0)
        else:
            other = None

        return sorted_pm, other

    def popular_journals(self, current=0, task=None):
        self.logger.info("Finding popular journals", current=current, task=task)
        journal_stats = self.df.groupby(['journal', 'comp']).size().reset_index(name='counts')
        # drop papers with undefined subtopic
        journal_stats = journal_stats[journal_stats.comp != -1]
        journal_stats['journal'].replace('', np.nan, inplace=True)
        journal_stats.dropna(subset=['journal'], inplace=True)

        journal_stats.sort_values(by=['journal', 'counts'], ascending=False, inplace=True)

        journal_stats = journal_stats.groupby('journal').agg(
            {'comp': lambda x: list(x), 'counts': [lambda x: list(x), 'sum']}).reset_index()

        journal_stats.columns = journal_stats.columns.droplevel(level=1)
        journal_stats.columns = ['journal', 'comp', 'counts', 'sum']

        journal_stats = journal_stats.sort_values(by=['sum'], ascending=False)

        return journal_stats.head(n=20)

    def popular_authors(self, current=0, task=None):
        self.logger.info("Finding popular authors", current=current, task=task)

        author_stats = pd.DataFrame()
        author_stats['author'] = self.df[self.df.authors != -1]['authors'].apply(lambda authors: authors.split(', '))

        author_stats = author_stats.author.apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame(
            'author').join(self.df[['id', 'comp']], how='left')

        author_stats = author_stats.groupby(['author', 'comp']).size().reset_index(name='counts')
        # drop papers with undefined subtopic
        author_stats = author_stats[author_stats.comp != -1]
        author_stats['author'].replace('', np.nan, inplace=True)
        author_stats.dropna(subset=['author'], inplace=True)

        author_stats.sort_values(by=['author', 'counts'], ascending=False, inplace=True)

        author_stats = author_stats.groupby('author').agg(
            {'comp': lambda x: list(x), 'counts': [lambda x: list(x), 'sum']}).reset_index()

        author_stats.columns = author_stats.columns.droplevel(level=1)
        author_stats.columns = ['author', 'comp', 'counts', 'sum']
        author_stats = author_stats.sort_values(by=['sum'], ascending=False)

        return author_stats.head(n=20)
