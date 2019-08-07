import community
import networkx as nx
import numpy as np
import pandas as pd

from .pm_loader import PubmedLoader
from .progress_logger import ProgressLogger
from .ss_loader import SemanticScholarLoader
from .utils import get_subtopic_descriptions, get_tfidf_words


# from memory_profiler import profile


class KeyPaperAnalyzer:
    SEED = 20190723

    def __init__(self, loader, test=False):
        self.logger = ProgressLogger()

        self.loader = loader
        loader.set_logger(self.logger)

        # Determine source to provide correct URLs to articles
        if isinstance(self.loader, PubmedLoader):
            self.source = 'pubmed'
        elif isinstance(self.loader, SemanticScholarLoader):
            self.source = 'semantic'
        elif not test:
            raise TypeError("loader should be either PubmedLoader or SemanticScholarLoader")

    # @profile
    def launch(self, *terms, task=None):
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
            self.cit_stats_df = self.build_cit_df(cit_stats_df_from_query, self.n_papers, current=3.5, task=task)
            if len(self.cit_stats_df) == 0:
                raise RuntimeError("Citations stats not found DB")

            self.df, self.min_year, self.max_year, self.citation_years = self.merge_citation_stats(self.pub_df,
                                                                                                   self.cit_stats_df)
            if len(self.df) == 0:
                raise RuntimeError("Failed to merge publications and citations")

            self.cocit_df = self.loader.load_cocitations(current=4, task=task)
            cocit_grouped_df = self.build_cocit_grouped_df(self.cocit_df)
            self.CG = self.build_cocitation_graph(cocit_grouped_df, current=5, task=task)
            if len(self.CG.nodes()) == 0:
                raise RuntimeError("Failed to build co-citations graph")

            # Perform subtopic analysis and get subtopic descriptions
            self.df, self.components, self.comp_other, self.pm, self.pmcomp_sizes = self.subtopic_analysis(
                self.df, self.CG, current=7, task=task
            )
            self.df_kwd = self.subtopic_descriptions(self.df)

            # Find interesting papers
            self.top_cited_papers, self.top_cited_df = self.find_top_cited_papers(self.df, current=8, task=task)

            self.max_gain_papers, self.max_gain_df = self.find_max_gain_papers(self.df, self.citation_years,
                                                                               current=9, task=task)

            self.max_rel_gain_papers, self.max_rel_gain_df = self.find_max_relative_gain_papers(
                self.df, self.citation_years, current=10, task=task
            )

            # Perform subtopic evolution analysis and get subtopic descriptions
            self.evolution_df, self.evolution_year_range = self.subtopic_evolution_analysis(self.cocit_df, current=11,
                                                                                            task=task)
            self.evolution_kwds = self.subtopic_evolution_descriptions(self.df, self.evolution_df,
                                                                       self.evolution_year_range, self.terms)

            # Find top journals
            self.journal_stats = self.popular_journals(self.df, current=12, task=task)

            # Find top authors
            self.author_stats = self.popular_authors(self.df, current=13, task=task)

            return self.logger.stream.getvalue()
        finally:
            self.loader.close_connection()
            self.logger.remove_handler()

    def build_cit_df(self, cit_stats_df_from_query, n_papers, current=None, task=None):
        # Get citation stats with columns 'id', year_1, ..., year_N and fill NaN with 0
        cit_df = cit_stats_df_from_query.pivot(index='id', columns='year',
                                               values='count').reset_index().fillna(0)

        # Fix column names from float 'YYYY.0' to int 'YYYY'
        mapper = {}
        for col in cit_df.columns:
            if col != 'id':
                mapper[col] = int(col)
        cit_df = cit_df.rename(mapper)

        cit_df['total'] = cit_df.iloc[:, 1:].sum(axis=1)
        cit_df = cit_df.sort_values(by='total', ascending=False)
        self.logger.debug(f"Loaded citation stats for {len(cit_df)} of {n_papers} articles.\n" +
                          "Others may either have zero citations or be absent in the local database.",
                          current=current, task=task)

        return cit_df

    def build_cocit_grouped_df(self, cocit_df, current=0, task=None):
        self.logger.debug(f'Aggregating co-citations', current=current, task=task)
        cocit_grouped_df = cocit_df.groupby(['cited_1', 'cited_2', 'year']).count().reset_index()
        cocit_grouped_df = cocit_grouped_df.pivot_table(index=['cited_1', 'cited_2'],
                                                        columns=['year'], values=['citing']).reset_index()
        cocit_grouped_df = cocit_grouped_df.replace(np.nan, 0)
        cocit_grouped_df['total'] = cocit_grouped_df.iloc[:, 2:].sum(axis=1)
        cocit_grouped_df = cocit_grouped_df.sort_values(by='total', ascending=False)
        self.logger.debug(f'Filtering top {self.loader.max_number_of_cocitations} of all the co-citations',
                          current=current, task=task)
        cocit_grouped_df = cocit_grouped_df.iloc[:min(self.loader.max_number_of_cocitations,
                                                      len(cocit_grouped_df)), :]

        for col in cocit_grouped_df:
            cocit_grouped_df[col] = cocit_grouped_df[col].astype(object)

        return cocit_grouped_df

    @staticmethod
    def merge_citation_stats(pub_df, cit_df):
        df = pd.merge(pub_df, cit_df, on='id', how='outer')

        # Fill only new columns to preserve year NaN values
        df[cit_df.columns] = df[cit_df.columns].fillna(0)

        # Publication and citation year range
        citation_years = [int(col) for col in list(df.columns) if isinstance(col, (int, float))]
        min_year, max_year = int(df['year'].min()), int(df['year'].max())

        return df, min_year, max_year, citation_years

    def build_citation_graph(self, cit_df, current=0, task=None):
        G = nx.DiGraph()
        for row in cit_df.iterrows():
            v, u = row['id_out'], row['id_in']
            G.add_edge(v, u)

        self.logger.debug(f'Built citation graph - nodes {len(G.nodes())} edges {len(G.edges())}',
                          current=current, task=task)
        return G

    def build_cocitation_graph(self, cocit_grouped_df, year=None, current=0, task=None):
        if year:
            self.logger.info(f'Building co-citations graph for {year} year', current=current, task=task)
        else:
            self.logger.info(f'Building co-citations graph', current=current, task=task)
        CG = nx.Graph()

        # NOTE: we use nodes id as String to avoid problems str keys in jsonify
        # during graph visualization
        for el in cocit_grouped_df[['cited_1', 'cited_2', 'total']].values:
            start, end, weight = el
            CG.add_edge(str(start), str(end), weight=int(weight))
        self.logger.debug(f'Co-citations graph nodes {len(CG.nodes())} edges {len(CG.edges())}\n',
                          current=current, task=task)
        return CG

    def subtopic_analysis(self, df, cocitation_graph, current=0, task=None):
        self.logger.info(f'Louvain community clustering of co-citation graph', current=current, task=task)
        connected_components = nx.number_connected_components(cocitation_graph)
        self.logger.debug(f'Co-citation graph has {connected_components} connected components',
                          current=current, task=task)

        # Graph clustering via Louvain algorithm
        p = community.best_partition(cocitation_graph, random_state=KeyPaperAnalyzer.SEED)
        self.logger.debug(f'Found {len(set(p.values()))} components', current=current, task=task)

        # Calculate modularity for partition
        modularity = community.modularity(p, cocitation_graph)
        self.logger.debug(f'Graph modularity (possible range is [-1, 1]): {modularity :.3f}',
                          current=current, task=task)

        # Merge small components to 'Other'
        pm, components_merged = self.merge_components(p)
        pm, comp_other = self.sort_components(pm, components_merged)
        components = set(pm.values())
        pmcomp_sizes = {com: sum([pm[node] == com for node in pm.keys()]) for com in
                        components}
        for k, v in pmcomp_sizes.items():
            self.logger.debug(f'Cluster {k}: {v} ({int(100 * v / len(pm))}%)', current=current, task=task)

        # Added 'comp' column containing the ID of component
        df_comp = pd.Series(pm).reset_index().rename(columns={'index': 'id', 0: 'comp'})
        df_comp['id'] = df_comp['id'].astype(str)
        df_merged = pd.merge(df, df_comp,
                             on='id', how='outer')
        df_merged['comp'] = df_merged['comp'].fillna(-1).apply(int)
        return df_merged, components, comp_other, pm, pmcomp_sizes

    def subtopic_descriptions(self, df, current=0, task=None):
        # Get n-gram descriptions for subtopics
        self.logger.debug('Getting n-gram descriptions for subtopics', current=current, task=task)
        comps = df.groupby('comp')['id'].apply(list).to_dict()
        kwds = get_subtopic_descriptions(df, comps)
        for k, v in kwds.items():
            self.logger.debug(f'{k}: {v}', current=current, task=task)
        df_kwd = pd.Series(kwds).reset_index()
        df_kwd = df_kwd.rename(columns={'index': 'comp', 0: 'kwd'})
        self.logger.debug('Done\n', current=current, task=task)
        return df_kwd

    def find_top_cited_papers(self, df, max_papers=50, threshold=0.1, min_papers=1, current=0, task=None):
        self.logger.info(f'Identifying top cited papers overall', current=current, task=task)
        papers_to_show = max(min(max_papers, round(len(df) * threshold)), min_papers)
        top_cited_df = df.sort_values(by='total',
                                      ascending=False).iloc[:papers_to_show, :]
        top_cited_papers = set(top_cited_df['id'].values)
        return top_cited_papers, top_cited_df

    def find_max_gain_papers(self, df, citation_years, current=0, task=None):
        self.logger.info('Identifying papers with max citation gain for each year', current=current, task=task)
        max_gain_data = []
        for year in citation_years:
            max_gain = df[year].astype(int).max()
            if max_gain > 0:
                sel = df[df[year] == max_gain]
                max_gain_data.append([year, str(sel['id'].values[0]),
                                      sel['title'].values[0],
                                      sel['authors'].values[0],
                                      sel['year'].values[0], max_gain])

        max_gain_df = pd.DataFrame(max_gain_data,
                                   columns=['year', 'id', 'title', 'authors',
                                            'paper_year', 'count'])
        max_gain_papers = set(max_gain_df['id'].values)
        return max_gain_papers, max_gain_df

    def find_max_relative_gain_papers(self, df, citation_years, current=0, task=None):
        self.logger.info('Identifying papers with max relative citation gain for each year', current=current,
                         task=task)
        current_sum = pd.Series(np.zeros(len(df), ))
        df_rel = df.loc[:, ['id', 'title', 'authors', 'year']]
        for year in citation_years:
            df_rel[year] = df[year] / (current_sum + (current_sum == 0))
            current_sum += df[year]

        max_rel_gain_data = []
        for year in citation_years:
            max_rel_gain = df_rel[year].max()
            if max_rel_gain > 1e-6:
                sel = df_rel[df_rel[year] == max_rel_gain]
                max_rel_gain_data.append([year, str(sel['id'].values[0]),
                                          sel['title'].values[0],
                                          sel['authors'].values[0],
                                          sel['year'].values[0], max_rel_gain])

        max_rel_gain_df = pd.DataFrame(max_rel_gain_data,
                                       columns=['year', 'id', 'title', 'authors',
                                                'paper_year', 'rel_gain'])
        max_rel_gain_papers = set(max_rel_gain_df['id'].values)
        return max_rel_gain_papers, max_rel_gain_df

    def subtopic_evolution_analysis(self, cocit_df, step=5, min_papers=0, current=0, task=None):
        min_year = int(cocit_df['year'].min())
        max_year = int(cocit_df['year'].max())
        self.logger.info(f'Studying evolution of subtopic clusters in {min_year} - {max_year}',
                         current=current, task=task)

        components_merged = {}
        cg = {}
        year_range = list(np.arange(max_year, min_year - 1, step=-step).astype(int))
        self.logger.debug(f"Years when subtopics are studied: {', '.join([str(year) for year in year_range])}",
                          current=current, task=task)

        # Use results of subtopic analysis for current year, perform analysis for other years
        years_processed = 1
        evolution_series = [pd.Series(self.pm)]
        for i, year in enumerate(year_range[1:]):
            # Use only co-citations earlier than year
            cocit_grouped_df = self.build_cocit_grouped_df(cocit_df[cocit_df['year'] <= year])
            cg[year] = self.build_cocitation_graph(cocit_grouped_df, year=year, current=current, task=task)

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

        evolution_df = pd.concat(evolution_series, axis=1).rename(
            columns=dict(enumerate(year_range)))
        evolution_df['current'] = evolution_df[max_year]
        evolution_df = evolution_df[list(reversed(list(evolution_df.columns)))]

        # Assign -1 to articles that do not belong to any cluster at some step
        evolution_df = evolution_df.fillna(-1.0)

        evolution_df = evolution_df.reset_index().rename(columns={'index': 'id'})
        evolution_df['id'] = evolution_df['id'].astype(str)
        return evolution_df, year_range

    def subtopic_evolution_descriptions(self, df, evolution_df, year_range, terms, keywords=15, current=0, task=None):
        evolution_kwds = {}
        for col in evolution_df:
            if col in year_range:
                self.logger.debug(f'Generating TF-IDF descriptions for year {col}',
                                  current=current, task=task)
                if isinstance(col, (int, float)):
                    evolution_df[col] = evolution_df[col].apply(int)
                    comps = dict(evolution_df.groupby(col)['id'].apply(list))
                    evolution_kwds[col] = get_tfidf_words(df, comps, terms, size=keywords)

        return evolution_kwds

    def merge_components(self, p, granularity=0.05, current=0, task=None):
        self.logger.debug(f'Merging components smaller than {granularity} to "Other" component',
                          current=current, task=task)
        threshold = int(granularity * len(p))
        components = set(p.values())
        comp_sizes = {com: sum([p[node] == com for node in p.keys()]) for com in components}
        comp_to_merge = {com: comp_sizes[com] <= threshold for com in components}
        components_merged = sum(comp_to_merge.values())
        if components_merged > 1:
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
            self.logger.debug(f'No need to reassign components',
                              current=current, task=task)
            pm = p
        return pm, components_merged

    def sort_components(self, pm, components_merged, current=0, task=None):
        self.logger.debug('Sorting components by size descending', current=current, task=task)
        components = set(pm.values())
        comp_sizes = {com: sum([pm[node] == com for node in pm.keys()]) for com in components}

        argsort = lambda seq: sorted(range(len(seq)), key=seq.__getitem__, reverse=True)
        sorted_comps = list(argsort(list(comp_sizes.values())))
        mapping = dict(zip(sorted_comps, range(len(components))))
        self.logger.debug(f'Mapping: {mapping}', current=current, task=task)
        sorted_pm = {node: mapping[c] for node, c in pm.items()}

        if components_merged:
            other = sorted_comps.index(0)
        else:
            other = None

        return sorted_pm, other

    def popular_journals(self, df, n=20, current=0, task=None):
        self.logger.info("Finding popular journals", current=current, task=task)
        journal_stats = df.groupby(['journal', 'comp']).size().reset_index(name='counts')
        # drop papers with undefined subtopic
        journal_stats = journal_stats[journal_stats.comp != -1]

        journal_stats.sort_values(by=['journal', 'counts'], ascending=False, inplace=True)

        journal_stats = journal_stats.groupby('journal').agg(
            {'comp': lambda x: list(x), 'counts': [lambda x: list(x), 'sum']}).reset_index()

        journal_stats.columns = journal_stats.columns.droplevel(level=1)
        journal_stats.columns = ['journal', 'comp', 'counts', 'sum']

        journal_stats = journal_stats.sort_values(by=['sum'], ascending=False)

        if journal_stats['journal'].iloc[0] == '':
            journal_stats.drop(journal_stats.index[0], inplace=True)

        return journal_stats.head(n=n)

    def popular_authors(self, df, n=20, current=0, task=None):
        self.logger.info("Finding popular authors", current=current, task=task)

        author_stats = pd.DataFrame()
        author_stats['author'] = df[df.authors != '']['authors'].apply(lambda authors: authors.split(', '))

        author_stats = author_stats.author.apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame(
            'author').join(df[['id', 'comp']], how='left')

        author_stats = author_stats.groupby(['author', 'comp']).size().reset_index(name='counts')
        # drop papers with undefined subtopic
        author_stats = author_stats[author_stats.comp != -1]

        author_stats.sort_values(by=['author', 'counts'], ascending=False, inplace=True)

        author_stats = author_stats.groupby('author').agg(
            {'comp': lambda x: list(x), 'counts': [lambda x: list(x), 'sum']}).reset_index()

        author_stats.columns = author_stats.columns.droplevel(level=1)
        author_stats.columns = ['author', 'comp', 'counts', 'sum']
        author_stats = author_stats.sort_values(by=['sum'], ascending=False)

        if author_stats['author'].iloc[0] == '':
            author_stats.drop(author_stats.index[0], inplace=True)

        return author_stats.head(n=n)
