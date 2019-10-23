import community
import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph

from models.prediction.arxiv_loader import ArxivLoader
from .pm_loader import PubmedLoader
from .progress_logger import ProgressLogger
from .ss_loader import SemanticScholarLoader
from .utils import get_subtopic_descriptions, get_tfidf_words, split_df_list


class KeyPaperAnalyzer:
    SEED = 20190723
    TOTAL_STEPS = 16
    EXPERIMENTAL_STEPS = 2

    def __init__(self, loader, config, test=False):
        self.config = config
        self.experimental = config.experimental
        self.logger = ProgressLogger(KeyPaperAnalyzer.TOTAL_STEPS +
                                     (KeyPaperAnalyzer.EXPERIMENTAL_STEPS if self.experimental else 0))

        self.loader = loader
        loader.set_logger(self.logger)

        # Determine source to provide correct URLs to articles
        if isinstance(self.loader, PubmedLoader):
            self.source = 'Pubmed'
        elif isinstance(self.loader, SemanticScholarLoader):
            self.source = 'Semantic Scholar'
        elif isinstance(self.loader, ArxivLoader):
            self.source = 'arxiv'
        elif not test:
            raise TypeError("loader should be either PubmedLoader or SemanticScholarLoader")

    def log(self):
        return self.logger.stream.getvalue()

    def teardown(self):
        self.logger.remove_handler()

    def search_terms(self, query, limit=None, sort=None, task=None):
        # Search articles relevant to the terms
        ids = self.loader.search(query, limit=limit, sort=sort, current=1, task=task)
        if len(ids) == 0:
            raise RuntimeError(f"Nothing found in DB for search query: {query}")
        # Load data about publications
        pub_df = self.loader.load_publications(ids, current=2, task=task)
        if len(pub_df) == 0:
            raise RuntimeError(f"Nothing found in DB for ids: {ids}")
        return ids, pub_df

    def process_id_list(self, id_list, zoom, task=None):
        # Load data about publications with given ids
        ids = id_list
        if len(ids) == 0:
            raise RuntimeError(f"Nothing found in DB for empty ids list")
        for _ in range(zoom):
            ids = self.loader.expand(ids, current=1, task=task)
        # Load data about publications
        pub_df = self.loader.load_publications(ids, current=2, task=task)
        if len(pub_df) == 0:
            raise RuntimeError(f"Nothing found in DB for ids: {ids}")
        return ids, pub_df

    def analyze_papers(self, ids, pub_df, query, task=None):
        """:return full log"""
        self.ids = ids
        self.pub_df = pub_df
        self.query = query
        self.n_papers = len(self.ids)
        self.pub_types = list(set(self.pub_df['type']))
        cit_stats_df_from_query = self.loader.load_citation_stats(self.ids, current=3, task=task)
        self.cit_stats_df = self.build_cit_stats_df(cit_stats_df_from_query, self.n_papers, current=4, task=task)
        if len(self.cit_stats_df) == 0:
            raise RuntimeError("No citations of papers were found")
        self.df, self.min_year, self.max_year, self.citation_years = self.merge_citation_stats(self.pub_df,
                                                                                               self.cit_stats_df)
        if len(self.df) == 0:
            raise RuntimeError("Failed to merge publications and citations")
        self.cit_df = self.loader.load_citations(self.ids, current=5, task=task)
        self.G = self.build_citation_graph(self.cit_df, current=6, task=task)
        self.cocit_df = self.loader.load_cocitations(self.ids, current=7, task=task)
        cocit_grouped_df = self.build_cocit_grouped_df(self.cocit_df)
        self.CG = self.build_cocitation_graph(cocit_grouped_df, current=8, task=task, add_citation_edges=True)
        if len(self.CG.nodes()) == 0:
            raise RuntimeError("Failed to build co-citations graph")
        # Perform subtopic analysis and get subtopic descriptions
        self.components, self.comp_other, self.partition, self.comp_sizes = self.subtopic_analysis(
            self.CG, current=9, task=task
        )
        self.df = self.merge_col(self.df, self.partition, col='comp')
        self.df_kwd = self.subtopic_descriptions(self.df, current=10, task=task)
        # Perform PageRank analysis
        self.pr = self.pagerank(self.G, current=11, task=task)
        self.df = self.merge_col(self.df, self.pr, col='pagerank')
        # Find interesting papers
        self.top_cited_papers, self.top_cited_df = self.find_top_cited_papers(self.df, current=12, task=task)
        self.max_gain_papers, self.max_gain_df = self.find_max_gain_papers(self.df, self.citation_years,
                                                                           current=13, task=task)
        self.max_rel_gain_papers, self.max_rel_gain_df = self.find_max_relative_gain_papers(
            self.df, self.citation_years, current=13, task=task
        )
        # Find top journals
        self.journal_stats = self.popular_journals(self.df, current=15, task=task)
        # Find top authors
        self.author_stats = self.popular_authors(self.df, current=16, task=task)
        # Experimental features, can be turned off in 'config.properties'
        if self.experimental:
            # Perform subtopic evolution analysis and get subtopic descriptions
            self.evolution_df, self.evolution_year_range = self.subtopic_evolution_analysis(self.cocit_df,
                                                                                            current=17, task=task)
            self.evolution_kwds = self.subtopic_evolution_descriptions(
                self.df, self.evolution_df, self.evolution_year_range, self.query, current=18, task=task
            )

    def build_cit_stats_df(self, cit_stats_df_from_query, n_papers, current=None, task=None):
        # Get citation stats with columns 'id', year_1, ..., year_N and fill NaN with 0
        cit_stats_df = cit_stats_df_from_query.pivot(index='id', columns='year',
                                                     values='count').reset_index().fillna(0)

        # Fix column names from float 'YYYY.0' to int 'YYYY'
        mapper = {}
        for col in cit_stats_df.columns:
            if col != 'id':
                mapper[col] = int(col)
        cit_stats_df = cit_stats_df.rename(mapper)

        cit_stats_df['total'] = cit_stats_df.iloc[:, 1:].sum(axis=1)
        cit_stats_df = cit_stats_df.sort_values(by='total', ascending=False)
        self.logger.debug(f"Loaded citation stats for {len(cit_stats_df)} of {n_papers} papers.\n" +
                          "Others may either have zero citations or be absent in the local database.",
                          current=current, task=task)

        return cit_stats_df

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
        df.authors = df.authors.fillna('')

        # Publication and citation year range
        citation_years = [int(col) for col in list(df.columns) if isinstance(col, (int, float))]
        min_year, max_year = int(df['year'].min()), int(df['year'].max())

        return df, min_year, max_year, citation_years

    def build_citation_graph(self, cit_df, current=0, task=None):
        self.logger.info(f'Building citation graph', current=current, task=task)
        G = nx.DiGraph()
        for index, row in cit_df.iterrows():
            v, u = row['id_out'], row['id_in']
            G.add_edge(v, u)

        self.logger.info(f'Built citation graph - nodes {len(G.nodes())} edges {len(G.edges())}',
                         current=current, task=task)
        return G

    def build_cocitation_graph(self, cocit_grouped_df, year=None, current=0, task=None, add_citation_edges=False,
                               citation_weight=0.3):
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

        if add_citation_edges:
            for u, v in self.G.edges:
                if CG.has_edge(u, v):
                    CG.add_edge(u, v, weight=CG[u][v]['weight'] + citation_weight)
                else:
                    CG.add_edge(u, v, weight=citation_weight)

        self.logger.info(f'Built co-citations graph - nodes {len(CG.nodes())} edges {len(CG.edges())}',
                         current=current, task=task)
        return CG

    def subtopic_analysis(self, cocitation_graph, current=0, task=None):
        self.logger.info(f'Extracting subtopics from co-citation graph', current=current, task=task)
        connected_components = nx.number_connected_components(cocitation_graph)
        self.logger.debug(f'Co-citation graph has {connected_components} connected components',
                          current=current, task=task)

        # Graph clustering via Louvain community algorithm
        partition = community.best_partition(cocitation_graph, random_state=KeyPaperAnalyzer.SEED)
        self.logger.debug(f'Found {len(set(partition.values()))} components', current=current, task=task)

        # Calculate modularity for partition
        modularity = community.modularity(partition, cocitation_graph)
        self.logger.debug(f'Graph modularity (possible range is [-1, 1]): {modularity :.3f}',
                          current=current, task=task)

        # Merge small components to 'Other'
        partition_merged, components_merged = self.merge_components(partition)
        partition_merged, comp_other = self.sort_components(partition_merged, components_merged)
        components = set(partition_merged.values())
        comp_sizes = {c: sum([partition_merged[node] == c for node in partition_merged.keys()]) for c in components}
        for k, v in comp_sizes.items():
            self.logger.debug(f'Cluster {k}: {v} ({int(100 * v / len(partition_merged))}%)',
                              current=current, task=task)

        return components, comp_other, partition_merged, comp_sizes

    @staticmethod
    def merge_col(df, data, col):
        # Added 'comp' column containing the ID of component
        df_comp = pd.Series(data).reset_index().rename(columns={'index': 'id', 0: col})
        df_comp['id'] = df_comp['id'].astype(str)
        df_merged = pd.merge(df, df_comp,
                             on='id', how='outer')
        df_merged[col] = df_merged[col].fillna(-1).apply(int)

        return df_merged

    def subtopic_descriptions(self, df, n=200, current=0, task=None):
        # Get descriptions for subtopics
        self.logger.debug(f'Getting descriptions for subtopics using top {n} cited papers',
                          current=current, task=task)
        comps = self.get_most_cited_papers_for_comps(df, n=n)
        kwds = get_subtopic_descriptions(df, comps)
        for k, v in kwds.items():
            self.logger.debug(f'{k}: {v}', current=current, task=task)
        df_kwd = pd.Series(kwds).reset_index()
        df_kwd = df_kwd.rename(columns={'index': 'comp', 0: 'kwd'})
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
        year_range = list(np.arange(max_year, min_year - 1, step=-step).astype(int))

        # Cannot analyze evolution
        if len(year_range) < 2:
            self.logger.info(f'Year step is too big to analyze evovution of subtopics in {min_year} - {max_year}',
                             current=current, task=task)
            return None, None

        self.logger.info(f'Studying evolution of subtopics in {min_year} - {max_year}',
                         current=current, task=task)

        components_merged = {}
        cg = {}

        self.logger.debug(f"Years when subtopics are studied: {', '.join([str(year) for year in year_range])}",
                          current=current, task=task)

        # Use results of subtopic analysis for current year, perform analysis for other years
        years_processed = 1
        evolution_series = [pd.Series(self.partition)]
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

    def subtopic_evolution_descriptions(self, df, evolution_df, year_range, query,
                                        n=200, keywords=15, current=0, task=None):
        # Subtopic evolution failed, no need to generate keywords
        if evolution_df is None or not year_range:
            return None

        self.logger.info(f'Generating descriptions for subtopics during evolution using top {n} cited papers',
                         current=current, task=task)
        evolution_kwds = {}
        for col in evolution_df:
            if col in year_range:
                self.logger.debug(f'Generating TF-IDF descriptions for year {col}',
                                  current=current, task=task)
                if isinstance(col, (int, float)):
                    evolution_df[col] = evolution_df[col].apply(int)
                    comps = evolution_df.groupby(col)['id'].apply(list).to_dict()
                    evolution_kwds[col] = get_tfidf_words(df, comps, query, size=keywords)

        return evolution_kwds

    def merge_components(self, partition, granularity=0.05, current=0, task=None):
        self.logger.debug(f'Merging components smaller than {granularity} to "Other" component',
                          current=current, task=task)
        threshold = int(granularity * len(partition))
        components = set(partition.values())
        comp_sizes = {c: sum([partition[node] == c for node in partition.keys()]) for c in components}
        comp_to_merge = {com: comp_sizes[com] <= threshold for com in components}
        components_merged = sum(comp_to_merge.values())
        if components_merged > 1:
            self.logger.debug(f'Reassigning components', current=current, task=task)
            partition_merged = {}
            new_comps = {}
            ci = 1  # Other component is 0.
            for k, v in partition.items():
                if comp_sizes[v] <= threshold:
                    partition_merged[k] = 0  # Other
                    continue
                if v not in new_comps:
                    new_comps[v] = ci
                    ci += 1
                partition_merged[k] = new_comps[v]
            self.logger.debug(f'Processed {len(set(partition_merged.values()))} components',
                              current=current, task=task)
        else:
            self.logger.debug(f'No need to reassign components',
                              current=current, task=task)
            partition_merged = partition
        return partition_merged, components_merged

    def sort_components(self, partition_merged, components_merged, current=0, task=None):
        self.logger.debug('Sorting components by size descending', current=current, task=task)
        components = set(partition_merged.values())
        comp_sizes = {c: sum([partition_merged[node] == c for node in partition_merged.keys()]) for c in components}

        argsort = lambda seq: sorted(range(len(seq)), key=seq.__getitem__, reverse=True)
        sorted_comps = list(argsort(list(comp_sizes.values())))
        mapping = dict(zip(sorted_comps, range(len(components))))
        self.logger.debug(f'Mapping: {mapping}', current=current, task=task)
        sorted_partition_merged = {node: mapping[c] for node, c in partition_merged.items()}

        if components_merged:
            other = sorted_comps.index(0)
        else:
            other = None

        return sorted_partition_merged, other

    def popular_journals(self, df, n=50, current=0, task=None):
        self.logger.info("Finding popular journals", current=current, task=task)
        journal_stats = df.groupby(['journal', 'comp']).size().reset_index(name='counts')
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

        return journal_stats.head(n=n)

    def popular_authors(self, df, n=50, current=0, task=None):
        self.logger.info("Finding popular authors", current=current, task=task)

        author_stats = df[['authors', 'comp']].copy()
        author_stats['authors'].replace({'': np.nan, -1: np.nan}, inplace=True)
        author_stats.dropna(subset=['authors'], inplace=True)

        author_stats = split_df_list(author_stats, target_column='authors', separator=', ')
        author_stats.rename(columns={'authors': 'author'}, inplace=True)

        author_stats = author_stats.groupby(['author', 'comp']).size().reset_index(name='counts')
        # drop papers with undefined subtopic
        author_stats = author_stats[author_stats.comp != -1]

        author_stats.sort_values(by=['author', 'counts'], ascending=False, inplace=True)

        author_stats = author_stats.groupby('author').agg(
            {'comp': lambda x: list(x), 'counts': [lambda x: list(x), 'sum']}).reset_index()

        author_stats.columns = author_stats.columns.droplevel(level=1)
        author_stats.columns = ['author', 'comp', 'counts', 'sum']
        author_stats = author_stats.sort_values(by=['sum'], ascending=False)

        return author_stats.head(n=n)

    @staticmethod
    def get_most_cited_papers_for_comps(df, n):
        ids = df[df['comp'] >= 0].sort_values(by='total', ascending=False).groupby('comp')['id']
        return ids.apply(list).apply(lambda x: x[:n]).to_dict()

    def dump(self):
        """
        Dump valuable fields of KeyPaperAnalyzer to JSON-serializable dict. Use 'load' to restore analyzer.
        """
        return {'cg': json_graph.node_link_data(self.CG),
                'df': self.df.to_json(),
                'df_kwd': self.df_kwd.to_json(),
                'g': json_graph.node_link_data(self.G)}

    def load(self, fields):
        """
        Load valuable fields of KeyPaperAnalyzer from JSON-serializable dict. Use 'dump' to dump analyzer.
        """
        # Restore main dataframe
        self.df = pd.read_json(fields['df'])
        self.df['id'] = self.df['id'].apply(str)

        mapping = {}
        for col in self.df.columns:
            try:
                mapping[col] = int(col)
            except ValueError:
                mapping[col] = col
        self.df = self.df.rename(columns=mapping)

        # Restore subtopic descriptions
        self.df_kwd = pd.read_json(fields['df_kwd'])
        self.df_kwd['kwd'] = self.df_kwd['kwd'].str.split(',').apply(list)
        self.df_kwd['kwd'] = self.df_kwd['kwd'].apply(lambda x: [el.split(':') for el in x])
        self.df_kwd['kwd'] = self.df_kwd['kwd'].apply(lambda x: [(el[0], float(el[1])) for el in x])

        # Restore citation and co-citation graphs
        self.CG = json_graph.node_link_graph(fields['cg'])
        self.G = json_graph.node_link_graph(fields['g'])

    def pagerank(self, G, current=0, task=None):
        self.logger.info('Performing PageRank analysis', current=current, task=task)
        # Apply PageRank algorithm with damping factor of 0.5
        return nx.pagerank(G, alpha=0.5, tol=1e-9)
