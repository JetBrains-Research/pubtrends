import community
import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph
from collections import Counter

from models.prediction.arxiv_loader import ArxivLoader
from .pm_loader import PubmedLoader
from .progress_logger import ProgressLogger
from .ss_loader import SemanticScholarLoader
from .utils import get_subtopic_descriptions, get_tfidf_words, split_df_list, get_most_common_tokens


class KeyPaperAnalyzer:
    SEED = 20190723
    TOTAL_STEPS = 17
    EXPERIMENTAL_STEPS = 2

    def __init__(self, loader, config, test=False):
        self.config = config
        self.experimental = config.experimental
        self.progress = ProgressLogger(KeyPaperAnalyzer.TOTAL_STEPS +
                                       (KeyPaperAnalyzer.EXPERIMENTAL_STEPS if self.experimental else 0))

        self.loader = loader
        loader.set_progress_logger(self.progress)

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
        return self.progress.stream.getvalue()

    def teardown(self):
        self.progress.remove_handler()

    def search_terms(self, query, limit=None, sort=None, task=None):
        # Search articles relevant to the terms
        ids = self.loader.search(query, limit=limit, sort=sort, current=1, task=task)
        if len(ids) == 0:
            raise RuntimeError(f"Nothing found for search query: {query}")
        return ids

    def process_id_list(self, id_list, zoom, current=1, task=None):
        # Zoom and load data about publications with given ids
        ids = id_list
        if len(ids) == 0:
            raise RuntimeError(f"Nothing found in DB for empty ids list")
        for _ in range(zoom):
            if len(ids) > self.config.max_number_to_expand:
                self.progress.info('Too many related papers, stop references expanding',
                                   current=current, task=task)
                break
            ids = self.loader.expand(ids, current=current, task=task)
        return ids

    def analyze_papers(self, ids, query, task=None):
        """:return full log"""
        self.ids = ids
        self.query = query

        # Load data about publications
        self.pub_df = self.loader.load_publications(ids, current=2, task=task)
        if len(self.pub_df) == 0:
            raise RuntimeError(f"Nothing found in DB for ids: {ids}")
        self.n_papers = len(self.pub_df)
        self.pub_types = list(set(self.pub_df['type']))

        # Load data about citations statistics (including outer papers)
        cit_stats_df_from_query = self.loader.load_citation_stats(self.ids, current=3, task=task)
        self.cit_stats_df = self.build_cit_stats_df(cit_stats_df_from_query, self.n_papers, current=4, task=task)
        if len(self.cit_stats_df) == 0:
            raise RuntimeError("No citations of papers were found")
        self.df, self.min_year, self.max_year, self.citation_years = self.merge_citation_stats(self.pub_df,
                                                                                               self.cit_stats_df)
        if len(self.df) == 0:
            raise RuntimeError("Failed to merge publications and citations")

        # Load data about citations within given papers (excluding outer papers)
        # IMPORTANT: cit_df may contain not all the publications for query
        self.cit_df = self.loader.load_citations(self.ids, current=5, task=task)

        # Building inner citations graph for pagerank analysis
        self.G = self.build_citation_graph(self.cit_df, current=6, task=task)

        # Loading data about cocitations
        self.cocit_df = self.loader.load_cocitations(self.ids, current=7, task=task)

        # Building cocitation graph, including all the papers from citations graph
        # IMPORTANT: not all the publications are still covered
        cocit_grouped_df = self.build_cocit_grouped_df(self.cocit_df)
        self.CG = self.build_cocitation_graph(cocit_grouped_df, current=8, task=task, add_citation_edges=True)

        if len(self.CG.nodes()) == 0:
            self.progress.debug("Co-citations graph is empty", current=9, task=task)
            self.progress.info("Not enough papers to process topics analysis", current=9, task=task)
            self.df['comp'] = 0  # Technical value for top authors and papers analysis
            self.df_kwd = pd.DataFrame({'comp': [0], 'kwd': ['']})
        else:
            # Perform subtopic analysis and get subtopic descriptions
            partition, n_components_merged = self.subtopic_analysis(self.CG, current=9, task=task)
            # Get description per component
            kwds = self.subtopic_descriptions(self.df, partition, current=9, task=task)
            # Update non set components
            missing_comps = self.get_missing_components(self.df, partition, kwds, n_components_merged > 0,
                                                        current=10, task=task)
            self.partition, self.comp_other, self.components, self.comp_sizes, sort_order = self.update_components(
                partition, n_components_merged, missing_comps, task
            )
            self.df_kwd = self.update_subtopic_descriptions_df(kwds, sort_order, 20, current=10, task=task)
            # Update df with information for all papers
            self.df = self.merge_col(self.df, self.partition, col='comp')

        # Perform PageRank analysis
        pr = self.pagerank(self.G, current=11, task=task)
        self.df = self.merge_col(self.df, pr, col='pagerank')

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
            self.progress.info('Done', current=19, task=task)
        else:
            self.progress.info('Done', current=17, task=task)

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
        self.progress.debug(f"Loaded citation stats for {len(cit_stats_df)} of {n_papers} papers.\n" +
                            "Others may either have zero citations or be absent in the local database.",
                            current=current, task=task)

        return cit_stats_df

    def build_cocit_grouped_df(self, cocit_df, current=0, task=None):
        self.progress.debug(f'Aggregating co-citations', current=current, task=task)
        cocit_grouped_df = cocit_df.groupby(['cited_1', 'cited_2', 'year']).count().reset_index()
        cocit_grouped_df = cocit_grouped_df.pivot_table(index=['cited_1', 'cited_2'],
                                                        columns=['year'], values=['citing']).reset_index()
        cocit_grouped_df = cocit_grouped_df.replace(np.nan, 0)
        cocit_grouped_df['total'] = cocit_grouped_df.iloc[:, 2:].sum(axis=1)
        cocit_grouped_df = cocit_grouped_df.sort_values(by='total', ascending=False)
        self.progress.debug(f'Filtering top {self.loader.max_number_of_cocitations} of all the co-citations',
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
        self.progress.info(f'Building citation graph', current=current, task=task)
        G = nx.DiGraph()
        for index, row in cit_df.iterrows():
            v, u = row['id_out'], row['id_in']
            G.add_edge(v, u)

        self.progress.info(f'Built citation graph - nodes {len(G.nodes())} edges {len(G.edges())}',
                           current=current, task=task)
        return G

    def build_cocitation_graph(self, cocit_grouped_df, year=None, current=0, task=None, add_citation_edges=False,
                               citation_weight=0.1):
        if year:
            self.progress.info(f'Building co-citations graph for {year} year', current=current, task=task)
        else:
            self.progress.info(f'Building co-citations graph', current=current, task=task)
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

        self.progress.info(f'Built co-citations graph - nodes {len(CG.nodes())} edges {len(CG.edges())}',
                           current=current, task=task)
        return CG

    def subtopic_analysis(self, cocitation_graph, current=0, task=None):
        self.progress.info(f'Extracting subtopics from co-citation graph', current=current, task=task)
        connected_components = nx.number_connected_components(cocitation_graph)
        self.progress.debug(f'Co-citation graph has {connected_components} connected components',
                            current=current, task=task)

        # Graph clustering via Louvain community algorithm
        partition_louvain = community.best_partition(cocitation_graph, random_state=KeyPaperAnalyzer.SEED)
        self.progress.debug(f'Found {len(set(partition_louvain.values()))} components', current=current, task=task)

        # Calculate modularity for partition
        modularity = community.modularity(partition_louvain, cocitation_graph)
        self.progress.debug(f'Graph modularity (possible range is [-1, 1]): {modularity :.3f}',
                            current=current, task=task)

        # Merge small components to 'Other'
        partition, n_components_merged = self.merge_components(partition_louvain)
        return partition, n_components_merged

    def get_missing_components(self, df, partition, kwds, comps_merged, n_words=100, current=0, task=None):
        no_comps = [pid for pid in df['id'] if pid not in partition]
        self.progress.info(f'Assigning topics for all publications', current=current, task=task)
        missing_comps = {}
        for pid in no_comps:
            indx_pid = df['id'] == pid
            pid_mcts = get_most_common_tokens(
                df.loc[indx_pid]['title'] + ' ' + df.loc[indx_pid]['abstract'], 1.0)
            comps_counter = Counter()
            for c, ckwds in kwds.items():
                comps_counter[c] += sum([pid_mcts.get(w, 0) * f for w, f in ckwds[:n_words]])
                if comps_merged and c == 0:  # Other component marker
                    comps_counter[c] += 1e-10
            match_comp = comps_counter.most_common(1)[0][0]  # Get component closest by text
            missing_comps[pid] = match_comp

        self.progress.debug(f'Done assignment missing topics', current=current, task=task)
        return missing_comps

    def update_components(self, partition, n_components_merged, missing_comps, current=0, task=None):
        # Sort components by size
        sort_order, partition_full, comp_other = self.sort_components(
            {**partition, **missing_comps},  # Merge two dictionaries
            n_components_merged
        )
        self.progress.debug(f'Component OTHER: {comp_other}', current=current, task=task)
        components = set(partition_full.values())
        comp_sizes = {c: sum([partition_full[node] == c for node in partition_full.keys()])
                      for c in components}
        for k, v in comp_sizes.items():
            self.progress.debug(f'Cluster {k}: {v} ({int(100 * v / len(partition_full))}%)',
                                current=current, task=task)
        return partition_full, comp_other, components, comp_sizes, sort_order

    def update_subtopic_descriptions_df(self, kwds, sort_order, n_words=20, current=0, task=None):
        # Update descriptions with components reordering
        comps, ckwds = [], []
        for k, v in kwds.items():
            nc = sort_order[k]
            comps.append(nc)
            descr = ','.join([f'{w}:{max(1e-3, f):.3f}' for w, f in v[:n_words]])
            ckwds.append(descr)
            self.progress.debug(f'{nc}: {descr}', current=current, task=task)
        return pd.DataFrame({'comp': comps, 'kwd': ckwds})

    @staticmethod
    def merge_col(df, data, col):
        # Added 'comp' column containing the ID of component
        df_comp = pd.Series(data).reset_index().rename(columns={'index': 'id', 0: col})
        df_comp['id'] = df_comp['id'].astype(str)
        df_merged = pd.merge(df, df_comp,
                             on='id', how='outer')
        df_merged[col] = df_merged[col].fillna(-1).apply(int)

        return df_merged

    def subtopic_descriptions(self, df, partition, n_papers=100, current=0, task=None):
        # Get descriptions for subtopics
        self.progress.debug(f'Computing subtopics descriptions by top {n_papers} cited papers',
                            current=current, task=task)
        most_cited_per_comp = self.get_most_cited_papers_for_comps(df, partition, n=n_papers)
        return get_subtopic_descriptions(df, most_cited_per_comp)

    def find_top_cited_papers(self, df, max_papers=50, threshold=0.1, min_papers=1, current=0, task=None):
        self.progress.info(f'Identifying top cited papers overall', current=current, task=task)
        papers_to_show = max(min(max_papers, round(len(df) * threshold)), min_papers)
        top_cited_df = df.sort_values(by='total',
                                      ascending=False).iloc[:papers_to_show, :]
        top_cited_papers = set(top_cited_df['id'].values)
        return top_cited_papers, top_cited_df

    def find_max_gain_papers(self, df, citation_years, current=0, task=None):
        self.progress.info('Identifying papers with max citation gain for each year', current=current, task=task)
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
        self.progress.info('Identifying papers with max relative citation gain % for each year', current=current,
                           task=task)
        current_sum = pd.Series(np.zeros(len(df), ))
        df_rel = df.loc[:, ['id', 'title', 'authors', 'year']]
        for year in citation_years:
            df_rel[year] = df[year] / (current_sum + (current_sum == 0))
            current_sum += df[year]

        max_rel_gain_data = []
        for year in citation_years:
            max_rel_gain = df_rel[year].max()
            # Ignore less than 1 percent relative gain
            if max_rel_gain >= 0.01:
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
            self.progress.info(f'Year step is too big to analyze evolution of subtopics in {min_year} - {max_year}',
                               current=current, task=task)
            return None, None

        self.progress.info(f'Studying evolution of subtopics in {min_year} - {max_year}',
                           current=current, task=task)

        n_components_merged = {}
        cg = {}

        self.progress.debug(f"Years when subtopics are studied: {', '.join([str(year) for year in year_range])}",
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
                p, n_components_merged[year] = self.merge_components(p)
                evolution_series.append(pd.Series(p))
                years_processed += 1
            else:
                self.progress.debug(f'Total number of papers is less than {min_papers}, stopping.',
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

        self.progress.info(f'Generating descriptions for subtopics during evolution using top {n} cited papers',
                           current=current, task=task)
        evolution_kwds = {}
        for col in evolution_df:
            if col in year_range:
                self.progress.debug(f'Generating TF-IDF descriptions for year {col}',
                                    current=current, task=task)
                if isinstance(col, (int, float)):
                    evolution_df[col] = evolution_df[col].apply(int)
                    comps = evolution_df.groupby(col)['id'].apply(list).to_dict()
                    evolution_kwds[col] = get_tfidf_words(df, comps, query, size=keywords)

        return evolution_kwds

    def merge_components(self, partition, granularity=0.05, current=0, task=None):
        self.progress.debug(f'Merging components smaller than {granularity} to "Other" component',
                            current=current, task=task)
        threshold = int(granularity * len(partition))
        components = set(partition.values())
        comp_sizes = {c: sum([partition[node] == c for node in partition.keys()]) for c in components}
        comp_to_merge = {com: comp_sizes[com] <= threshold for com in components}
        n_components_merged = sum(comp_to_merge.values())
        if n_components_merged > 1:
            self.progress.debug(f'Reassigning components', current=current, task=task)
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
            self.progress.debug(f'Processed {len(set(partition_merged.values()))} components',
                                current=current, task=task)
        else:
            self.progress.debug(f'No need to reassign components',
                                current=current, task=task)
            partition_merged = partition
        return partition_merged, n_components_merged

    def sort_components(self, partition_merged, n_components_merged, current=0, task=None):
        self.progress.debug('Sorting components by size descending', current=current, task=task)
        components = set(partition_merged.values())
        comp_sizes = {c: sum([partition_merged[node] == c for node in partition_merged.keys()]) for c in components}
        # Hack to sort map values by key
        keysort = lambda seq: sorted(range(len(seq)), key=seq.__getitem__, reverse=True)
        sorted_comps = list(keysort(list(comp_sizes.values())))
        sort_order = dict(zip(sorted_comps, range(len(components))))
        self.progress.debug(f'Components reordering by size: {sort_order}', current=current, task=task)
        sorted_partition = {node: sort_order[c] for node, c in partition_merged.items()}

        if n_components_merged > 0:
            comp_other = sorted_comps.index(0)  # Other component is 0!
        else:
            comp_other = None
        self.progress.debug(f'Other component: {comp_other}', current=current, task=task)
        return sort_order, sorted_partition, comp_other

    def popular_journals(self, df, n=50, current=0, task=None):
        self.progress.info("Finding popular journals", current=current, task=task)
        journal_stats = df.groupby(['journal', 'comp']).size().reset_index(name='counts')
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
        self.progress.info("Finding popular authors", current=current, task=task)

        author_stats = df[['authors', 'comp']].copy()
        author_stats['authors'].replace({'': np.nan, -1: np.nan}, inplace=True)
        author_stats.dropna(subset=['authors'], inplace=True)

        author_stats = split_df_list(author_stats, target_column='authors', separator=', ')
        author_stats.rename(columns={'authors': 'author'}, inplace=True)

        author_stats = author_stats.groupby(['author', 'comp']).size().reset_index(name='counts')
        author_stats.sort_values(by=['author', 'counts'], ascending=False, inplace=True)

        author_stats = author_stats.groupby('author').agg(
            {'comp': lambda x: list(x), 'counts': [lambda x: list(x), 'sum']}).reset_index()

        author_stats.columns = author_stats.columns.droplevel(level=1)
        author_stats.columns = ['author', 'comp', 'counts', 'sum']
        author_stats = author_stats.sort_values(by=['sum'], ascending=False)

        return author_stats.head(n=n)

    @staticmethod
    def get_most_cited_papers_for_comps(df, partition, n):
        pdf = pd.DataFrame(partition.items(), columns=['id', 'comp'])
        ids_comp_df = pd.merge(left=df[['id', 'total']], left_on='id',
                               right=pdf, right_on='id', how='inner')
        ids = ids_comp_df.sort_values(by='total', ascending=False).groupby('comp')['id']
        return ids.apply(list).apply(lambda x: x[:n]).to_dict()

    def dump(self):
        """
        Dump valuable fields of KeyPaperAnalyzer to JSON-serializable dict. Use 'load' to restore analyzer.
        """
        return {
            'cg': json_graph.node_link_data(self.CG),
            'df': self.df.to_json(),
            'df_kwd': self.df_kwd.to_json(),
            'g': json_graph.node_link_data(self.G),
            'top_cited_papers': list(self.top_cited_papers),
            'max_gain_papers': list(self.max_gain_papers),
            'max_rel_gain_papers': list(self.max_rel_gain_papers),
        }

    @staticmethod
    def load(fields):
        """
        Load valuable fields of KeyPaperAnalyzer from JSON-serializable dict. Use 'dump' to dump analyzer.
        """
        # Restore main dataframe
        df = pd.read_json(fields['df'])
        df['id'] = df['id'].apply(str)

        mapping = {}
        for col in df.columns:
            try:
                mapping[col] = int(col)
            except ValueError:
                mapping[col] = col
        df = df.rename(columns=mapping)

        # Restore subtopic descriptions
        df_kwd = pd.read_json(fields['df_kwd'])

        # Extra filter is applied to overcome split behaviour problem: split('') = [''] problem
        df_kwd['kwd'] = [kwd.split(',') if kwd != '' else [] for kwd in df_kwd['kwd']]
        df_kwd['kwd'] = df_kwd['kwd'].apply(lambda x: [el.split(':') for el in x])
        df_kwd['kwd'] = df_kwd['kwd'].apply(lambda x: [(el[0], float(el[1])) for el in x])

        # Restore citation and co-citation graphs
        CG = json_graph.node_link_graph(fields['cg'])
        G = json_graph.node_link_graph(fields['g'])

        top_cited_papers = set(fields['top_cited_papers'])
        max_gain_papers = set(fields['max_gain_papers'])
        max_rel_gain_papers = set(fields['max_rel_gain_papers'])

        return {
            'cg': CG,
            'df': df,
            'df_kwd': df_kwd,
            'g': G,
            'top_cited_papers': top_cited_papers,
            'max_gain_papers': max_gain_papers,
            'max_rel_gain_papers': max_rel_gain_papers
        }

    def init(self, fields):
        loaded = KeyPaperAnalyzer.load(fields)
        self.df, self.df_kwd, self.G, self.CG = loaded['df'], loaded['df_kwd'], loaded['g'], loaded['cg']
        self.top_cited_papers = loaded['top_cited_papers']
        self.max_gain_papers = loaded['max_gain_papers']
        self.max_rel_gain_papers = loaded['max_rel_gain_papers']

    def pagerank(self, G, current=0, task=None):
        self.progress.info('Performing PageRank analysis', current=current, task=task)
        # Apply PageRank algorithm with damping factor of 0.5
        return nx.pagerank(G, alpha=0.5, tol=1e-9)
