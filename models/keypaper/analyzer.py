import logging
from collections import Counter

import community
import networkx as nx
import numpy as np
import pandas as pd
from itertools import product as cart_product
from networkx.readwrite import json_graph
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from models.keypaper.pm_loader import PubmedLoader
from models.keypaper.progress import Progress
from models.keypaper.ss_loader import SemanticScholarLoader
from models.keypaper.utils import split_df_list, get_frequent_tokens, get_topics_description, tokenize
from models.prediction.arxiv_loader import ArxivLoader

logger = logging.getLogger(__name__)


class KeyPaperAnalyzer:
    SEED = 20190723

    TOP_CITED_PAPERS = 50
    TOP_CITED_PAPERS_FRACTION = 0.1

    SIMILARITY_COCITATION = 10
    SIMILARITY_BIBLIOGRAPHIC_COUPLING = 1
    SIMILARITY_CITATION = 0.1

    TOPIC_GRANULARITY = 0.05
    TOPIC_PAPERS = 50
    TOPIC_WORDS = 20
    TEXT_WORDS = 1000

    TOP_JOURNALS = 50
    TOP_AUTHORS = 50

    def __init__(self, loader, config, test=False):
        self.config = config
        self.progress = Progress(self.total_steps())

        self.loader = loader
        loader.set_progress(self.progress)

        # Determine source to provide correct URLs to articles
        if isinstance(self.loader, PubmedLoader):
            self.source = 'Pubmed'
        elif isinstance(self.loader, SemanticScholarLoader):
            self.source = 'Semantic Scholar'
        elif isinstance(self.loader, ArxivLoader):
            self.source = 'Arxiv'
        elif not test:
            raise TypeError(f'Unknown loader {self.loader}')

    def total_steps(self):
        return 19  # 18 + 1 for visualization

    def teardown(self):
        self.progress.remove_handler()

    def search_terms(self, query, limit=None, sort=None, task=None):
        # Search articles relevant to the terms
        if len(query) == 0:
            raise Exception(f'Empty search string, please use search terms or '
                            f'all the query wrapped in "" for phrasal search')
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
        self.ids = set(self.pub_df['id'])  # Limit ids to existing papers only!
        self.n_papers = len(self.ids)
        self.pub_types = list(set(self.pub_df['type']))

        # Load data about citations statistics (including outer papers)
        cit_stats_df_from_query = self.loader.load_citation_stats(self.ids, current=3, task=task)
        self.cit_stats_df = self.build_cit_stats_df(cit_stats_df_from_query, self.n_papers, current=4, task=task)
        if len(self.cit_stats_df) == 0:
            raise RuntimeError("No citations of papers were found")
        self.df, self.min_year, self.max_year, self.citation_years = self.merge_citation_stats(
            self.pub_df, self.cit_stats_df)
        if len(self.df) == 0:
            raise RuntimeError("Failed to merge publications and citations")

        # Load data about citations within given papers (excluding outer papers)
        # IMPORTANT: cit_df may contain not all the publications for query
        self.cit_df = self.loader.load_citations(self.ids, current=5, task=task)

        # Building inner citations graph for pagerank analysis
        self.citations_graph = self.build_citation_graph(self.cit_df, current=6, task=task)

        # Loading data about cocitations
        self.cocit_df = self.loader.load_cocitations(self.ids, current=7, task=task)
        cocit_grouped_df = self.build_cocit_grouped_df(self.cocit_df)

        # Loading data about bibliographic coupling
        self.bibliographic_coupling_df = self.loader.load_bibliographic_coupling(self.ids, current=8, task=task)

        # IMPORTANT: not all the publications might be still covered
        self.similarity_graph = self.build_similarity_graph(
            self.citations_graph, cocit_grouped_df, self.bibliographic_coupling_df, current=9, task=task
        )

        if len(self.similarity_graph.nodes()) == 0:
            logger.debug("Paper similarity graph is empty")
            self.progress.info("Not enough papers to process topics analysis", current=10, task=task)
            self.df['comp'] = 0  # Technical value for top authors and papers analysis
            self.df_kwd = pd.DataFrame({'comp': [0], 'kwd': ['']})
            self.structure_graph = nx.Graph()
        else:
            # Perform subtopic analysis and get subtopic descriptions
            partition, n_components_merged = self.subtopic_analysis(self.similarity_graph, current=10, task=task)
            # Get descriptions for subtopics
            logger.debug(f'Computing subtopics descriptions by top cited papers')
            most_cited_per_comp = self.get_most_cited_papers_for_comps(self.df, partition)
            # Compute topics TF-IDF metrics
            tfidf_per_comp = get_topics_description(self.df, most_cited_per_comp, query, self.TEXT_WORDS)
            # Update papers non assigned with components
            missing_comps = self.get_missing_components(self.df, partition, tfidf_per_comp, n_components_merged > 0,
                                                        current=11, task=task)
            self.partition, self.comp_other, self.components, self.comp_sizes, sort_order = self.update_components(
                partition, n_components_merged, missing_comps, task
            )
            self.df = self.merge_col(self.df, self.partition, col='comp')
            # Prepare information for word cloud
            kwds = [(sort_order[comp], ','.join([f'{t}:{max(1e-3, v):.3f}' for t, v in vs[:self.TOPIC_WORDS]]))
                    for comp, vs in tfidf_per_comp.items()]
            self.df_kwd = pd.DataFrame(kwds, columns=['comp', 'kwd'])
            logger.debug(f'Components description\n{self.df_kwd["kwd"]}')
            # Build structure graph
            self.structure_graph = self.build_structure_graph(self.df, self.similarity_graph,
                                                              current=12, task=task)

        # Perform PageRank analysis
        pr = self.pagerank(self.citations_graph, current=13, task=task)
        self.df = self.merge_col(self.df, pr, col='pagerank')

        # Find interesting papers
        self.top_cited_papers, self.top_cited_df = self.find_top_cited_papers(self.df, current=14, task=task)
        self.max_gain_papers, self.max_gain_df = self.find_max_gain_papers(self.df, self.citation_years,
                                                                           current=15, task=task)
        self.max_rel_gain_papers, self.max_rel_gain_df = self.find_max_relative_gain_papers(
            self.df, self.citation_years, current=16, task=task
        )

        # Find top journals
        self.journal_stats = self.popular_journals(self.df, current=17, task=task)
        # Find top authors
        self.author_stats = self.popular_authors(self.df, current=18, task=task)

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
        logger.debug(f'Loaded citation stats for {len(cit_stats_df)} of {n_papers} papers')

        return cit_stats_df

    def build_cocit_grouped_df(self, cocit_df, current=0, task=None):
        logger.debug(f'Aggregating co-citations')
        cocit_grouped_df = cocit_df.groupby(['cited_1', 'cited_2', 'year']).count().reset_index()
        cocit_grouped_df = cocit_grouped_df.pivot_table(index=['cited_1', 'cited_2'],
                                                        columns=['year'], values=['citing']).reset_index()
        cocit_grouped_df = cocit_grouped_df.replace(np.nan, 0)
        cocit_grouped_df['total'] = cocit_grouped_df.iloc[:, 2:].sum(axis=1)
        cocit_grouped_df = cocit_grouped_df.sort_values(by='total', ascending=False)

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

        self.progress.info(f'Built citation graph - {len(G.nodes())} nodes and {len(G.edges())} edges',
                           current=current, task=task)
        return G

    def build_similarity_graph(self, citations_graph, cocit_df, bibliographic_coupling_df, current=0, task=None):
        """
        Graph is build using three citation-based method:
        bibliographic coupling (BC), co-citation (CC) and direct citation (DC).

        See paper: Which type of citation analysis generates the most accurate taxonomy of
        scientific and technical knowledge? (https://arxiv.org/pdf/1511.05078.pdf)
        ...bibliographic coupling (BC) was the most accurate,  followed by co-citation (CC).
        Direct citation (DC) was a distant third among the three...
        """
        self.progress.info(f'Building papers similarity graph', current=current, task=task)

        result = nx.Graph()
        # NOTE: we use nodes id as String to avoid problems str keys in jsonify
        # during graph visualization
        for el in cocit_df[['cited_1', 'cited_2', 'total']].values:
            start, end, cocitation = str(el[0]), str(el[1]), float(el[2])
            result.add_edge(start, end, cocitation=cocitation)

        for el in bibliographic_coupling_df[['citing_1', 'citing_2', 'total']].values:
            start, end, bibcoupling = str(el[0]), str(el[1]), float(el[2])
            if result.has_edge(start, end):
                result[start][end]['bibcoupling'] = bibcoupling
            else:
                result.add_edge(start, end, bibcoupling=bibcoupling)

        for u, v in citations_graph.edges:
            if result.has_edge(u, v):
                result[u][v]['citation'] = 1
            else:
                result.add_edge(u, v, citation=1)

        self.progress.info(f'Built papers similarity graph - '
                           f'{len(result.nodes())} nodes and {len(result.edges())} edges',
                           current=current, task=task)
        return result

    def subtopic_analysis(self, similarity_graph, current=0, task=None):
        self.progress.info(f'Extracting subtopics from paper similarity graph', current=current, task=task)
        connected_components = nx.number_connected_components(similarity_graph)
        logger.debug(f'Similarity graph has {connected_components} connected components')

        # Compute aggregated weight
        for _, _, d in similarity_graph.edges(data=True):
            d['similarity'] = self.SIMILARITY_COCITATION * d.get('cocitation', 0) + \
                self.SIMILARITY_BIBLIOGRAPHIC_COUPLING * d.get('bibcoupling', 0) + \
                self.SIMILARITY_CITATION * d.get('citation', 0)

        # Graph clustering via Louvain community algorithm
        partition_louvain = community.best_partition(similarity_graph, random_state=KeyPaperAnalyzer.SEED)
        logger.debug(f'Found {len(set(partition_louvain.values()))} components')

        # Calculate modularity for partition
        modularity = community.modularity(partition_louvain, similarity_graph)
        logger.debug(f'Graph modularity (possible range is [-1, 1]): {modularity :.3f}')

        # Merge small components to 'Other'
        partition, n_components_merged = self.merge_components(partition_louvain)
        partition_counter = Counter(partition.values())
        logger.debug(f'Merged components sizes {partition_counter}')
        return partition, n_components_merged

    def get_missing_components(self, df, partition, tfidf_per_comp, comps_merged, n_words=TEXT_WORDS,
                               current=0, task=None):
        no_comps = [pid for pid in df['id'] if pid not in partition]
        self.progress.info(f'Assigning topics for publications based on text similarity', current=current, task=task)
        missing_comps = {}
        for pid in no_comps:
            df_pid = df.loc[df['id'] == pid]
            # Get all the tokens for paper
            pid_frequent_tokens = get_frequent_tokens(df_pid, self.query, 1.0)
            comps_counter = Counter()
            for comp, comp_tfidf in tfidf_per_comp.items():
                # Compute cosine distance between frequent words in document and TF-IDFs for each component
                cosine_distance = cosine(
                    [pid_frequent_tokens.get(t[0], 0) for t in comp_tfidf[:n_words]],
                    [t[1] for t in comp_tfidf[:n_words]]
                )
                comps_counter[comp] += 1 - cosine_distance  # The bigger similarity the better
                if comps_merged and comp == 0:  # Other component marker
                    comps_counter[comp] += 1e-10
            # Get component closest by text (max dot product)
            missing_comps[pid] = comps_counter.most_common(1)[0][0]
        missing_comps_counter = Counter(missing_comps.values())
        self.progress.info(f'Assigned topics for {len(no_comps)} papers', current=current, task=task)
        logger.debug(f'Assigned missing topics {missing_comps_counter}')
        return missing_comps

    def update_components(self, partition, n_components_merged, missing_comps, current=0, task=None):
        # Sort components by size
        sort_order, partition_full, comp_other = self.sort_components(
            {**partition, **missing_comps},  # Merge two dictionaries
            n_components_merged
        )
        components = set(partition_full.values())
        comp_sizes = {c: sum([partition_full[node] == c for node in partition_full.keys()])
                      for c in components}
        for k, v in comp_sizes.items():
            logger.debug(f'Cluster {k}: {v} ({int(100 * v / len(partition_full))}%)')
        return partition_full, comp_other, components, comp_sizes, sort_order

    def build_structure_graph(self, df, similarity_graph, n_words=TEXT_WORDS, n_gram=1, current=0, task=None):
        self.progress.info('Building structure graph', current=current, task=task)
        graph = similarity_graph.copy()
        for (u, v, w) in similarity_graph.edges.data('similarity'):
            graph[u][v]['distance'] = -w

        logger.debug('Compute global TF-IDF')
        corpus = [f'{t} {a}' for t, a in zip(df['title'], df['abstract'])]
        comps = set(df['comp'])
        vectorizer = CountVectorizer(max_df=0.8, ngram_range=(1, n_gram),
                                     max_features=n_words * len(comps),
                                     tokenizer=lambda t: tokenize(t))
        counts = vectorizer.fit_transform(corpus)
        tfidf_transformer = TfidfTransformer()
        tfidf = tfidf_transformer.fit_transform(counts)
        cos_distances = 1 - cosine_similarity(tfidf)
        idx_map = {pid: i for i, pid in enumerate(df['id'])}

        logger.debug('Adding cosine based edges to graph between out-of-graph edges and components')
        for comp in comps:
            df_comp = df[df['comp'] == comp]
            cit_ids = [pid for pid in df_comp['id'] if similarity_graph.has_node(pid)]
            text_ids = [pid for pid in df_comp['id'] if not similarity_graph.has_node(pid)]

            for cid, tid in cart_product(cit_ids, text_ids):
                cos_distance = cos_distances[idx_map[cid], idx_map[tid]]
                if np.isfinite(cos_distance):
                    graph.add_edge(cid, tid, cos_similarity=1 - cos_distance, distance=cos_distance)
            for i, tid1 in enumerate(text_ids):
                for j in range(i + 1, len(text_ids)):
                    tid2 = text_ids[j]
                    cos_distance = cos_distances[idx_map[tid1], idx_map[tid2]]
                    if np.isfinite(cos_distance):
                        graph.add_edge(tid1, tid2, cos_similarity=1 - cos_distance, distance=cos_distance)

        logger.debug('Computing min spanning tree')
        mst = nx.minimum_spanning_tree(graph, 'distance')

        logger.info('Cleanup')
        for _, _, d in mst.edges(data=True):
            del d['distance']

        return mst

    @staticmethod
    def merge_col(df, data, col):
        # Added 'comp' column containing the ID of component
        df_comp = pd.Series(data).reset_index().rename(columns={'index': 'id', 0: col})
        df_comp['id'] = df_comp['id'].astype(str)
        df_merged = pd.merge(df, df_comp,
                             on='id', how='outer')
        df_merged[col] = df_merged[col].fillna(-1).apply(int)

        return df_merged

    def find_top_cited_papers(self, df, n_papers=TOP_CITED_PAPERS, threshold=TOP_CITED_PAPERS_FRACTION,
                              current=0, task=None):
        self.progress.info(f'Identifying top cited papers overall', current=current, task=task)
        papers_to_show = max(min(n_papers, round(len(df) * threshold)), 1)
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

    def merge_components(self, partition, granularity=TOPIC_GRANULARITY, current=0, task=None):
        logger.debug(f'Merging components smaller than {granularity} to "Other" component')
        threshold = int(granularity * len(partition))
        components = set(partition.values())
        comp_sizes = {c: sum([partition[node] == c for node in partition.keys()]) for c in components}
        comp_to_merge = {com: comp_sizes[com] <= threshold for com in components}
        n_components_merged = sum(comp_to_merge.values())
        if n_components_merged > 1:
            logger.debug(f'Reassigning components')
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
            logger.debug(f'Got {len(set(partition_merged.values()))} components')
        else:
            logger.debug(f'No need to reassign components')
            partition_merged = partition
        return partition_merged, n_components_merged

    def sort_components(self, partition_merged, n_components_merged, current=0, task=None):
        logger.debug('Sorting components by size descending')
        components = set(partition_merged.values())
        comp_sizes = {c: sum([partition_merged[node] == c for node in partition_merged.keys()]) for c in components}
        # Hack to sort map values by key
        keysort = lambda seq: sorted(range(len(seq)), key=seq.__getitem__, reverse=True)
        sorted_comps = list(keysort(list(comp_sizes.values())))
        sort_order = dict(zip(sorted_comps, range(len(components))))
        logger.debug(f'Components reordering by size: {sort_order}')
        sorted_partition = {node: sort_order[c] for node, c in partition_merged.items()}

        if n_components_merged > 0:
            comp_other = sorted_comps.index(0)  # Other component is 0!
        else:
            comp_other = None
        logger.debug(f'Component OTHER: {comp_other}')
        return sort_order, sorted_partition, comp_other

    def popular_journals(self, df, n=TOP_JOURNALS, current=0, task=None):
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

    def popular_authors(self, df, n=TOP_AUTHORS, current=0, task=None):
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
    def get_most_cited_papers_for_comps(df, partition, n_papers=TOPIC_PAPERS):
        pdf = pd.DataFrame(partition.items(), columns=['id', 'comp'])
        ids_comp_df = pd.merge(left=df[['id', 'total']], left_on='id',
                               right=pdf, right_on='id', how='inner')
        ids = ids_comp_df.sort_values(by='total', ascending=False).groupby('comp')['id']
        return ids.apply(list).apply(lambda x: x[:n_papers]).to_dict()

    def dump(self):
        """
        Dump valuable fields of KeyPaperAnalyzer to JSON-serializable dict. Use 'load' to restore analyzer.
        """
        return {
            'df': self.df.to_json(),
            'df_kwd': self.df_kwd.to_json(),
            'citations_graph': json_graph.node_link_data(self.citations_graph),
            'similarity_graph': json_graph.node_link_data(self.similarity_graph),
            'structure_graph': json_graph.node_link_data(self.structure_graph),
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

        # Restore citation and structure graphs
        citations_graph = json_graph.node_link_graph(fields['citations_graph'])
        similarity_graph = json_graph.node_link_graph(fields['similarity_graph'])
        structure_graph = json_graph.node_link_graph(fields['structure_graph'])

        top_cited_papers = set(fields['top_cited_papers'])
        max_gain_papers = set(fields['max_gain_papers'])
        max_rel_gain_papers = set(fields['max_rel_gain_papers'])

        return {
            'df': df,
            'df_kwd': df_kwd,
            'citations_graph': citations_graph,
            'similarity_graph': similarity_graph,
            'structure_graph': structure_graph,
            'top_cited_papers': top_cited_papers,
            'max_gain_papers': max_gain_papers,
            'max_rel_gain_papers': max_rel_gain_papers
        }

    def init(self, fields):
        loaded = KeyPaperAnalyzer.load(fields)
        self.df = loaded['df']
        self.df_kwd = loaded['df_kwd']
        self.citations_graph = loaded['citations_graph']
        self.similarity_graph = loaded['similarity_graph']
        self.structure_graph = loaded['structure_graph']
        self.top_cited_papers = loaded['top_cited_papers']
        self.max_gain_papers = loaded['max_gain_papers']
        self.max_rel_gain_papers = loaded['max_rel_gain_papers']

    def pagerank(self, G, current=0, task=None):
        self.progress.info('Performing PageRank analysis', current=current, task=task)
        # Apply PageRank algorithm with damping factor of 0.5
        return nx.pagerank(G, alpha=0.5, tol=1e-9)