import logging
import re
from math import floor
from queue import PriorityQueue

import community
import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from pysrc.papers.db.pm_loader import PubmedLoader
from pysrc.papers.db.ss_loader import SemanticScholarLoader
from pysrc.papers.progress import Progress
from pysrc.papers.utils import split_df_list, get_topics_description, tokenize
from pysrc.prediction.ss_arxiv_loader import SSArxivLoader
from pysrc.prediction.ss_pubmed_loader import SSPubmedLoader

logger = logging.getLogger(__name__)


class KeyPaperAnalyzer:
    SEED = 20190723

    TOP_CITED_PAPERS = 50
    TOP_CITED_PAPERS_FRACTION = 0.1

    # ...bibliographic coupling (BC) was the most accurate,  followed by co-citation (CC).
    # Direct citation (DC) was a distant third among the three...
    SIMILARITY_BIBLIOGRAPHIC_COUPLING = 10
    SIMILARITY_COCITATION = 1
    SIMILARITY_TEXT_CITATION = 0.1
    SIMILARITY_CITATION = 0.01

    SIMILARITY_TEXT_MIN = 0.3  # Minimal cosine similarity for potential text citation
    SIMILARITY_TEXT_CITATION_N = 20  # Max number of potential text citations for paper

    # Reduce number of edges in smallest communities, i.e. topics
    STRUCTURE_LOW_LEVEL_SPARSITY = 0.3
    # Reduce number of edges between different topics to min number of inner edges * scale factor
    STRUCTURE_BETWEEN_TOPICS_SPARSITY = 0.05

    TOPIC_MIN_SIZE = 10
    TOPICS_MAX_NUMBER = 100
    TOPIC_PAPERS_TFIDF = 50
    TOPIC_WORDS = 20
    TFIDF_WORDS = 1000

    TOP_JOURNALS = 50
    TOP_AUTHORS = 50

    def __init__(self, loader, config, test=False):
        self.config = config
        self.progress = Progress(self.total_steps())

        self.loader = loader
        loader.set_progress(self.progress)

        # Determine source to provide correct URLs to articles,
        # see paper.py#get_loader_and_url_prefix
        # TODO: Bad design, refactor
        if isinstance(self.loader, PubmedLoader):
            self.source = 'Pubmed'
        elif isinstance(self.loader, SemanticScholarLoader):
            self.source = 'Semantic Scholar'
        elif isinstance(self.loader, SSArxivLoader):
            self.source = 'SSArxiv'
        elif isinstance(self.loader, SSPubmedLoader):
            self.source = 'SSPubmed'
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
            ids = self.loader.expand(ids, self.config.max_number_to_expand, current=current, task=task)
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

        # Building paper similarity graph, including all the papers from citations graph
        # IMPORTANT: not all the publications might be still covered
        self.similarity_graph = self.build_similarity_graph(
            self.df,
            self.citations_graph, cocit_grouped_df, self.bibliographic_coupling_df, current=9, task=task
        )

        if len(self.similarity_graph.nodes()) == 0:
            self.progress.info("Not enough papers to process topics analysis", current=10, task=task)
            self.df['comp'] = 0  # Technical value for top authors and papers analysis
            self.df_kwd = pd.DataFrame({'comp': [0], 'kwd': ['']})
            self.structure_graph = nx.Graph()
        else:
            # Perform topic analysis and get topic descriptions
            self.topics_dendrogram, self.partition, self.comp_other, self.components, self.comp_sizes = \
                self.topic_analysis(self.similarity_graph, current=10, task=task)

            self.progress.info('Computing topics descriptions by top cited papers', current=11, task=task)
            most_cited_per_comp = self.get_most_cited_papers_for_comps(self.df, self.partition)
            self.df = self.merge_col(self.df, self.partition, col='comp', na=-1)

            logger.debug('Prepare information for word cloud')
            tfidf_per_comp = get_topics_description(self.df, most_cited_per_comp, query, self.TFIDF_WORDS)
            kwds = [(comp, ','.join([f'{t}:{max(1e-3, v):.3f}' for t, v in vs[:self.TOPIC_WORDS]]))
                    for comp, vs in tfidf_per_comp.items()]
            self.df_kwd = pd.DataFrame(kwds, columns=['comp', 'kwd'])
            logger.debug(f'Components description\n{self.df_kwd["kwd"]}')
            # Build structure graph
            self.structure_graph = self.build_structure_graph(self.df, self.similarity_graph,
                                                              current=12, task=task)

        # Perform PageRank analysis
        pr = self.pagerank(self.citations_graph, current=13, task=task)
        self.df = self.merge_col(self.df, pr, col='pagerank', na=0.0)

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

    def build_similarity_graph(self, df, citations_graph, cocit_df, bibliographic_coupling_df, current=0, task=None):
        """
        Relationship graph is build using citation and text based methods.

        See papers:
        Which type of citation analysis generates the most accurate taxonomy of
        scientific and technical knowledge? (https://arxiv.org/pdf/1511.05078.pdf)
        ...bibliographic coupling (BC) was the most accurate,  followed by co-citation (CC).
        Direct citation (DC) was a distant third among the three...

        Sugiyama, K., Kan, M.Y.:
        Exploiting potential citation papers in scholarly paper recommendation. In: JCDL (2013)
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

        pids = list(df['id'])
        if len(df) >= 2:  # If we have any corpus
            self.progress.info(f'Citations based graph - {len(result.nodes())} nodes and {len(result.edges())} edges',
                               current=current, task=task)
            self.progress.info(f'Processing possible citations based on text similarity',
                               current=current, task=task)
            tfidf = self.compute_tfidf(df, self.TFIDF_WORDS, n_gram=1)
            cos_similarities = cosine_similarity(tfidf)
            text_citations = [PriorityQueue(maxsize=self.SIMILARITY_TEXT_CITATION_N) for _ in range(len(df))]

            # Adding text citations
            for i, pid1 in enumerate(df['id']):
                text_citations_i = text_citations[i]
                for j in range(i + 1, len(df)):
                    similarity = cos_similarities[i, j]
                    if np.isfinite(similarity) and similarity >= self.SIMILARITY_TEXT_MIN:
                        if text_citations_i.full():
                            text_citations_i.get()  # Removes the element with lowest similarity
                        text_citations_i.put((similarity, j))

            for i, pid1 in enumerate(df['id']):
                text_citations_i = text_citations[i]
                while not text_citations_i.empty():
                    similarity, j = text_citations_i.get()
                    pid2 = pids[j]
                    if result.has_edge(pid1, pid2):
                        pid1_pid2_edge = result[pid1][pid2]
                        pid1_pid2_edge['text'] = similarity
                    else:
                        result.add_edge(pid1, pid2, text=similarity)
        # Ensure all the papers are in the graph, separated ones will be placed to other component
        for pid in pids:
            if not result.has_node(pid):
                result.add_node(pid)
        self.progress.info(f'Built full similarity graph - {len(result.nodes())} nodes and {len(result.edges())} edges',
                           current=current, task=task)
        return result

    def topic_analysis(self, similarity_graph, current=0, task=None):
        self.progress.info(f'Extracting topics from paper similarity graph', current=current, task=task)
        connected_components = nx.number_connected_components(similarity_graph)
        logger.debug(f'Relations graph has {connected_components} connected components')

        logger.debug('Compute aggregated weight')
        for _, _, d in similarity_graph.edges(data=True):
            d['similarity'] = KeyPaperAnalyzer.get_similarity(d)

        logger.debug('Graph clustering via Louvain community algorithm')
        dendrogram = community.generate_dendrogram(
            similarity_graph, weight='similarity', random_state=KeyPaperAnalyzer.SEED
        )
        # Smallest communities
        partition_louvain = dendrogram[0]
        logger.debug(f'Found {len(set(partition_louvain.values()))} components')
        components = set(partition_louvain.values())
        comp_sizes = {c: sum([partition_louvain[node] == c for node in partition_louvain.keys()]) for c in components}
        logger.debug(f'Components: {comp_sizes}')
        if len(similarity_graph.edges) > 0:
            logger.debug('Calculate modularity for partition')
            modularity = community.modularity(partition_louvain, similarity_graph)
            logger.debug(f'Graph modularity (possible range is [-1, 1]): {modularity :.3f}')

        # Merge small components to 'OTHER'
        partition, n_components_merged = KeyPaperAnalyzer.merge_components(
            partition_louvain, topic_min_size=self.TOPIC_MIN_SIZE, max_topics_number=self.TOPICS_MAX_NUMBER)

        logger.debug('Sorting components by size descending')
        components = set(partition.values())
        comp_sizes = {c: sum([partition[node] == c for node in partition.keys()]) for c in components}
        # Hack to sort map values by key
        keysort = lambda seq: sorted(range(len(seq)), key=seq.__getitem__, reverse=True)
        sorted_comps = list(keysort(list(comp_sizes.values())))
        sort_order = dict(zip(sorted_comps, range(len(components))))
        logger.debug(f'Components reordering by size: {sort_order}')
        sorted_partition = {p: sort_order[c] for p, c in partition.items()}
        sorted_comp_sizes = {c: comp_sizes[sort_order[c]] for c in range(len(comp_sizes))}

        if n_components_merged > 0:
            comp_other = sorted_comps.index(0)  # Other component is 0!
        else:
            comp_other = None
        logger.debug(f'Component OTHER: {comp_other}')

        for k, v in sorted_comp_sizes.items():
            logger.debug(f'Component {k}: {v} ({int(100 * v / len(partition))}%)')

        logger.debug('Update components dendrogram according to merged topics')
        if len(dendrogram) >= 2:
            rename_map = {}
            for pid, v in partition_louvain.items():  # Pid -> smallest community
                if v not in rename_map:
                    rename_map[v] = sorted_partition[pid]
            comp_level = {rename_map[k]: v for k, v in dendrogram[1].items() if k in rename_map}

            logger.debug('Add artificial path for OTHER component')
            if comp_other is not None:
                comp_level[comp_other] = -1
                for d in dendrogram[2:]:
                    d[-1] = -1
            comp_dendrogram = [comp_level] + dendrogram[2:]
        else:
            comp_dendrogram = []

        return comp_dendrogram, sorted_partition, comp_other, components, sorted_comp_sizes

    def build_structure_graph(self, df, similarity_graph, current=0, task=None):
        """
        Structure graph is a hierarchical visualization of all the papers.
        It uses louvain community dendrogram as a structure.
        * For all the groups on the lowest level we show L-sparse subgraph.
        * Add top similarity connections between different topics to keep overall structure.
        * Relax continuous (2 or more consequent) hierarchical nodes to remove clutter.
        * Create dedicated group nodes for directly connected to the root nodes (improves OTHER visualization).
        """
        self.progress.info('Building structure graph', current=current, task=task)
        dendrogram = community.generate_dendrogram(
            similarity_graph, weight='similarity', random_state=KeyPaperAnalyzer.SEED
        )
        logger.debug('Processing louvain community dendrogram')
        result = nx.Graph()
        last_level = []
        topics_edges = {}  # Number of edges in locally sparsified graph for component
        topics_mean_similarities = {}  # Average similarity per component
        for i, dendrogram_level in enumerate(dendrogram):
            if i == 0:  # Smallest communities level, corresponds to topics
                papers_level_pids = {}
                for k, v in dendrogram_level.items():
                    if v not in papers_level_pids:
                        papers_level_pids[v] = set()
                    papers_level_pids[v].add(k)
                for v, pids in papers_level_pids.items():
                    logger.debug(f'Processing louvain hierarchical group {v} with pids {pids}')
                    topic_sparse = KeyPaperAnalyzer.local_sparse(similarity_graph.subgraph(pids),
                                                                 e=self.STRUCTURE_LOW_LEVEL_SPARSITY)
                    connected_components = [cc for cc in nx.connected_components(topic_sparse)]
                    logger.debug(f'Connected components {connected_components}')
                    # Build a map node -> connected group
                    connected_map = {}
                    for ci, cc in enumerate(connected_components):
                        for node in cc:
                            connected_map[node] = ci
                    logger.debug(f'Connected map {connected_map}')

                    logger.debug('Processing edges within sparse graph')
                    topic_similarity_sum = 0
                    topic_similarity_n = 0
                    topic_n = None
                    for (pu, pv, d) in topic_sparse.edges(data=True):
                        result.add_edge(pu, pv, **d)
                        if topic_n is None:
                            topic_n = int(df.loc[df['id'] == pu]['comp'].values[0])
                            topics_edges[topic_n] = len(topic_sparse.edges)
                        topic_similarity_n += 1
                        topic_similarity_sum += d['similarity']
                    topics_mean_similarities[topic_n] = topic_similarity_sum / max(1, topic_similarity_n)

                    logger.debug('Connecting top cited paper of each connected component to the hierarchy')
                    node_v = f'level_{len(dendrogram)}_{v}'
                    connected_set = set()
                    for node in df.loc[df['id'].isin(pids)].sort_values(by='total', ascending=False)['id']:
                        # Isolated nodes don't belong to any connected component
                        if node not in connected_map:
                            result.add_edge(node, node_v)
                        else:
                            ci = connected_map[node]
                            if ci not in connected_set:
                                result.add_edge(node, node_v)
                                connected_set.add(ci)  # Mark connected component as connected to the hierarchy
                    # Connect to root if necessary
                    if i == len(dendrogram) - 1:
                        last_level.append(node_v)
            else:
                for k, v in dendrogram_level.items():
                    node_k = f'level_{len(dendrogram) - i + 1}_{k}'
                    node_v = f'level_{len(dendrogram) - i}_{v}'
                    if result.has_node(node_k):
                        result.add_edge(node_k, node_v)
                        # Connect to root
                        if i == len(dendrogram) - 1:
                            last_level.append(node_v)

        root_node = f'root'
        for n in last_level:
            result.add_edge(n, root_node)

        logger.debug('Cleanup bypass hierarchical nodes')
        ws = set()
        for node in result.nodes:
            if re.match('level_.*', node):
                ws.add(node)
        while len(ws) > 0:
            nws = set()
            for node in ws:
                if result.has_node(node):
                    neighbors = list(result.neighbors(node))
                    if len(neighbors) == 2:
                        n1, n2 = neighbors
                        # Leave intermediate node for directly-connected nodes!
                        if not (n1 == root_node and not re.match('level_.*', n2) or
                                n2 == root_node and not re.match('level_.*', n1)):
                            result.remove_edge(node, n1)
                            result.remove_edge(node, n2)
                            result.add_edge(n1, n2)
                            result.remove_node(node)
                            if re.match('level_.*', n1):
                                nws.add(n1)
                            if re.match('level_.*', n1):
                                nws.add(n2)
                        else:
                            # Process special node for component
                            if n1 == root_node:
                                comp = int(df.loc[df['id'] == n2]['comp'].values[0])
                            else:
                                comp = int(df.loc[df['id'] == n1]['comp'].values[0])
                            result.remove_edge(node, n1)
                            result.remove_edge(node, n2)
                            result.remove_node(node)
                            comp_node = f'level_{comp}'
                            if not result.has_node(comp_node):
                                result.add_edge(root_node, comp_node)
                            if n1 == root_node:
                                result.add_edge(comp_node, n2)
                            else:
                                result.add_edge(comp_node, n1)
            ws = nws

        logger.debug('Ensure all the papers are processed, separated ones will be placed to other component')
        for pid in df['id']:
            if not result.has_node(pid):
                result.add_node(pid)

        logger.debug('Add top similarity edges between topics')
        sources = [None] * len(similarity_graph.edges)
        targets = [None] * len(similarity_graph.edges)
        similarities = [0.0] * len(similarity_graph.edges)
        i = 0
        for u, v, data in similarity_graph.edges(data=True):
            sources[i] = u
            targets[i] = v
            similarities[i] = KeyPaperAnalyzer.get_similarity(data)
            i += 1
        similarity_df = pd.DataFrame(data={'source': sources, 'target': targets, 'similarity': similarities})

        logger.debug('Assign each paper with corresponding component / topic')
        similarity_topics_df = similarity_df.merge(df[['id', 'comp']], how='left', left_on='source', right_on='id') \
            .merge(df[['id', 'comp']], how='left', left_on='target', right_on='id')

        inter_topics = {}
        for i, row in similarity_topics_df.iterrows():
            pid1, c1, pid2, c2, similarity = row['id_x'], row['comp_x'], row['id_y'], row['comp_y'], row['similarity']
            if c1 == c2:
                continue  # Ignore same group
            if c2 > c1:  # Swap
                pidt, ct = pid1, c1
                pid1, c1 = pid2, c2
                pid2, c2 = pidt, ct
            # Ignore between components similarities less than average within groups
            if similarity < (topics_mean_similarities[c1] + topics_mean_similarities[c2]) / 2:
                continue
            if (c1, c2) not in inter_topics:
                # Do not add edges between topics more than minimum * scale factor
                max_connections = int(min(topics_edges[c1], topics_edges[c2]) * self.STRUCTURE_BETWEEN_TOPICS_SPARSITY)
                pq = inter_topics[(c1, c2)] = PriorityQueue(maxsize=max_connections)
            else:
                pq = inter_topics[(c1, c2)]
            if pq.full():
                pq.get()  # Removes the element with lowest similarity
            pq.put((similarity, pid1, pid2))
        for pq in inter_topics.values():
            while not pq.empty():
                _, pid1, pid2 = pq.get()
                # Add edge with full info
                result.add_edge(pid1, pid2, **similarity_graph.edges[pid1, pid2])
        return result

    @staticmethod
    def local_sparse(graph, e):
        assert 0 < e < 1, f'sparsity e parameter should be in 0..1'
        result = nx.Graph()
        neighbours = {node: set(graph.neighbors(node)) for node in graph.nodes}
        sim_queues = {node: PriorityQueue(maxsize=max(1, floor(pow(len(neighbours[node]), e))))
                      for node in graph.nodes}
        for (u, v, s) in graph.edges(data='similarity'):
            qu = sim_queues[u]
            if qu.full():
                qu.get()  # Removes the element with lowest similarity
            qu.put((s, v))
            qv = sim_queues[v]
            if qv.full():
                qv.get()  # Removes the element with lowest similarity
            qv.put((s, u))
        for u, q in sim_queues.items():
            while not q.empty():
                s, v = q.get()
                if not result.has_edge(u, v):
                    result.add_edge(u, v, **graph.edges[u, v])
        return result

    @staticmethod
    def get_similarity(d):
        return \
            KeyPaperAnalyzer.SIMILARITY_COCITATION * d.get('cocitation', 0) + \
            KeyPaperAnalyzer.SIMILARITY_BIBLIOGRAPHIC_COUPLING * d.get('bibcoupling', 0) + \
            KeyPaperAnalyzer.SIMILARITY_CITATION * d.get('citation', 0) + \
            KeyPaperAnalyzer.SIMILARITY_TEXT_CITATION * d.get('text', 0)

    @staticmethod
    def compute_tfidf(df, max_features, n_gram):
        logger.debug(f'Compute global TF-IDF {len(df)}x{max_features}')
        corpus = [f'{t} {a}' for t, a in zip(df['title'], df['abstract'])]
        vectorizer = CountVectorizer(max_df=0.8, ngram_range=(1, n_gram),
                                     max_features=max_features,
                                     tokenizer=lambda t: tokenize(t))
        counts = vectorizer.fit_transform(corpus)
        tfidf_transformer = TfidfTransformer()
        tfidf = tfidf_transformer.fit_transform(counts)
        return tfidf

    @staticmethod
    def merge_col(df, data, col, na):
        t = pd.Series(data).reset_index().rename(columns={'index': 'id', 0: col})
        t['id'] = t['id'].astype(str)
        df_merged = pd.merge(df, t, on='id', how='outer')
        df_merged[col] = df_merged[col].fillna(na)
        return df_merged

    def find_top_cited_papers(self, df, n_papers=TOP_CITED_PAPERS, threshold=TOP_CITED_PAPERS_FRACTION,
                              current=0, task=None):
        self.progress.info(f'Identifying top cited papers', current=current, task=task)
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

    @staticmethod
    def merge_components(partition, topic_min_size, max_topics_number):
        logger.debug(f'Merging components to get max {max_topics_number} components into to "Other" component')
        components = set(partition.values())
        comp_sizes = {c: sum([partition[node] == c for node in partition.keys()]) for c in components}
        sorted_comps = sorted(comp_sizes.keys(), key=lambda c: comp_sizes[c], reverse=True)
        # Limit max number of topics
        if len(components) > max_topics_number:
            components_to_merge = set(sorted_comps[max_topics_number - 1:])
        else:
            components_to_merge = set()
        # Merge tiny topics
        for c, csize in comp_sizes.items():
            if csize < topic_min_size:
                components_to_merge.add(c)
        if components_to_merge:
            n_components_merged = len(components_to_merge)
            logger.debug(f'Reassigning components')
            partition_merged = {}
            new_comps = {}
            ci = 1  # Start with 1, OTHER component is 0
            for node, comp in partition.items():
                if comp in components_to_merge:
                    partition_merged[node] = 0  # Other
                    continue
                if comp not in new_comps:
                    new_comps[comp] = ci
                    ci += 1
                partition_merged[node] = new_comps[comp]
            logger.debug(f'Got {len(set(partition_merged.values()))} components')
            return partition_merged, n_components_merged
        else:
            logger.debug(f'No need to reassign components')
            return partition, 0

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
    def get_most_cited_papers_for_comps(df, partition, n_papers=TOPIC_PAPERS_TFIDF):
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
            'comp_other': self.comp_other,
            'df_kwd': self.df_kwd.to_json(),
            'citations_graph': json_graph.node_link_data(self.citations_graph),
            'similarity_graph': json_graph.node_link_data(self.similarity_graph),
            'structure_graph': json_graph.node_link_data(self.structure_graph),
            'topics_dendrogram': self.topics_dendrogram,
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
        comp_other = fields['comp_other']
        mapping = {}
        for col in df.columns:
            try:
                mapping[col] = int(col)
            except ValueError:
                mapping[col] = col
        df = df.rename(columns=mapping)

        # Restore topic descriptions
        df_kwd = pd.read_json(fields['df_kwd'])

        # Extra filter is applied to overcome split behaviour problem: split('') = [''] problem
        df_kwd['kwd'] = [kwd.split(',') if kwd != '' else [] for kwd in df_kwd['kwd']]
        df_kwd['kwd'] = df_kwd['kwd'].apply(lambda x: [el.split(':') for el in x])
        df_kwd['kwd'] = df_kwd['kwd'].apply(lambda x: [(el[0], float(el[1])) for el in x])

        # Restore citation and structure graphs
        citations_graph = json_graph.node_link_graph(fields['citations_graph'])
        similarity_graph = json_graph.node_link_graph(fields['similarity_graph'])
        structure_graph = json_graph.node_link_graph(fields['structure_graph'])
        topics_dendrogram = fields['topics_dendrogram']

        top_cited_papers = set(fields['top_cited_papers'])
        max_gain_papers = set(fields['max_gain_papers'])
        max_rel_gain_papers = set(fields['max_rel_gain_papers'])

        return {
            'df': df,
            'comp_other': comp_other,
            'df_kwd': df_kwd,
            'citations_graph': citations_graph,
            'similarity_graph': similarity_graph,
            'structure_graph': structure_graph,
            'topics_dendrogram': topics_dendrogram,
            'top_cited_papers': top_cited_papers,
            'max_gain_papers': max_gain_papers,
            'max_rel_gain_papers': max_rel_gain_papers
        }

    def init(self, fields):
        logger.debug(f'Loading\n{fields}')
        loaded = KeyPaperAnalyzer.load(fields)
        logger.debug(f'Loaded\n{loaded}')
        self.df = loaded['df']
        self.comp_other = loaded['comp_other']
        self.df_kwd = loaded['df_kwd']
        self.citations_graph = loaded['citations_graph']
        self.similarity_graph = loaded['similarity_graph']
        self.structure_graph = loaded['structure_graph']
        self.topics_dendrogram = loaded['topics_dendrogram']
        self.top_cited_papers = loaded['top_cited_papers']
        self.max_gain_papers = loaded['max_gain_papers']
        self.max_rel_gain_papers = loaded['max_rel_gain_papers']

    def pagerank(self, G, current=0, task=None):
        self.progress.info('Performing PageRank analysis', current=current, task=task)
        # Apply PageRank algorithm with damping factor of 0.5
        return nx.pagerank(G, alpha=0.5, tol=1e-9)
