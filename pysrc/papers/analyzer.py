import html
import logging
from collections import Counter
from math import floor
from queue import PriorityQueue

import community
import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph

from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.extract_numbers import extract_metrics, MetricExtractor
from pysrc.papers.progress import Progress
from pysrc.papers.utils import split_df_list, get_topics_description, SORT_MOST_CITED, \
    compute_tfidf, cosine_similarity, vectorize_corpus, tokens_stems, get_evolution_topics_description

logger = logging.getLogger(__name__)


class KeyPaperAnalyzer:
    SEED = 20190723

    TOP_CITED_PAPERS = 50

    # ...bibliographic coupling (BC) was the most accurate,  followed by co-citation (CC).
    # Direct citation (DC) was a distant third among the three...
    SIMILARITY_BIBLIOGRAPHIC_COUPLING = 2
    SIMILARITY_COCITATION = 1
    SIMILARITY_CITATION = 0.5  # Maximum single citation between two papers
    SIMILARITY_TEXT_CITATION = 1  # Cosine similarity is <= 1

    SIMILARITY_TEXT_MIN = 0.3  # Minimal cosine similarity for potential text citation
    SIMILARITY_TEXT_CITATION_N = 20  # Max number of potential text citations for paper

    # Reduce number of edges in smallest communities, i.e. topics
    STRUCTURE_LOW_LEVEL_SPARSITY = 0.5
    # Reduce number of edges between different topics to min number of inner edges * scale factor
    STRUCTURE_BETWEEN_TOPICS_SPARSITY = 0.2

    TOPIC_MIN_SIZE = 10
    TOPICS_MAX_NUMBER = 100
    TOPIC_PAPERS_TFIDF = 50
    TOPIC_WORDS = 20

    VECTOR_WORDS = 10000
    VECTOR_NGRAMS = 1
    VECTOR_MIN_DF = 0.01
    VECTOR_MAX_DF = 0.5

    TOP_JOURNALS = 50
    TOP_AUTHORS = 50

    EXPAND_STEPS = 2
    # Limit citations count of expanded papers to avoid prevalence of related methods
    EXPAND_CITATIONS_SIGMA = 5
    # Take up to fraction of top similarity
    EXPAND_SIMILARITY_THRESHOLD = 0.2
    EXPAND_ZOOM_OUT = 100

    EVOLUTION_MIN_PAPERS = 100
    EVOLUTION_STEP = 10

    def __init__(self, loader, config, test=False):
        self.config = config
        self.progress = Progress(self.total_steps())

        self.loader = loader
        self.source = Loaders.source(self.loader, test)

    def total_steps(self):
        return 21 + 1  # One extra step for visualization

    def teardown(self):
        self.progress.remove_handler()

    def search_terms(self, query, limit=None, sort=None, noreviews=True, task=None):
        # Search articles relevant to the terms
        if len(query) == 0:
            raise SearchError('Empty search string, please use search terms or '
                              'all the query wrapped in "" for phrasal search')
        limit = limit or self.config.max_number_of_articles
        sort = sort or SORT_MOST_CITED
        self.progress.info(f'Searching {limit} {sort.lower()} publications matching {html.escape(query)}',
                           current=1, task=task)
        if noreviews:
            self.progress.info('Preferring non review papers', current=1, task=task)

        ids = self.loader.search(query, limit=limit, sort=sort, noreviews=noreviews)
        if len(ids) == 0:
            raise SearchError(f"Nothing found for search query: {query}")
        else:
            self.progress.info(f'Found {len(ids)} publications in the local database', current=1,
                               task=task)
        return ids

    def expand_ids(self, ids, limit, steps, current=1, task=None):
        if len(ids) > self.config.max_number_to_expand:
            self.progress.info('Too many related papers, nothing to expand', current=current, task=task)
            return ids
        self.progress.info('Expanding related papers by references', current=current, task=task)
        logger.debug(f'Expanding {len(ids)} papers to: {limit}')
        mean, std = self.loader.estimate_citations(ids)
        logger.debug(f'Estimated citations count mean={mean}, std={std}')
        current_ids = ids

        publications = self.loader.load_publications(ids)
        mesh_stems = [s for s, _ in tokens_stems(
            ' '.join(publications['mesh'] + ' ' + publications['keywords']).replace(',', ' ')
        )]
        mesh_counter = Counter(mesh_stems)

        # Expand while we can
        i = 0
        new_ids = []
        while True:
            if i == steps or len(current_ids) >= limit:
                break
            i += 1
            logger.debug(f'Step {i}: current_ids: {len(current_ids)}, new_ids: {len(new_ids)}, limit: {limit}')
            papers_to_expand = new_ids or current_ids
            number_to_expand = limit - len(current_ids) if not mesh_stems else self.config.max_number_to_expand
            expanded_df = self.loader.expand(papers_to_expand, number_to_expand)
            logger.debug(f'Expanded {len(new_ids)} papers')

            new_df = expanded_df.loc[np.logical_not(expanded_df['id'].isin(set(current_ids)))]
            logging.debug(f'New papers {len(new_df)}')

            if len(ids) > 1:  # Don't keep citations distribution in case of paper analysis
                logger.debug(f'Filter by citations count mean({mean}) +- {self.EXPAND_CITATIONS_SIGMA} * std({std})')
                new_df = new_df.loc[[
                    mean - self.EXPAND_CITATIONS_SIGMA * std <= t <= mean + self.EXPAND_CITATIONS_SIGMA * std
                    for t in new_df['total']]]
                new_ids = list(new_df['id'])
                logger.debug(f'Citations filtered: {len(new_ids)}')

            if len(new_ids) == 0:
                break

            # No additional filtration required
            if not mesh_stems:
                current_ids += new_ids
                continue

            logger.debug(f'Mesh most common:\n' + ','.join(f'{k}:{"{0:.3f}".format(v / len(mesh_stems))}'
                                                           for k, v in mesh_counter.most_common(100)))
            new_publications = self.loader.load_publications(new_ids)
            fcs = []
            for _, row in new_publications.iterrows():
                pid = row['id']
                mesh = row['mesh']
                keywords = row['keywords']
                title = row['title']
                new_mesh_stems = [s for s, _ in tokens_stems((mesh + ' ' + keywords).replace(',', ' '))]
                if new_mesh_stems:
                    # Estimate fold change of similarity vs random single paper
                    similarity = sum([mesh_counter[s] / (len(mesh_stems) / len(ids)) for s in new_mesh_stems])
                    fcs.append([pid, False, similarity, title, ','.join(new_mesh_stems)])
                else:
                    fcs.append([pid, True, 0.0, title, ''])

            fcs.sort(key=lambda v: v[2], reverse=True)
            sim_threshold = fcs[0][2] * self.EXPAND_SIMILARITY_THRESHOLD  # Compute threshold as a fraction of top
            for v in fcs:
                v[1] = v[1] or v[2] > sim_threshold

            logger.debug('Pid\tOk\tSimilarity\tTitle\tMesh\n' +
                         '\n'.join(f'{p}\t{"+" if a else "-"}\t{int(s)}\t{t}\t{m}' for
                                   p, a, s, t, m in fcs))
            new_mesh_ids = [v[0] for v in fcs if v[1]][:limit - len(current_ids)]
            logger.debug(f'Similar by mesh papers: {len(new_mesh_ids)}')
            if len(new_mesh_ids) == 0:
                break
            new_ids = new_mesh_ids
            current_ids += new_ids

        self.progress.info(f'Expanded to {len(current_ids)} papers', current=current, task=task)
        return current_ids

    def analyze_papers(self, ids, query, noreviews=True, task=None):
        """:return full log"""
        self.ids = ids
        self.query = query

        self.progress.info('Loading publication data', current=2, task=task)
        self.pub_df = self.loader.load_publications(ids)
        if len(self.pub_df) == 0:
            raise SearchError(f'Nothing found for ids: {ids}')
        self.ids = set(self.pub_df['id'])  # Limit ids to existing papers only!
        self.n_papers = len(self.ids)
        self.pub_types = list(set(self.pub_df['type']))

        self.progress.info('Analyzing title and abstract texts', current=3, task=task)
        self.corpus_ngrams, self.corpus_counts = \
            vectorize_corpus(self.pub_df,
                             max_features=KeyPaperAnalyzer.VECTOR_WORDS, n_gram=KeyPaperAnalyzer.VECTOR_NGRAMS,
                             min_df=KeyPaperAnalyzer.VECTOR_MIN_DF, max_df=KeyPaperAnalyzer.VECTOR_MAX_DF)
        tfidf = compute_tfidf(self.corpus_counts)

        self.progress.info('Processing texts similarity', current=4, task=task)
        self.texts_similarity = self.analyze_texts_similarity(self.pub_df, tfidf)

        self.progress.info('Loading citations statistics by year', current=5, task=task)
        cits_by_year_df = self.loader.load_citations_by_year(self.ids)
        self.progress.info(f'Found {len(cits_by_year_df)} records of citations by year',
                           current=5, task=task)

        self.cit_stats_df = self.build_cit_stats_df(cits_by_year_df, self.n_papers)
        if len(self.cit_stats_df) == 0:
            raise SearchError('No citations of papers were found')
        self.df, self.min_year, self.max_year, self.citation_years = self.merge_citation_stats(
            self.pub_df, self.cit_stats_df)

        # Load data about citations between given papers (excluding outer papers)
        # IMPORTANT: cit_df may contain not all the publications for query
        self.progress.info('Loading citations data', current=6, task=task)
        self.cit_df = self.loader.load_citations(self.ids)
        self.progress.info(f'Found {len(self.cit_df)} citations between papers', current=5, task=task)

        # Building inner citations graph for pagerank analysis
        self.citations_graph = self.build_citation_graph(self.cit_df, current=7, task=task)

        self.progress.info('Calculating co-citations for selected papers', current=8, task=task)
        self.cocit_df = self.loader.load_cocitations(self.ids)
        self.progress.info(f'Found {len(self.cocit_df)} co-cited pairs of papers', current=8, task=task)

        cocit_grouped_df = self.build_cocit_grouped_df(self.cocit_df)

        self.progress.info('Processing bibliographic coupling for selected papers', current=9, task=task)
        self.bibliographic_coupling_df = self.loader.load_bibliographic_coupling(self.ids)
        self.progress.info(f'Found {len(self.bibliographic_coupling_df)} bibliographic coupling pairs of papers',
                           current=9, task=task)

        # All the papers will be covered by default
        self.similarity_graph = self.build_similarity_graph(
            self.df, self.texts_similarity,
            self.citations_graph, cocit_grouped_df, self.bibliographic_coupling_df,
            current=10, task=task
        )

        if len(self.similarity_graph.nodes()) == 0:
            self.progress.info('Not enough papers to process topics analysis', current=11, task=task)
            self.df['comp'] = 0  # Technical value for top authors and papers analysis
            self.df_kwd = pd.DataFrame({'comp': [0], 'kwd': ['']})
            self.structure_graph = nx.Graph()
        else:
            self.progress.info('Extracting topics from paper similarity graph', current=11, task=task)
            self.topics_dendrogram, self.partition, self.comp_other, self.components, self.comp_sizes = \
                self.topic_analysis(self.similarity_graph)
            self.df = self.merge_col(self.df, self.partition, col='comp', na=-1)

            self.progress.info('Computing topics descriptions by top cited papers', current=12, task=task)
            most_cited_per_comp = self.get_most_cited_papers_for_comps(
                self.df.loc[self.df['type'] != 'Review'] if noreviews else self.df,
                self.partition
            )
            tfidf_per_comp = get_topics_description(self.df, most_cited_per_comp,
                                                    self.corpus_ngrams, self.corpus_counts,
                                                    query, self.TOPIC_WORDS)
            kwds = [(comp, ','.join([f'{t}:{max(1e-3, v):.3f}' for t, v in vs[:self.TOPIC_WORDS]]))
                    for comp, vs in tfidf_per_comp.items()]
            self.df_kwd = pd.DataFrame(kwds, columns=['comp', 'kwd'])
            logger.debug(f'Components description\n{self.df_kwd["kwd"]}')
            # Build structure graph
            self.structure_graph = self.build_structure_graph(self.df, self.similarity_graph)

        self.progress.info('Performing PageRank analysis', current=14, task=task)
        pr = nx.pagerank(self.citations_graph, alpha=0.5, tol=1e-9)
        self.df = self.merge_col(self.df, pr, col='pagerank', na=0.0)

        self.progress.info('Identifying top cited papers', current=15, task=task)
        self.top_cited_papers, self.top_cited_df = self.find_top_cited_papers(self.df)

        self.progress.info('Identifying top cited papers for each year', current=16, task=task)
        self.max_gain_papers, self.max_gain_df = self.find_max_gain_papers(self.df, self.citation_years)

        self.progress.info('Identifying hot papers of the year', current=17, task=task)
        self.max_rel_gain_papers, self.max_rel_gain_df = self.find_max_relative_gain_papers(
            self.df, self.citation_years
        )

        self.progress.info("Finding popular journals", current=18, task=task)
        self.journal_stats = self.popular_journals(self.df)

        self.progress.info("Finding popular authors", current=19, task=task)
        self.author_stats = self.popular_authors(self.df)

        if len(self.df) >= 0:
            logger.debug('Perform numbers extraction')
            self.progress.info('Extracting numbers from publication abstracts', current=20, task=task)
            self.numbers_df = self.extract_numbers(self.df)
        else:
            logger.debug('Not enough papers for numbers extraction')

        if len(self.df) >= KeyPaperAnalyzer.EVOLUTION_MIN_PAPERS:
            logger.debug('Perform topic evolution analysis and get topic descriptions')
            self.evolution_df, self.evolution_year_range = \
                self.topic_evolution_analysis(self.cocit_df, current=21, task=task)
            self.evolution_kwds = self.topic_evolution_descriptions(
                self.df, self.evolution_df, self.evolution_year_range, current=21, task=task
            )
        else:
            logger.debug('Not enough papers for topics evolution')
            self.evolution_df = None

    @staticmethod
    def build_cit_stats_df(cits_by_year_df, n_papers):
        # Get citation stats with columns 'id', year_1, ..., year_N and fill NaN with 0
        df = cits_by_year_df.pivot(index='id', columns='year', values='count').reset_index().fillna(0)

        # Fix column names from float 'YYYY.0' to int 'YYYY'
        mapper = {col: int(col) for col in df.columns if col != 'id'}
        df = df.rename(mapper)

        df['total'] = df.iloc[:, 1:].sum(axis=1)
        df = df.sort_values(by='total', ascending=False)
        logger.debug(f'Loaded citation stats for {len(df)} of {n_papers} papers')

        return df

    @staticmethod
    def build_cocit_grouped_df(cocit_df):
        logger.debug('Aggregating co-citations')
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
        self.progress.info('Building citation graph', current=current, task=task)
        G = nx.DiGraph()
        for index, row in cit_df.iterrows():
            v, u = row['id_out'], row['id_in']
            G.add_edge(v, u)

        self.progress.info(f'Built citation graph - {len(G.nodes())} nodes and {len(G.edges())} edges',
                           current=current, task=task)
        return G

    def build_similarity_graph(
            self,
            df, texts_similarity,
            citations_graph, cocit_df, bibliographic_coupling_df,
            process_cocitations=True,
            process_bibliographic_coupling=True,
            process_citations=True,
            process_text_citations=True,
            process_all_papers=True,
            current=0, task=None
    ):
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
        self.progress.info('Building papers similarity graph', current=current, task=task)
        pids = list(df['id'])

        result = nx.Graph()
        # NOTE: we use nodes id as String to avoid problems str keys in jsonify
        # during graph visualization
        if process_cocitations and len(cocit_df) > 0:
            for el in cocit_df[['cited_1', 'cited_2', 'total']].values:
                start, end, cocitation = str(el[0]), str(el[1]), float(el[2])
                result.add_edge(start, end, cocitation=cocitation)

        if process_bibliographic_coupling and len(bibliographic_coupling_df) > 0:
            for el in bibliographic_coupling_df[['citing_1', 'citing_2', 'total']].values:
                start, end, bibcoupling = str(el[0]), str(el[1]), float(el[2])
                if result.has_edge(start, end):
                    result[start][end]['bibcoupling'] = bibcoupling
                else:
                    result.add_edge(start, end, bibcoupling=bibcoupling)

        if process_citations:
            for u, v in citations_graph.edges:
                if result.has_edge(u, v):
                    result[u][v]['citation'] = 1
                else:
                    result.add_edge(u, v, citation=1)

        self.progress.info(f'Citations based graph - {len(result.nodes())} nodes and {len(result.edges())} edges',
                           current=current, task=task)

        if process_text_citations:
            if len(df) >= 2:  # If we have any corpus
                for i, pid1 in enumerate(df['id']):
                    similarity_queue = texts_similarity[i]
                    while not similarity_queue.empty():
                        similarity, j = similarity_queue.get()
                        pid2 = pids[j]
                        if result.has_edge(pid1, pid2):
                            pid1_pid2_edge = result[pid1][pid2]
                            pid1_pid2_edge['text'] = similarity
                        else:
                            result.add_edge(pid1, pid2, text=similarity)

        if process_all_papers:
            # Ensure all the papers are in the graph
            for pid in pids:
                if not result.has_node(pid):
                    result.add_node(pid)

        self.progress.info(f'Built full similarity graph - {len(result.nodes())} nodes and {len(result.edges())} edges',
                           current=current, task=task)
        return result

    @staticmethod
    def analyze_texts_similarity(df, tfidf):
        cos_similarities = cosine_similarity(tfidf)
        similarity_queues = [PriorityQueue(maxsize=KeyPaperAnalyzer.SIMILARITY_TEXT_CITATION_N)
                             for _ in range(len(df))]
        # Adding text citations
        for i, pid1 in enumerate(df['id']):
            queue_i = similarity_queues[i]
            for j in range(i + 1, len(df)):
                similarity = cos_similarities[i, j]
                if np.isfinite(similarity) and similarity >= KeyPaperAnalyzer.SIMILARITY_TEXT_MIN:
                    if queue_i.full():
                        queue_i.get()  # Removes the element with lowest similarity
                    queue_i.put((similarity, j))
        return similarity_queues

    @staticmethod
    def topic_analysis(similarity_graph):
        connected_components = nx.number_connected_components(similarity_graph)
        logger.debug(f'Relations graph has {connected_components} connected components')

        logger.debug('Compute aggregated similarity')
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

        # Reorder and merge small components to 'OTHER'
        partition, n_components_merged = KeyPaperAnalyzer.merge_components(
            partition_louvain,
            topic_min_size=KeyPaperAnalyzer.TOPIC_MIN_SIZE,
            max_topics_number=KeyPaperAnalyzer.TOPICS_MAX_NUMBER
        )

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

    @staticmethod
    def build_structure_graph(df, similarity_graph):
        """
        Structure graph is visualization of similarity connections.
        It uses topics a structure.
        * For all the topics we show locally sparse subgraph.
        * Add top similarity connections between different topics to keep overall structure.
        """
        result = nx.Graph()
        topics_edges = {}  # Number of edges in locally sparse graph for topic
        topics_mean_similarities = {}  # Average similarity per component

        logger.debug('Processing topics local sparse graphs')
        for c in set(df['comp']):
            comp_df = df.loc[df['comp'] == c]
            logger.debug(f'Processing component {c}')
            topic_sparse = KeyPaperAnalyzer.local_sparse(similarity_graph.subgraph(comp_df['id']),
                                                         e=KeyPaperAnalyzer.STRUCTURE_LOW_LEVEL_SPARSITY)
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

        logger.debug('Ensure all the papers are processed')
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
                max_connections = \
                    int(min(topics_edges[c1], topics_edges[c2]) * KeyPaperAnalyzer.STRUCTURE_BETWEEN_TOPICS_SPARSITY)
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
        assert 0 < e < 1, f'sparsity parameter {e} should be in 0..1'
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
            KeyPaperAnalyzer.SIMILARITY_BIBLIOGRAPHIC_COUPLING * d.get('bibcoupling', 0) + \
            KeyPaperAnalyzer.SIMILARITY_COCITATION * d.get('cocitation', 0) + \
            KeyPaperAnalyzer.SIMILARITY_CITATION * d.get('citation', 0) + \
            KeyPaperAnalyzer.SIMILARITY_TEXT_CITATION * d.get('text', 0)

    @staticmethod
    def merge_col(df, data, col, na):
        t = pd.Series(data).reset_index().rename(columns={'index': 'id', 0: col})
        t['id'] = t['id'].astype(str)
        df_merged = pd.merge(df, t, on='id', how='outer')
        df_merged[col] = df_merged[col].fillna(na)
        return df_merged

    @staticmethod
    def find_top_cited_papers(df, n_papers=TOP_CITED_PAPERS):
        papers_to_show = min(n_papers, len(df))
        top_cited_df = df.sort_values(by='total', ascending=False).iloc[:papers_to_show, :]
        top_cited_papers = set(top_cited_df['id'].values)
        return top_cited_papers, top_cited_df

    @staticmethod
    def find_max_gain_papers(df, citation_years):
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

    @staticmethod
    def find_max_relative_gain_papers(df, citation_years):
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
            logger.debug('Reassigning components')
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
            logger.debug('No need to reassign components')
            return partition, 0

    @staticmethod
    def popular_journals(df, n=TOP_JOURNALS):
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

    @staticmethod
    def popular_authors(df, n=TOP_AUTHORS):
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

    @staticmethod
    def extract_numbers(df):
        # Slow, currently moved out of the class to speed up fixing & rerunning the code of MetricExtractor
        metrics_data = []
        for _, data in df.iterrows():
            paper_metrics_data = [data['id'], *extract_metrics(data['abstract'])]
            metrics_data.append(paper_metrics_data)
        me = MetricExtractor(metrics_data)
        result = pd.merge(left=me.metrics_df, left_on='ID', right=df[['id', 'title']], right_on='id')
        result = result[['id', 'title', 'Metrics']]
        result['numbers'] = [
            '; '.join(
                f'{number}:{",".join(str(v) for v in sorted(set(v[0] for v in values)))}'
                for number, values in row['Metrics'].items()
            ) for _, row in result.iterrows()
        ]
        result = result.loc[result['numbers'] != '']
        return result[['id', 'title', 'numbers']]

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
        # Used for components naming
        self.df_kwd = loaded['df_kwd']
        # Used for structure visualization
        self.citations_graph = loaded['citations_graph']
        self.similarity_graph = loaded['similarity_graph']
        self.structure_graph = loaded['structure_graph']
        self.topics_dendrogram = loaded['topics_dendrogram']
        # Used for navigation
        self.top_cited_papers = loaded['top_cited_papers']
        self.max_gain_papers = loaded['max_gain_papers']
        self.max_rel_gain_papers = loaded['max_rel_gain_papers']
