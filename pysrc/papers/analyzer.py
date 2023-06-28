import logging

import numpy as np
import pandas as pd
from networkx.readwrite import json_graph
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from pysrc.papers.analysis.citations import find_top_cited_papers, find_max_gain_papers, \
    find_max_relative_gain_papers, build_cit_stats_df, merge_citation_stats, build_cocit_grouped_df
from pysrc.papers.analysis.evolution import topic_evolution_analysis, topic_evolution_descriptions
from pysrc.papers.analysis.graph import build_papers_graph, \
    to_weighted_graph, sparse_graph
from pysrc.papers.analysis.metadata import popular_authors, popular_journals
from pysrc.papers.analysis.node2vec import node2vec
from pysrc.papers.analysis.numbers import extract_numbers
from pysrc.papers.analysis.text import texts_embeddings, vectorize_corpus, tokens_embeddings
from pysrc.papers.analysis.topics import get_topics_description, cluster_and_sort
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.progress import Progress
from pysrc.papers.utils import SORT_MOST_CITED

logger = logging.getLogger(__name__)


class PapersAnalyzer:
    TOP_CITED_PAPERS = 50

    # Features are originally taken from paper:
    # 1) Which type of citation analysis generates the most accurate taxonomy of
    #   scientific and technical knowledge? (https://arxiv.org/pdf/1511.05078.pdf)
    #   ...bibliographic coupling (BC) was the most accurate,  followed by co-citation (CC).
    #   Direct citation (DC) was a distant third among the three...
    #
    SIMILARITY_COCITATION = 10  # Limiter by number of co-citations, applied to log
    SIMILARITY_BIBLIOGRAPHIC_COUPLING = 3  # Limited by number of references, applied to log
    SIMILARITY_CITATION = 1  # Limited by 1 citation

    # Minimal number of common references, used to reduces papers graph edges count
    # Value > 1 is especially useful while analysing single paper, removes meaningless connections by construction
    SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN = 1

    # Minimal number of common references, used to reduces papers graph edges count
    SIMILARITY_COCITATION_MIN = 1

    # Papers embeddings is a concatenation of graph and text embeddings times corresponding factors
    # Graph embeddings produce more clear topics separation, so it goes with bigger coefficient
    GRAPH_EMBEDDINGS_FACTOR = 5
    TEXT_EMBEDDINGS_FACTOR = 1

    # Reduce number of edges in papers graph
    PAPERS_GRAPH_EDGES_TO_NODES = 10

    # Global vectorization max vocabulary size
    VECTOR_WORDS = 10000
    # Terms with lower frequency will be ignored, remove rare words
    VECTOR_MIN_DF = 0.001
    # Terms with higher frequency will be ignored, remove abundant words
    VECTOR_MAX_DF = 0.8

    PCA_COMPONENTS = 30

    # Configure number and size of topics
    TOPICS_NUMBER_SMALL = dict(max_number=10, min_size=50)
    TOPICS_NUMBER_MEDIUM = dict(max_number=20, min_size=20)
    TOPICS_NUMBER_LARGE = dict(max_number=50, min_size=10)

    @staticmethod
    def get_topics_info(topics):
        if topics.lower() == 'small':
            info = PapersAnalyzer.TOPICS_NUMBER_SMALL
        elif topics.lower() == 'medium':
            info = PapersAnalyzer.TOPICS_NUMBER_MEDIUM
        elif topics.lower() == 'large':
            info = PapersAnalyzer.TOPICS_NUMBER_LARGE
        else:
            raise Exception(f'Unknown topics size: {topics}')
        return info['max_number'], info['min_size']

    # Number of top cited papers in topic picked for description computation
    TOPIC_MOST_CITED_PAPERS = 50
    # Number of words for topic description
    TOPIC_DESCRIPTION_WORDS = 10

    POPULAR_JOURNALS = 50
    POPULAR_AUTHORS = 50

    # Expand limit by references before filtration by citations and keywords
    EXPAND_LIMIT = 5000
    # Control citations count
    EXPAND_CITATIONS_Q_LOW = 5
    EXPAND_CITATIONS_Q_HIGH = 95
    EXPAND_CITATIONS_SIGMA = 3
    # Take up to fraction of top similarity
    EXPAND_SIMILARITY_THRESHOLD = 0.5

    # Impact of single paper when analyzing citations and mesh terms when analysing paper
    SINGLE_PAPER_IMPACT = 20

    EVOLUTION_MIN_PAPERS = 100
    EVOLUTION_STEP = 10

    def __init__(self, loader, config, test=False):
        self.config = config
        self.progress = Progress(self.total_steps())

        self.loader = loader
        self.source = Loaders.source(self.loader, test)

    def total_steps(self):
        return 14 + 1  # One extra step for visualization

    def teardown(self):
        self.progress.remove_handler()

    def search_terms(self, query, limit=None, sort=None, noreviews=True, task=None):
        # Search articles relevant to the terms
        if len(query) == 0:
            raise SearchError('Empty search string, please use search terms or '
                              'all the query wrapped in "" for phrasal search')
        limit = limit or self.config.max_number_of_articles
        sort = sort or SORT_MOST_CITED
        noreviews_msg = ", not reviews" if noreviews else ""
        self.progress.info(f'Searching {limit} {sort.lower()} publications matching {query}{noreviews_msg}',
                           current=1, task=task)
        ids = self.loader.search(query, limit=limit, sort=sort, noreviews=noreviews)
        if len(ids) == 0:
            raise SearchError(f"Nothing found for search query: {query}")
        else:
            self.progress.info(f'Found {len(ids)} publications in the database', current=1, task=task)
        return ids

    def analyze_papers(self, ids, query, topics='medium', test=False, task=None):
        self.progress.info('Loading publication data', current=2, task=task)
        self.query = query
        self.df = self.loader.load_publications(ids)
        if len(self.df) == 0:
            raise SearchError(f'Nothing found for ids: {ids}')
        else:
            self.progress.info(f'Total {len(self.df)} papers in database', current=2, task=task)
        ids = list(self.df['id'])  # Limit ids to existing papers only!
        self.pub_types = list(set(self.df['type']))

        self.progress.info('Analyzing title and abstract texts', current=3, task=task)
        self.corpus, self.corpus_tokens, self.corpus_counts = vectorize_corpus(
            self.df,
            max_features=PapersAnalyzer.VECTOR_WORDS,
            min_df=PapersAnalyzer.VECTOR_MIN_DF,
            max_df=PapersAnalyzer.VECTOR_MAX_DF,
            test=test
        )
        if PapersAnalyzer.TEXT_EMBEDDINGS_FACTOR != 0:
            logger.debug('Analyzing tokens embeddings')
            self.corpus_tokens_embedding = tokens_embeddings(
                self.corpus, self.corpus_tokens, test=test
            )
            logger.debug('Analyzing texts embeddings')
            self.texts_embeddings = texts_embeddings(
                self.corpus_counts, self.corpus_tokens_embedding
            )
        else:
            self.texts_embeddings = np.zeros(shape=(len(self.df), 0))

        self.progress.info('Loading citations for papers', current=4, task=task)
        logger.debug('Loading citations by year statistics')
        cits_by_year_df = self.loader.load_citations_by_year(ids)
        logger.debug(f'Found {len(cits_by_year_df)} records of citations by year')

        self.cit_stats_df = build_cit_stats_df(cits_by_year_df, len(ids))
        if len(self.cit_stats_df) == 0:
            logger.warning('No citations of papers were found')
        self.df, self.citation_years = merge_citation_stats(self.df, self.cit_stats_df)
        logger.debug('Loading citations information')
        self.cit_df = self.loader.load_citations(ids)
        logger.debug(f'Found {len(self.cit_df)} citations between papers')

        self.progress.info('Calculating co-citations for selected papers', current=5, task=task)
        self.cocit_df = self.loader.load_cocitations(ids)
        cocit_grouped_df = build_cocit_grouped_df(self.cocit_df)
        logger.debug(f'Found {len(cocit_grouped_df)} co-cited pairs of papers')
        self.cocit_grouped_df = cocit_grouped_df[cocit_grouped_df['total'] >= self.SIMILARITY_COCITATION_MIN].copy()
        logger.debug(f'Filtered {len(self.cocit_grouped_df)} co-cited pairs of papers, '
                     f'threshold {self.SIMILARITY_COCITATION_MIN}')

        self.progress.info('Processing bibliographic coupling for selected papers', current=6, task=task)
        bibliographic_coupling_df = self.loader.load_bibliographic_coupling(ids)
        logger.debug(f'Found {len(bibliographic_coupling_df)} bibliographic coupling pairs of papers')
        self.bibliographic_coupling_df = bibliographic_coupling_df[
            bibliographic_coupling_df['total'] >= self.SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN].copy()
        logger.debug(f'Filtered {len(self.bibliographic_coupling_df)} bibliographic coupling pairs of papers '
                     f'threshold {self.SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN}')

        self.progress.info('Building papers graph', current=7, task=task)
        self.papers_graph = build_papers_graph(
            self.df, self.cit_df, self.cocit_grouped_df, self.bibliographic_coupling_df,
        )
        self.progress.info(f'Analyzing papers graph - {self.papers_graph.number_of_nodes()} nodes and '
                           f'{self.papers_graph.number_of_edges()} edges', current=7, task=task)
        logger.debug('Analyzing papers graph embeddings')
        self.weighted_similarity_graph = to_weighted_graph(self.papers_graph, PapersAnalyzer.similarity)
        logger.debug('Prepare sparse graph for visualization')
        self.sparse_papers_graph = self.prepare_sparse_papers_graph(self.papers_graph, self.weighted_similarity_graph)

        if PapersAnalyzer.GRAPH_EMBEDDINGS_FACTOR != 0:
            gs = sparse_graph(self.weighted_similarity_graph, PapersAnalyzer.PAPERS_GRAPH_EDGES_TO_NODES)
            self.graph_embeddings = node2vec(self.df['id'], gs)
        else:
            self.graph_embeddings = np.zeros(shape=(len(self.df), 0))

        logger.debug('Computing aggregated graph and text embeddings for papers')
        papers_embeddings = np.concatenate(
            (self.graph_embeddings * PapersAnalyzer.GRAPH_EMBEDDINGS_FACTOR,
             self.texts_embeddings * PapersAnalyzer.TEXT_EMBEDDINGS_FACTOR), axis=1)

        if len(self.df) > 1:
            logger.debug('Computing PCA projection')
            pca = PCA(n_components=min(len(papers_embeddings), PapersAnalyzer.PCA_COMPONENTS))
            t = StandardScaler().fit_transform(papers_embeddings)
            self.pca_coords = pca.fit_transform(t)
            logger.debug(f'Explained variation {int(np.sum(pca.explained_variance_ratio_) * 100)}%')
            logger.debug('Apply TSNE transformation on papers PCA coords')
            tsne_embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(self.pca_coords)
            self.df['x'] = tsne_embeddings_2d[:, 0]
            self.df['y'] = tsne_embeddings_2d[:, 1]
        else:
            self.pca_coords = np.zeros(shape=(len(self.df), 128))
            self.df['x'] = 0
            self.df['y'] = 0

        self.progress.info(f'Extracting {topics.lower()} number of topics from papers text and graph similarity',
                           current=8, task=task)
        topics_max_number, topic_min_size = PapersAnalyzer.get_topics_info(topics)
        if self.papers_graph.number_of_nodes() <= topic_min_size:
            logger.debug('Small graph - single topic')
            self.clusters, self.dendrogram = [0] * len(self.df), None
            self.df['comp'] = self.clusters
        else:
            logger.debug('Extracting topics from papers embeddings')
            self.clusters, self.dendrogram = cluster_and_sort(self.pca_coords, topics_max_number, topic_min_size)
            self.df['comp'] = self.clusters

        self.progress.info(f'Analyzing {len(set(self.df["comp"]))} topics descriptions',
                           current=9, task=task)
        comp_pids = self.df[['id', 'comp']].groupby('comp')['id'].apply(list).to_dict()
        self.topics_description = get_topics_description(
            self.df, comp_pids,
            self.corpus, self.corpus_tokens, self.corpus_counts,
            n_words=self.TOPIC_DESCRIPTION_WORDS
        )
        kwds = [(comp, ','.join([f'{t}:{v:.3f}' for t, v in vs[:self.TOPIC_DESCRIPTION_WORDS]]))
                for comp, vs in self.topics_description.items()]
        self.kwd_df = pd.DataFrame(kwds, columns=['comp', 'kwd'])

        self.progress.info('Identifying top cited papers', current=10, task=task)
        logger.debug('Top cited papers')
        self.top_cited_papers, self.top_cited_df = find_top_cited_papers(self.df, self.TOP_CITED_PAPERS)

        logger.debug('Top cited papers per year')
        self.max_gain_papers, self.max_gain_df = find_max_gain_papers(self.df, self.citation_years)

        logger.debug('Hot paper per year')
        self.max_rel_gain_papers, self.max_rel_gain_df = find_max_relative_gain_papers(
            self.df, self.citation_years
        )

        # Additional analysis steps
        if self.config.feature_authors_enabled:
            self.progress.info("Analyzing authors and groups", current=11, task=task)
            self.author_stats = popular_authors(self.df, n=self.POPULAR_AUTHORS)

        if self.config.feature_journals_enabled:
            self.progress.info("Analyzing popular journals", current=12, task=task)
            self.journal_stats = popular_journals(self.df, n=self.POPULAR_JOURNALS)

        if self.config.feature_numbers_enabled:
            if len(self.df) >= 0:
                self.progress.info('Extracting quantitative features from abstracts texts', current=13, task=task)
                self.numbers_df = extract_numbers(self.df)
            else:
                logger.debug('Not enough papers for numbers extraction')

        if self.config.feature_evolution_enabled:
            if len(self.df) >= PapersAnalyzer.EVOLUTION_MIN_PAPERS:
                self.progress.info(f'Analyzing evolution of topics {self.df["year"].min()} - {self.df["year"].max()}',
                                   current=14, task=task)
                logger.debug('Perform topic evolution analysis and get topic descriptions')
                self.evolution_df, self.evolution_year_range = topic_evolution_analysis(
                    self.df,
                    self.cit_df,
                    self.cocit_grouped_df,
                    self.bibliographic_coupling_df,
                    self.SIMILARITY_COCITATION_MIN,
                    self.similarity,
                    self.corpus_counts,
                    self.corpus_tokens_embedding,
                    PapersAnalyzer.GRAPH_EMBEDDINGS_FACTOR,
                    PapersAnalyzer.TEXT_EMBEDDINGS_FACTOR,
                    topics_max_number,
                    topic_min_size,
                    self.EVOLUTION_STEP
                )
                self.evolution_kwds = topic_evolution_descriptions(
                    self.df, self.evolution_df, self.evolution_year_range,
                    self.corpus, self.corpus_tokens, self.corpus_counts, self.TOPIC_DESCRIPTION_WORDS,
                    self.progress, current=14, task=task
                )
            else:
                logger.debug('Not enough papers for topics evolution')
                self.evolution_df = None
                self.evolution_kwds = None

    @staticmethod
    def prepare_sparse_papers_graph(papers_graph, weighted_similarity_graph):
        sparse_papers_graph = sparse_graph(weighted_similarity_graph, PapersAnalyzer.PAPERS_GRAPH_EDGES_TO_NODES)
        for i, j in sparse_papers_graph.edges():
            d = papers_graph.get_edge_data(i, j)
            for k, v in d.items():
                sparse_papers_graph[i][j][k] = v
            sparse_papers_graph[i][j]['similarity'] = PapersAnalyzer.similarity(d)
        return sparse_papers_graph

    @staticmethod
    def similarity(d):
        return \
            PapersAnalyzer.SIMILARITY_BIBLIOGRAPHIC_COUPLING * np.log1p(d.get('bibcoupling', 0)) + \
            PapersAnalyzer.SIMILARITY_COCITATION * np.log1p(d.get('cocitation', 0)) + \
            PapersAnalyzer.SIMILARITY_CITATION * d.get('citation', 0)

    def dump(self):
        """
        Dump valuable fields to JSON-serializable dict. Use 'load' to restore analyzer.
        """
        return dict(
            df=self.df.to_json(),
            cit_df=self.cit_df.to_json(),
            cocit_grouped_df=self.cocit_grouped_df.to_json(),
            bibliographic_coupling_df=self.bibliographic_coupling_df.to_json(),
            topics_description=self.topics_description,
            kwd_df=self.kwd_df.to_json(),
            pca_coords=self.pca_coords.tolist(),
            sparse_papers_graph=json_graph.node_link_data(self.sparse_papers_graph),
            top_cited_papers=self.top_cited_papers,
            max_gain_papers=self.max_gain_papers,
            max_rel_gain_papers=self.max_rel_gain_papers
        )

    @staticmethod
    def load(fields):
        """
        Load valuable fields from JSON-serializable dict. Use 'dump' to dump analyzer.
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
        cit_df = pd.read_json(fields['cit_df'])
        cit_df['id_in'] = cit_df['id_in'].astype(str)
        cit_df['id_out'] = cit_df['id_out'].astype(str)

        cocit_grouped_df = pd.read_json(fields['cocit_grouped_df'])
        cocit_grouped_df['cited_1'] = cocit_grouped_df['cited_1'].astype(str)
        cocit_grouped_df['cited_2'] = cocit_grouped_df['cited_2'].astype(str)

        bibliographic_coupling_df = pd.read_json(fields['bibliographic_coupling_df'])
        bibliographic_coupling_df['citing_1'] = bibliographic_coupling_df['citing_1'].astype(str)
        bibliographic_coupling_df['citing_2'] = bibliographic_coupling_df['citing_2'].astype(str)

        # Restore topic descriptions
        topics_description = {int(c): ks for c, ks in fields["topics_description"].items()}  # Restore int components
        kwd_df = pd.read_json(fields['kwd_df'])

        # Extra filter is applied to overcome split behaviour problem: split('') = [''] problem
        kwd_df['kwd'] = [kwd.split(',') if kwd != '' else [] for kwd in kwd_df['kwd']]
        kwd_df['kwd'] = kwd_df['kwd'].apply(lambda x: [el.split(':') for el in x])
        kwd_df['kwd'] = kwd_df['kwd'].apply(lambda x: [(el[0], float(el[1])) for el in x])

        # Restore original coordinates
        pca_coords = np.array(fields['pca_coords'])

        # Restore citation and structure graphs
        sparse_papers_graph = json_graph.node_link_graph(fields['sparse_papers_graph'])

        top_cited_papers = fields['top_cited_papers']
        max_gain_papers = fields['max_gain_papers']
        max_rel_gain_papers = fields['max_rel_gain_papers']

        return dict(
            df=df,
            cit_df=cit_df,
            cocit_grouped_df=cocit_grouped_df,
            bibliographic_coupling_df=bibliographic_coupling_df,
            topics_description=topics_description,
            kwd_df=kwd_df,
            pca_coords=pca_coords,
            sparse_papers_graph=sparse_papers_graph,
            top_cited_papers=top_cited_papers,
            max_gain_papers=max_gain_papers,
            max_rel_gain_papers=max_rel_gain_papers
        )

    def init(self, fields):
        """
        Init analyzer with required fields.
        NOTE: results page doesn't use dump/load for visualization.
        Look for init calls in the codebase to find out useful fields for serialization.
        :param fields: deserialized JSON
        """
        logger.debug('Loading analyzer')
        loaded = PapersAnalyzer.load(fields)
        self.df = loaded['df']
        self.cit_df = loaded['cit_df']
        self.cocit_grouped_df = loaded['cocit_grouped_df']
        self.bibliographic_coupling_df = loaded['bibliographic_coupling_df']
        # Used for components naming
        self.topics_description = loaded['topics_description']
        self.kwd_df = loaded['kwd_df']
        # Used for similar papers
        self.pca_coords = loaded['pca_coords']
        # Used for network visualization
        self.sparse_papers_graph = loaded['sparse_papers_graph']
        # Used for navigation
        self.top_cited_papers = loaded['top_cited_papers']
        self.max_gain_papers = loaded['max_gain_papers']
        self.max_rel_gain_papers = loaded['max_rel_gain_papers']
