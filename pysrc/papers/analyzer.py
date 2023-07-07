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
from pysrc.papers.analysis.graph import build_papers_graph, sparse_graph, similarity
from pysrc.papers.analysis.metadata import popular_authors, popular_journals
from pysrc.papers.analysis.node2vec import node2vec
from pysrc.papers.analysis.numbers import extract_numbers
from pysrc.papers.analysis.text import texts_embeddings, vectorize_corpus, tokens_embeddings
from pysrc.papers.analysis.topics import get_topics_description, cluster_and_sort
from pysrc.papers.config import *
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.progress import Progress
from pysrc.papers.utils import SORT_MOST_CITED

logger = logging.getLogger(__name__)


class PapersAnalyzer:

    def __init__(self, loader, config, test=False):
        self.config = config
        self.progress = Progress(self.total_steps())

        self.loader = loader
        self.source = Loaders.source(self.loader, test)

    def total_steps(self):
        return 14 + 1  # One extra step for visualization

    def teardown(self):
        self.progress.remove_handler()

    def search_terms(self, query, limit=None, sort=SORT_MOST_CITED, noreviews=True, task=None):
        limit = limit or self.config.show_max_articles_default_value
        # Search articles relevant to the terms
        if len(query) == 0:
            raise SearchError('Empty search string, please use search terms or '
                              'all the query wrapped in "" for phrasal search')
        noreviews_msg = ", not reviews" if noreviews else ""
        self.progress.info(f'Searching {limit} {sort.lower()} publications matching {query}{noreviews_msg}',
                           current=1, task=task)
        ids = self.loader.search(query, limit=limit, sort=sort, noreviews=noreviews)
        if len(ids) == 0:
            raise SearchError(f"Nothing found for search query: {query}")
        else:
            self.progress.info(f'Found {len(ids)} publications in the database', current=1, task=task)
        return ids

    def analyze_papers(self, ids, query, topics=TOPICS_NUMBER_MEDIUM, test=False, task=None):
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
            max_features=VECTOR_WORDS,
            min_df=VECTOR_MIN_DF,
            max_df=VECTOR_MAX_DF,
            test=test
        )
        if TEXT_EMBEDDINGS_FACTOR != 0:
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
        if not test:
            self.cocit_grouped_df = cocit_grouped_df[cocit_grouped_df['total'] >= SIMILARITY_COCITATION_MIN].copy()
            logger.debug(f'Filtered {len(self.cocit_grouped_df)} co-cited pairs of papers, '
                         f'threshold {SIMILARITY_COCITATION_MIN}')
        else:
            self.cocit_grouped_df = cocit_grouped_df

        self.progress.info('Processing bibliographic coupling for selected papers', current=6, task=task)
        bibliographic_coupling_df = self.loader.load_bibliographic_coupling(ids)
        logger.debug(f'Found {len(bibliographic_coupling_df)} bibliographic coupling pairs of papers')
        if not test:
            self.bibliographic_coupling_df = bibliographic_coupling_df[
                bibliographic_coupling_df['total'] >= SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN].copy()
            logger.debug(f'Filtered {len(self.bibliographic_coupling_df)} bibliographic coupling pairs of papers '
                         f'threshold {SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN}')
        else:
            self.bibliographic_coupling_df = bibliographic_coupling_df

        self.progress.info('Building papers graph', current=7, task=task)
        self.papers_graph = build_papers_graph(
            self.df, self.cit_df, self.cocit_grouped_df, self.bibliographic_coupling_df,
        )
        self.progress.info(f'Analyzing papers graph - {self.papers_graph.number_of_nodes()} nodes and '
                           f'{self.papers_graph.number_of_edges()} edges', current=7, task=task)
        if GRAPH_EMBEDDINGS_FACTOR != 0:
            logger.debug('Prepare sparse graph for embedding')
            sge = sparse_graph(self.papers_graph, EMBEDDINGS_SPARSE_GRAPH_EDGES_TO_NODES)
            # Add similarity key to sparse graph
            for i, j in sge.edges():
                sge[i][j]['similarity'] = similarity(self.papers_graph.get_edge_data(i, j))
            logger.debug('Analyzing papers graph embeddings')
            self.graph_embeddings = node2vec(self.df['id'], sge, key='similarity')
        else:
            self.graph_embeddings = np.zeros(shape=(len(self.df), 0))

        logger.debug('Computing aggregated graph and text embeddings for papers')
        papers_embeddings = np.concatenate(
            (self.graph_embeddings * GRAPH_EMBEDDINGS_FACTOR,
             self.texts_embeddings * TEXT_EMBEDDINGS_FACTOR), axis=1)

        if len(self.df) > 1:
            logger.debug('Computing PCA projection')
            pca = PCA(n_components=min(len(papers_embeddings), PCA_COMPONENTS))
            t = StandardScaler().fit_transform(papers_embeddings)
            self.pca_coords = pca.fit_transform(t)
            logger.debug(f'Explained variation {int(np.sum(pca.explained_variance_ratio_) * 100)}%')
            logger.debug('Apply TSNE transformation on papers PCA coords')
            if not test:
                tsne_embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(self.pca_coords)
            else:
                tsne_embeddings_2d = TSNE(n_components=2, random_state=42, perplexity=3).fit_transform(self.pca_coords)
            self.df['x'] = tsne_embeddings_2d[:, 0]
            self.df['y'] = tsne_embeddings_2d[:, 1]
        else:
            self.pca_coords = np.zeros(shape=(len(self.df), 128))
            self.df['x'] = 0
            self.df['y'] = 0

        self.progress.info(f'Extracting {topics} number of topics from papers text and graph similarity',
                           current=8, task=task)
        logger.debug('Extracting topics from papers embeddings')
        self.clusters, self.dendrogram = cluster_and_sort(self.pca_coords, topics)
        self.df['comp'] = self.clusters

        self.progress.info(f'Analyzing {len(set(self.df["comp"]))} topics descriptions',
                           current=9, task=task)
        comp_pids = self.df[['id', 'comp']].groupby('comp')['id'].apply(list).to_dict()
        self.topics_description = get_topics_description(
            self.df, comp_pids,
            self.corpus, self.corpus_tokens, self.corpus_counts,
            n_words=TOPIC_DESCRIPTION_WORDS
        )
        kwds = [(comp, ','.join([f'{t}:{v:.3f}' for t, v in vs[:TOPIC_DESCRIPTION_WORDS]]))
                for comp, vs in self.topics_description.items()]
        self.kwd_df = pd.DataFrame(kwds, columns=['comp', 'kwd'])

        self.progress.info('Identifying top cited papers', current=10, task=task)
        logger.debug('Top cited papers')
        self.top_cited_papers, self.top_cited_df = find_top_cited_papers(self.df, TOP_CITED_PAPERS)

        logger.debug('Top cited papers per year')
        self.max_gain_papers, self.max_gain_df = find_max_gain_papers(self.df, self.citation_years)

        logger.debug('Hot paper per year')
        self.max_rel_gain_papers, self.max_rel_gain_df = find_max_relative_gain_papers(
            self.df, self.citation_years
        )

        logger.debug('Prepare sparse graph for visualization')
        self.sparse_papers_graph = sparse_graph(self.papers_graph, VISUALIZATION_SPARSE_GRAPH_EDGES_TO_NODES)
        # Add similarity key to sparse graph
        for i, j in self.sparse_papers_graph.edges():
            self.sparse_papers_graph[i][j]['similarity'] = similarity(self.papers_graph.get_edge_data(i, j))

        # Additional analysis steps
        if self.config.feature_authors_enabled:
            self.progress.info("Analyzing authors and groups", current=11, task=task)
            self.author_stats = popular_authors(self.df, n=POPULAR_AUTHORS)

        if self.config.feature_journals_enabled:
            self.progress.info("Analyzing popular journals", current=12, task=task)
            self.journal_stats = popular_journals(self.df, n=POPULAR_JOURNALS)

        if self.config.feature_numbers_enabled:
            if len(self.df) >= 0:
                self.progress.info('Extracting quantitative features from abstracts texts', current=13, task=task)
                self.numbers_df = extract_numbers(self.df)
            else:
                logger.debug('Not enough papers for numbers extraction')

        if self.config.feature_evolution_enabled:
            if len(self.df) >= EVOLUTION_MIN_PAPERS:
                self.progress.info(f'Analyzing evolution of topics {self.df["year"].min()} - {self.df["year"].max()}',
                                   current=14, task=task)
                logger.debug('Perform topic evolution analysis and get topic descriptions')
                self.evolution_df, self.evolution_year_range = topic_evolution_analysis(
                    self.df,
                    self.cit_df,
                    self.cocit_grouped_df,
                    self.bibliographic_coupling_df,
                    self.corpus_counts,
                    self.corpus_tokens_embedding,
                    topics,
                )
                self.evolution_kwds = topic_evolution_descriptions(
                    self.df, self.evolution_df, self.evolution_year_range,
                    self.corpus, self.corpus_tokens, self.corpus_counts, TOPIC_DESCRIPTION_WORDS,
                    self.progress, current=14, task=task
                )
            else:
                logger.debug('Not enough papers for topics evolution')
                self.evolution_df = None
                self.evolution_kwds = None


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
