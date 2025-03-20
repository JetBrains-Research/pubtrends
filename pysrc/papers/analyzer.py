import json
import logging
import numpy as np
import pandas as pd
from io import StringIO
from networkx.readwrite import json_graph
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from pysrc.papers.analysis.citations import find_top_cited_papers, find_max_gain_papers, \
    find_max_relative_gain_papers, build_cit_stats_df, merge_citation_stats, build_cocit_grouped_df
from pysrc.papers.analysis.graph import build_papers_graph, sparse_graph, add_artificial_text_similarities_edges
from pysrc.papers.analysis.metadata import popular_authors, popular_journals
from pysrc.papers.analysis.node2vec import node2vec
from pysrc.papers.analysis.numbers import extract_numbers
from pysrc.papers.analysis.text import texts_embeddings, vectorize_corpus, tokens_embeddings
from pysrc.papers.analysis.topics import get_topics_description, cluster_and_sort
from pysrc.papers.config import *
from pysrc.papers.data import AnalysisData
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.progress import Progress
from pysrc.papers.utils import SORT_MOST_CITED

logger = logging.getLogger(__name__)

class PapersAnalyzer:

    def __init__(self, loader, config, test=False):
        self.config = config
        self.progress = Progress(self.total_steps(self.config))

        self.loader = loader
        self.source = Loaders.source(self.loader, test)

    def total_steps(self, config):
        return 11 + config.feature_authors_enabled + config.feature_journals_enabled + config.feature_numbers_enabled

    def set_current_step(self, step=2):
        self.current_step = step

    def get_step_and_inc(self):
        s = self.current_step
        self.current_step += 1
        return s

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

    def analyze_papers(self, ids, query, source, sort, limit, topics, test=False, task=None):
        self.progress.info('Loading publication data', current=2, task=task)
        self.query = query
        self.source = source
        self.sort = sort
        self.limit = limit
        self.topics = topics
        self.df = self.loader.load_publications(ids)
        self.set_current_step(2)
        if len(self.df) == 0:
            raise SearchError(f'Nothing found for ids: {ids}')
        else:
            self.progress.info(f'Total {len(self.df)} papers in database',
                               current=self.get_step_and_inc(), task=task)
        ids = list(self.df['id'])  # Limit ids to existing papers only!
        self.progress.info('Analyzing title and abstract texts',
                           current=self.get_step_and_inc(), task=task)
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
            papers_text_embeddings = texts_embeddings(
                self.corpus_counts, self.corpus_tokens_embedding
            )
        else:
            papers_text_embeddings = np.zeros(shape=(len(self.df), EMBEDDINGS_VECTOR_LENGTH))

        self.progress.info('Loading citations for papers',
                           current=self.get_step_and_inc(), task=task)
        logger.debug('Loading citations by year statistics')
        cits_by_year_df = self.loader.load_citations_by_year(ids)
        logger.debug(f'Found {len(cits_by_year_df)} records of citations by year')

        self.cit_stats_df = build_cit_stats_df(cits_by_year_df, len(ids))
        if len(self.cit_stats_df) == 0:
            logger.warning('No citations of papers were found')
        self.df, self.citation_years = merge_citation_stats(ids, self.df, self.cit_stats_df)
        logger.debug('Loading citations information')
        self.cit_df = self.loader.load_citations(ids)
        logger.debug(f'Found {len(self.cit_df)} citations between papers')

        self.progress.info('Calculating co-citations for selected papers',
                           current=self.get_step_and_inc(), task=task)
        self.cocit_df = self.loader.load_cocitations(ids)
        cocit_grouped_df = build_cocit_grouped_df(self.cocit_df)
        logger.debug(f'Found {len(cocit_grouped_df)} co-cited pairs of papers')
        if not test:
            self.cocit_grouped_df = cocit_grouped_df[cocit_grouped_df['total'] >= SIMILARITY_COCITATION_MIN].copy()
            logger.debug(f'Filtered {len(self.cocit_grouped_df)} co-cited pairs of papers, '
                         f'threshold {SIMILARITY_COCITATION_MIN}')
        else:
            self.cocit_grouped_df = cocit_grouped_df

        self.progress.info('Processing bibliographic coupling for selected papers',
                           current=self.get_step_and_inc(), task=task)
        bibliographic_coupling_df = self.loader.load_bibliographic_coupling(ids)
        logger.debug(f'Found {len(bibliographic_coupling_df)} bibliographic coupling pairs of papers')
        if not test:
            self.bibliographic_coupling_df = bibliographic_coupling_df[
                bibliographic_coupling_df['total'] >= SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN].copy()
            logger.debug(f'Filtered {len(self.bibliographic_coupling_df)} bibliographic coupling pairs of papers '
                         f'threshold {SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN}')
        else:
            self.bibliographic_coupling_df = bibliographic_coupling_df

        full_papers_graph = build_papers_graph(
            self.df, self.cit_df, self.cocit_grouped_df, self.bibliographic_coupling_df,
        )
        self.progress.info(f'Analyzing papers graph - {full_papers_graph.number_of_nodes()} nodes and '
                           f'{full_papers_graph.number_of_edges()} edges',
                           current=self.get_step_and_inc(), task=task)

        if GRAPH_EMBEDDINGS_FACTOR != 0:
            logger.debug('Analyzing papers graph embeddings')
            papers_graph_embeddings = node2vec(
                self.df['id'],
                sparse_graph(full_papers_graph, EMBEDDINGS_GRAPH_EDGES),
                key='similarity'
            )
        else:
            papers_graph_embeddings = np.zeros(shape=(len(self.df), 0))

        logger.debug('Computing aggregated graph and text embeddings for papers')
        self.papers_embeddings = (papers_graph_embeddings * GRAPH_EMBEDDINGS_FACTOR +
                                  papers_text_embeddings * TEXT_EMBEDDINGS_FACTOR
                                  ) / (GRAPH_EMBEDDINGS_FACTOR + TEXT_EMBEDDINGS_FACTOR)

        logger.debug('Prepare sparse graph to visualize')
        self.papers_graph = sparse_graph(full_papers_graph, GRAPH_BIBLIOGRAPHIC_EDGES)

        if TEXT_EMBEDDINGS_FACTOR != 0:
            logger.debug('Adding artificial text similarities edges for visualization purposes')
            add_artificial_text_similarities_edges(ids, papers_text_embeddings, self.papers_graph)

        if len(self.df) > 1:
            logger.debug('Computing PCA projection')
            pca = PCA(n_components=min(len(self.papers_embeddings), PCA_COMPONENTS))
            t = StandardScaler().fit_transform(self.papers_embeddings)
            pca_coords = pca.fit_transform(t)
            logger.debug(f'Explained variation {int(np.sum(pca.explained_variance_ratio_) * 100)}%')
            logger.debug('Apply TSNE transformation')
            if not test:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.df) - 1))
            else:
                tsne = TSNE(n_components=2, random_state=42, perplexity=3)
            tsne_embeddings_2d = tsne.fit_transform(pca_coords)
            self.df['x'] = tsne_embeddings_2d[:, 0]
            self.df['y'] = tsne_embeddings_2d[:, 1]
        else:
            pca_coords = np.zeros(shape=(len(self.df), PCA_COMPONENTS))
            self.df['x'] = 0
            self.df['y'] = 0

        self.progress.info(f'Extracting {topics} number of topics from papers text and graph similarity',
                           current=8, task=task)
        logger.debug('Extracting topics from papers embeddings')
        self.clusters, self.dendrogram = cluster_and_sort(pca_coords, topics)
        self.df['comp'] = self.clusters

        self.progress.info(f'Analyzing {len(set(self.df["comp"]))} topics descriptions',
                           current=self.get_step_and_inc(), task=task)
        comp_pids = self.df[['id', 'comp']].groupby('comp')['id'].apply(list).to_dict()
        self.topics_description = get_topics_description(
            self.df, comp_pids,
            self.corpus, self.corpus_tokens, self.corpus_counts,
            n_words=self.config.topic_description_words,
        )

        self.progress.info('Identifying top cited papers',
                           current=self.get_step_and_inc(), task=task)
        logger.debug('Top cited papers')
        self.top_cited_df = find_top_cited_papers(self.df, self.config.top_cited_papers)

        logger.debug('Top cited papers per year')
        self.max_gain_df = find_max_gain_papers(self.df, self.citation_years)

        logger.debug('Hot paper per year')
        self.max_rel_gain_df = find_max_relative_gain_papers(self.df, self.citation_years)

        # Additional analysis steps
        self.author_stats = None
        if self.config.feature_authors_enabled:
            self.progress.info("Analyzing authors and groups",
                               current=self.get_step_and_inc(), task=task)
            self.author_stats = popular_authors(self.df, n=self.config.popular_authors)

        self.journal_stats = None
        if self.config.feature_journals_enabled:
            self.progress.info("Analyzing popular journals",
                               current=self.get_step_and_inc(), task=task)
            self.journal_stats = popular_journals(self.df, n=self.config.popular_journals)

        self.numbers_df = None
        if self.config.feature_numbers_enabled:
            if len(self.df) >= 0:
                self.progress.info('Extracting quantitative features from abstracts texts',
                                   current=self.get_step_and_inc(), task=task)
                self.numbers_df = extract_numbers(self.df)
            else:
                logger.debug('Not enough papers for numbers extraction')
        self.progress.done('Done', task=task)

    def save(self) -> AnalysisData:
        return AnalysisData(
            query=self.query,
            source=self.source,
            sort=self.sort,
            limit=self.limit,
            df=self.df,
            cit_df=self.cit_df,
            cocit_grouped_df=self.cocit_grouped_df,
            bibliographic_coupling_df=self.bibliographic_coupling_df,
            top_cited_df=self.top_cited_df,
            max_gain_df=self.max_gain_df,
            max_rel_gain_df=self.max_rel_gain_df,
            dendrogram=self.dendrogram,
            topics_description=self.topics_description,
            corpus=self.corpus,
            corpus_tokens=self.corpus_tokens,
            corpus_counts=self.corpus_counts,
            papers_graph=self.papers_graph,
            papers_embeddings=self.papers_embeddings,
            author_stats=self.author_stats,
            journal_stats=self.journal_stats,
            numbers_df=self.numbers_df,
        )
