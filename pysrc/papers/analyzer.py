import hashlib
import logging

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from pysrc.config import *
from pysrc.papers.analysis.citations import find_top_cited_papers, find_max_gain_papers, \
    find_max_relative_gain_papers, build_cit_stats_df, merge_citation_stats, build_cocit_grouped_df
from pysrc.papers.analysis.clustering import cluster_and_sort
from pysrc.papers.analysis.graph import build_papers_graph, sparse_graph, add_text_similarities_edges, \
    similarity
from pysrc.papers.analysis.metadata import get_popular_authors, get_popular_journals
from pysrc.papers.analysis.node2vec import node2vec
from pysrc.papers.analysis.numbers import extract_numbers
from pysrc.papers.analysis.text import embeddings, vectorize_corpus, chunks_to_text_embeddings
from pysrc.papers.compute_or_load import compute_or_load
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
        return 8 + config.feature_authors_enabled + config.feature_journals_enabled + config.feature_numbers_enabled

    def set_current_step(self, step=2):
        self.current_step = step

    def get_step_and_inc(self):
        s = self.current_step
        self.current_step += 1
        return s

    def teardown(self):
        self.progress.remove_handler()

    def search_terms(self, query, limit=None, sort=SORT_MOST_CITED,
                     noreviews=True, min_year=None, max_year=None,
                     task=None):
        limit = limit or SHOW_MAX_ARTICLES_DEFAULT
        # Search articles relevant to the terms
        if len(query) == 0:
            raise SearchError('Empty search string, please use search terms or '
                              'all the query wrapped in "" for phrasal search')
        noreviews_msg = ", not reviews" if noreviews else ""
        min_year_msg = f", after year {min_year}" if min_year else ""
        max_year_msg = f", before year {max_year}" if max_year else ""
        self.progress.info(f'Searching {limit} {sort.lower()} publications matching {query}'
                           f'{noreviews_msg}{min_year_msg}{max_year_msg}',
                           current=1, task=task)
        ids = self.loader.search(query, limit=limit, sort=sort,
                                 noreviews=noreviews, min_year=min_year, max_year=max_year)
        if len(ids) == 0:
            raise SearchError(f"Nothing found for search query: {query}")
        else:
            self.progress.info(f'Found {len(ids)} publications in the database', current=1, task=task)
        return ids

    def analyze_papers(self, ids, topics, test=False, task=None, cache=False):
        self.progress.info('Loading publication data', current=2, task=task)
        ids_key = PapersAnalyzer.ids_key(ids)

        self.df = compute_or_load(
            f"publications_{ids_key}",
            lambda: self.loader.load_publications(ids),
            cache
        )
        self.set_current_step(2)
        if len(self.df) == 0:
            raise SearchError(f'Nothing found for ids: {ids}')
        else:
            self.progress.info(f'Loaded {len(self.df)} papers',
                               current=self.get_step_and_inc(), task=task)
        ids = list(self.df['id'])  # Limit ids to existing papers only!

        self.progress.info('Loading citations for papers',
                           current=self.get_step_and_inc(), task=task)
        logger.debug('Loading citations by year statistics')

        cits_by_year_df = compute_or_load(
            f"citations_by_year_{ids_key}",
            lambda: self.loader.load_citations_by_year(ids),
            cache
        )
        logger.debug(f'Found {len(cits_by_year_df)} records of citations by year')

        self.cit_stats_df = build_cit_stats_df(cits_by_year_df, len(ids))
        if len(self.cit_stats_df) == 0:
            logger.warning('No citations of papers were found')
        self.df, self.citation_years = merge_citation_stats(ids, self.df, self.cit_stats_df)
        logger.debug('Loading citations information')

        self.cit_df = compute_or_load(
            f"citations_{ids_key}",
            lambda: self.loader.load_citations(ids),
            cache
        )
        logger.debug(f'Found {len(self.cit_df)} citations between papers')

        self.progress.info('Calculating co-citations for selected papers',
                           current=self.get_step_and_inc(), task=task)

        self.cocit_df = compute_or_load(
            f"cocitations_{ids_key}",
            lambda: self.loader.load_cocitations(ids),
            cache
        )
        cocit_grouped_df = build_cocit_grouped_df(self.cocit_df)
        logger.debug(f'Found {len(cocit_grouped_df)} co-cited pairs of papers')
        if not test:
            self.cocit_grouped_df = cocit_grouped_df[cocit_grouped_df['total'] >= SIMILARITY_COCITATION_MIN].copy()
            logger.debug(f'Filtered {len(self.cocit_grouped_df)} co-cited pairs of papers, '
                         f'threshold {SIMILARITY_COCITATION_MIN}')
        else:
            self.cocit_grouped_df = cocit_grouped_df

        self.progress.info('Processing references for selected papers',
                           current=self.get_step_and_inc(), task=task)

        bibliographic_coupling_df = compute_or_load(
            f"bibliographic_coupling_{ids_key}",
            lambda: self.loader.load_bibliographic_coupling(ids),
            cache
        )
        logger.debug(f'Found {len(bibliographic_coupling_df)} bibliographic coupling pairs of papers')
        if not test:
            self.bibliographic_coupling_df = bibliographic_coupling_df[
                bibliographic_coupling_df['total'] >= SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN].copy()
            logger.debug(f'Filtered {len(self.bibliographic_coupling_df)} bibliographic coupling pairs of papers, '
                         f'threshold {SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN}')
        else:
            self.bibliographic_coupling_df = bibliographic_coupling_df

        graph = compute_or_load(
            f"graph_bibliometric_{ids_key}",
            lambda: build_papers_graph(self.df, self.cit_df, self.cocit_grouped_df, self.bibliographic_coupling_df),
            cache
        )
        logger.debug(f'Bibliographic edges/nodes='
                     f'{graph.number_of_edges() / graph.number_of_nodes()}')

        self.progress.info('Analyzing title and abstract texts',
                           current=self.get_step_and_inc(), task=task)
        self.corpus, self.corpus_tokens, self.corpus_counts = compute_or_load(
            f"vectorize_corpus_{ids_key}_3",  # Explicitly specify number of outputs
            lambda: vectorize_corpus(
                self.df, max_features=VECTOR_WORDS, min_df=VECTOR_MIN_DF, max_df=VECTOR_MAX_DF, test=test
            ),
            cache
        )

        logger.debug('Analyzing texts embeddings')
        self.chunks_embeddings, self.chunks_idx = compute_or_load(
            f"chunks_embeddings_{ids_key}_2",  # Explicitly specify number of outputs
            lambda: embeddings(
                self.df, self.corpus, self.corpus_tokens, self.corpus_counts, test=test
            ),
            cache
        )
        papers_text_embeddings = compute_or_load(
            f"papers_text_embeddings_{ids_key}",
            lambda: chunks_to_text_embeddings(self.df, self.chunks_embeddings, self.chunks_idx),
            cache
        )

        self.progress.info(f'Analyzing papers citations and text similarity network',
                           current=self.get_step_and_inc(), task=task)
        def add_text_to_graph(graph, papers_text_embeddings):
            add_text_similarities_edges(ids, papers_text_embeddings, graph, GRAPH_TEXT_SIMILARITY_EDGES)
            return graph

        logger.debug('Adding text similarities edges')
        graph = compute_or_load(
            f"graph_with_text_{ids_key}",
            lambda: add_text_to_graph(graph, papers_text_embeddings),
            cache
        )

        logger.debug(f'Bibliographic+text edges/nodes='
                     f'{graph.number_of_edges() / graph.number_of_nodes()}')

        logger.debug('Compute summary graph and text similarity')
        for i, j, data in graph.edges(data=True):
            data['similarity'] = similarity(data)

        logger.debug('Preparing sparse graph for analysis')
        self.papers_graph = compute_or_load(
            f"papers_graph_{ids_key}",
            lambda: sparse_graph(graph, NODE2VEC_GRAPH_EDGES),
            cache
        )

        logger.debug('Analyzing papers graph embeddings')
        graph_embeddings = compute_or_load(
            f"graph_embeddings_{ids_key}",
            lambda: node2vec(self.df['id'], self.papers_graph, 'similarity'),
            cache
        )
        logger.debug('Computing PCA projection')
        pca_coords = compute_or_load(
                f"pca_{ids_key}",
                lambda: PCA(n_components=PCA_VARIANCE, svd_solver="full")\
                    .fit_transform(StandardScaler().fit_transform(graph_embeddings)),
                cache
        )
        if not test and len(self.df) > 1:
            logger.debug('Apply visualization transformation')
            tse_coords = compute_or_load(
                f"tsne_{ids_key}",
                lambda: TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.df) - 1))\
                    .fit_transform(pca_coords),
                cache
            )
            self.df['x'] = tse_coords[:, 0]
            self.df['y'] = tse_coords[:, 1]
        else:
            self.df['x'] = 0
            self.df['y'] = 0

        logger.debug('Extracting topics from papers embeddings')
        self.clusters, self.dendrogram = compute_or_load(
            f"clusters_{ids_key}_2",
            lambda: cluster_and_sort(pca_coords, topics),
            cache
        )
        self.df['comp'] = self.clusters

        self.progress.info('Identifying top cited papers',
                           current=self.get_step_and_inc(), task=task)
        logger.debug('Top cited papers')
        self.top_cited_df = find_top_cited_papers(self.df, TOP_CITED_PAPERS)

        logger.debug('Top cited papers per year')
        self.max_gain_df = find_max_gain_papers(self.df, self.citation_years)

        logger.debug('Hot paper per year')
        self.max_rel_gain_df = find_max_relative_gain_papers(self.df, self.citation_years)

        # Additional analysis steps
        self.author_stats = None
        if self.config.feature_authors_enabled:
            self.progress.info("Analyzing authors and groups",
                               current=self.get_step_and_inc(), task=task)
            self.author_stats = get_popular_authors(self.df, n=POPULAR_AUTHORS)

        self.journal_stats = None
        if self.config.feature_journals_enabled:
            self.progress.info("Analyzing popular journals",
                               current=self.get_step_and_inc(), task=task)
            self.journal_stats = get_popular_journals(self.df, n=POPULAR_JOURNALS)

        self.numbers_df = None
        if self.config.feature_numbers_enabled:
            if len(self.df) >= 0:
                self.progress.info('Extracting quantitative features from abstracts texts',
                                   current=self.get_step_and_inc(), task=task)
                self.numbers_df = extract_numbers(self.df)
            else:
                logger.debug('Not enough papers for numbers extraction')
        logger.debug('Analysis finished')

    @staticmethod
    def ids_key(ids) -> str:
        return hashlib.md5(str(sorted(ids)).encode()).hexdigest()

    def save(
            self,
            analysis_type,
            search_ids,
            search_query,
            source,
            sort,
            limit,
            noreviews,
            min_year,
            max_year
    ) -> AnalysisData:
        return AnalysisData(
            analysis_type=analysis_type,
            search_query=search_query,
            search_ids=search_ids,
            source=source,
            sort=sort,
            limit=limit,
            noreviews=noreviews,
            min_year=min_year,
            max_year=max_year,
            df=self.df,
            cit_df=self.cit_df,
            cocit_grouped_df=self.cocit_grouped_df,
            bibliographic_coupling_df=self.bibliographic_coupling_df,
            top_cited_df=self.top_cited_df,
            max_gain_df=self.max_gain_df,
            max_rel_gain_df=self.max_rel_gain_df,
            dendrogram=self.dendrogram,
            corpus=self.corpus,
            corpus_tokens=self.corpus_tokens,
            corpus_counts=self.corpus_counts,
            chunks_embeddings=self.chunks_embeddings,
            chunks_idx=self.chunks_idx,
            papers_graph=self.papers_graph,
            author_stats=self.author_stats,
            journal_stats=self.journal_stats,
            numbers_df=self.numbers_df,
        )
