import logging

import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph

from pysrc.papers.analysis.citations import find_top_cited_papers, find_max_gain_papers, \
    find_max_relative_gain_papers, build_cit_stats_df, merge_citation_stats, build_cocit_grouped_df
from pysrc.papers.analysis.evolution import topic_evolution_analysis, topic_evolution_descriptions
from pysrc.papers.analysis.graph import build_citation_graph, build_similarity_graph, build_structure_graph
from pysrc.papers.analysis.metadata import popular_authors, popular_journals
from pysrc.papers.analysis.numbers import extract_numbers
from pysrc.papers.analysis.text import analyze_texts_similarity, vectorize_corpus
from pysrc.papers.analysis.topics import topic_analysis, get_topics_description
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.progress import Progress
from pysrc.papers.utils import SORT_MOST_CITED

logger = logging.getLogger(__name__)


class PapersAnalyzer:
    TOP_CITED_PAPERS = 50

    # ...bibliographic coupling (BC) was the most accurate,  followed by co-citation (CC).
    # Direct citation (DC) was a distant third among the three...
    SIMILARITY_BIBLIOGRAPHIC_COUPLING = 0.125  # Limited by number of references, applied to log
    SIMILARITY_COCITATION = 1  # Limiter by number of co-citations, applied to log
    SIMILARITY_CITATION = 0.125  # Limited by 1 citation
    SIMILARITY_TEXT_CITATION = 1  # Limited by cosine similarity <= 1

    # Minimal number of common references, used to reduces similarity graph edges count
    # Value > 1 is especially useful while analysing single paper, removes meaningless connections by construction
    SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN = 1

    # Minimal number of common references, used to reduces similarity graph edges count
    SIMILARITY_COCITATION_MIN = 1

    # Minimal cosine similarity for potential text citation, used to reduce similarity graph edges count
    SIMILARITY_TEXT_CITATION_MIN = 0.1

    # Max number of potential text citations for paper
    SIMILARITY_TEXT_CITATION_N = 20

    # Reduce number of edges in smallest communities, i.e. topics
    STRUCTURE_LOW_LEVEL_SPARSITY = 0.5
    # Limit number of edges between different topics to min number of inner edges * scale factor,
    # ignoring similarities less than average within groups
    STRUCTURE_BETWEEN_TOPICS_SPARSITY = 0.3

    # Global vectorization max vocabulary size
    VECTOR_WORDS = 10000
    # Terms with lower frequency will be ignored, remove rare words
    VECTOR_MIN_DF = 0.01
    # Terms with higher frequency will be ignored, remove abundant stop words
    VECTOR_MAX_DF = 0.8

    TOPIC_MIN_SIZE = 10
    TOPICS_MAX_NUMBER = 100

    # Number of top cited papers in topic picked for description computation
    TOPIC_MOST_CITED_PAPERS = 50
    # Number of words for topic description
    TOPIC_DESCRIPTION_WORDS = 10

    POPULAR_JOURNALS = 50
    POPULAR_AUTHORS = 50

    # Max expand before filtration by citations and keywords
    EXPAND_LIMIT = 10000
    # Control citations count
    EXPAND_CITATIONS_Q_LOW = 5
    EXPAND_CITATIONS_Q_HIGH = 95
    EXPAND_CITATIONS_SIGMA = 3
    # Take up to fraction of top similarity
    EXPAND_SIMILARITY_THRESHOLD = 0.3
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
        self.progress.info(f'Searching {limit} {sort.lower()} publications matching {query}',
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

    def load_references(self, pid, limit):
        logger.debug('Loading direct references for paper analysis')
        references = self.loader.load_references(pid, limit)
        logger.debug(f'Loaded {len(references)} references')
        return references

    def analyze_papers(self, ids, query, task=None):
        """:return full log"""
        self.progress.info('Loading publication data', current=2, task=task)
        self.query = query
        self.pub_df = self.loader.load_publications(ids)
        if len(self.pub_df) == 0:
            raise SearchError(f'Nothing found for ids: {ids}')
        self.ids = list(self.pub_df['id'])  # Limit ids to existing papers only!
        self.n_papers = len(self.ids)
        self.pub_types = list(set(self.pub_df['type']))

        self.progress.info('Analyzing title and abstract texts', current=3, task=task)
        self.corpus_terms, self.corpus_counts = vectorize_corpus(
            self.pub_df,
            max_features=PapersAnalyzer.VECTOR_WORDS,
            min_df=PapersAnalyzer.VECTOR_MIN_DF,
            max_df=PapersAnalyzer.VECTOR_MAX_DF
        )

        self.progress.info('Processing texts similarity', current=4, task=task)
        self.texts_similarity = analyze_texts_similarity(
            self.pub_df, self.corpus_counts,
            self.SIMILARITY_TEXT_CITATION_MIN, self.SIMILARITY_TEXT_CITATION_N
        )

        self.progress.info('Loading citations statistics by year', current=5, task=task)
        cits_by_year_df = self.loader.load_citations_by_year(self.ids)
        self.progress.info(f'Found {len(cits_by_year_df)} records of citations by year',
                           current=5, task=task)

        self.cit_stats_df = build_cit_stats_df(cits_by_year_df, self.n_papers)
        if len(self.cit_stats_df) == 0:
            logger.warning('No citations of papers were found')
        self.df, self.citation_years = merge_citation_stats(self.pub_df, self.cit_stats_df)
        self.min_year, self.max_year = self.df['year'].min(), self.df['year'].max()

        # Load data about citations between given papers (excluding outer papers)
        # IMPORTANT: cit_df may contain not all the publications for query
        self.progress.info('Loading citations data', current=6, task=task)
        self.cit_df = self.loader.load_citations(self.ids)
        self.progress.info(f'Found {len(self.cit_df)} citations between papers', current=5, task=task)

        self.progress.info('Building citation graph', current=7, task=task)
        self.citations_graph = build_citation_graph(self.cit_df)
        self.progress.info(f'Built citation graph - {len(self.citations_graph.nodes())} nodes and '
                           f'{len(self.citations_graph.edges())} edges',
                           current=7, task=task)

        self.progress.info('Calculating co-citations for selected papers', current=8, task=task)
        self.cocit_df = self.loader.load_cocitations(self.ids)
        cocit_grouped_df = build_cocit_grouped_df(self.cocit_df)
        self.progress.info(f'Found {len(cocit_grouped_df)} co-cited pairs of papers', current=8, task=task)
        self.cocit_grouped_df = cocit_grouped_df[cocit_grouped_df['total'] >= self.SIMILARITY_COCITATION_MIN].copy()
        self.progress.info(f'Filtering co-citations with min threshold '
                           f'{self.SIMILARITY_COCITATION_MIN}', current=8, task=task)
        self.progress.info(f'Filtered {len(self.cocit_grouped_df)} co-cited pairs of papers', current=8, task=task)

        self.progress.info('Processing bibliographic coupling for selected papers', current=9, task=task)
        bibliographic_coupling_df = self.loader.load_bibliographic_coupling(self.ids)
        self.progress.info(f'Found {len(bibliographic_coupling_df)} bibliographic coupling pairs of papers',
                           current=9, task=task)
        self.progress.info(f'Filtering bibliographic coupling with min threshold '
                           f'{self.SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN}', current=9, task=task)
        self.bibliographic_coupling_df = bibliographic_coupling_df[
            bibliographic_coupling_df['total'] >= self.SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN].copy()
        self.progress.info(f'Filtered {len(self.bibliographic_coupling_df)} bibliographic coupling pairs of papers',
                           current=9, task=task)

        self.progress.info('Building papers similarity graph', current=10, task=task)
        self.similarity_graph = build_similarity_graph(
            self.df, self.texts_similarity,
            self.citations_graph, self.cocit_grouped_df, self.bibliographic_coupling_df,
            process_all_papers=True,
        )
        self.progress.info(f'Built similarity graph - {len(self.similarity_graph.nodes())} nodes and '
                           f'{len(self.similarity_graph.edges())} edges',
                           current=10, task=task)

        if len(self.similarity_graph.nodes()) == 0:
            self.progress.info('Not enough papers to process topics analysis', current=11, task=task)
            self.df['comp'] = 0  # Technical value for top authors and papers analysis
            self.kwd_df = pd.DataFrame({'comp': [0], 'kwd': ['']})
            self.structure_graph = nx.Graph()
        else:
            self.progress.info('Extracting topics from paper similarity graph', current=11, task=task)
            self.topics_dendrogram, partition, self.comp_other, self.components, self.comp_sizes = \
                topic_analysis(
                    self.similarity_graph,
                    similarity_func=self.similarity,
                    topic_min_size=self.TOPIC_MIN_SIZE,
                    max_topics_number=self.TOPICS_MAX_NUMBER
                )
            self.df = self.add_to_dataframe(self.df, partition, col='comp', na=-1)

            self.progress.info('Computing topics descriptions', current=12, task=task)
            comp_pids = pd.DataFrame(partition.items(), columns=['id', 'comp']). \
                groupby('comp')['id'].apply(list).to_dict()
            topics_description = get_topics_description(
                self.df, comp_pids,
                self.corpus_terms, self.corpus_counts,
                query=query,
                n_words=self.TOPIC_DESCRIPTION_WORDS
            )
            kwds = [(comp, ','.join([f'{t}:{v:.3f}' for t, v in vs[:self.TOPIC_DESCRIPTION_WORDS]]))
                    for comp, vs in topics_description.items()]
            self.kwd_df = pd.DataFrame(kwds, columns=['comp', 'kwd'])
            # Build structure graph
            self.structure_graph = build_structure_graph(
                self.df,
                self.similarity_graph,
                similarity_func=self.similarity,
                comp_sparsity=self.STRUCTURE_LOW_LEVEL_SPARSITY,
                between_comp_sparsity=self.STRUCTURE_BETWEEN_TOPICS_SPARSITY
            )

        self.progress.info('Performing PageRank analysis', current=14, task=task)
        pr = nx.pagerank(self.citations_graph, alpha=0.5, tol=1e-9)
        self.df = self.add_to_dataframe(self.df, pr, col='pagerank', na=0.0)

        self.progress.info('Identifying top cited papers', current=15, task=task)
        self.top_cited_papers, self.top_cited_df = find_top_cited_papers(self.df, self.TOP_CITED_PAPERS)

        self.progress.info('Identifying top cited papers for each year', current=16, task=task)
        self.max_gain_papers, self.max_gain_df = find_max_gain_papers(self.df, self.citation_years)

        self.progress.info('Identifying hot papers of the year', current=17, task=task)
        self.max_rel_gain_papers, self.max_rel_gain_df = find_max_relative_gain_papers(
            self.df, self.citation_years
        )

        # Additional analysis steps
        if self.config.feature_authors_enabled:
            self.progress.info("Finding popular authors", current=18, task=task)
            self.author_stats = popular_authors(self.df, n=self.POPULAR_AUTHORS)

        if self.config.feature_journals_enabled:
            self.progress.info("Finding popular journals", current=19, task=task)
            self.journal_stats = popular_journals(self.df, n=self.POPULAR_JOURNALS)

        if self.config.feature_numbers_enabled:
            if len(self.df) >= 0:
                self.progress.info('Extracting numbers from publication abstracts', current=20, task=task)
                self.numbers_df = extract_numbers(self.df)
            else:
                logger.debug('Not enough papers for numbers extraction')

        if self.config.feature_evolution_enabled:
            if len(self.df) >= PapersAnalyzer.EVOLUTION_MIN_PAPERS:
                logger.debug('Perform topic evolution analysis and get topic descriptions')
                self.evolution_df, self.evolution_year_range = topic_evolution_analysis(
                    self.df, self.cit_df, self.cocit_df, self.bibliographic_coupling_df,
                    self.texts_similarity, self.SIMILARITY_COCITATION_MIN,
                    self.TOPIC_MIN_SIZE,
                    self.TOPICS_MAX_NUMBER,
                    similarity_func=self.similarity,
                    evolution_step=self.EVOLUTION_STEP,
                    progress=self.progress, current=21, task=task
                )
                self.evolution_kwds = topic_evolution_descriptions(
                    self.df, self.evolution_df, self.evolution_year_range,
                    self.corpus_terms, self.corpus_counts, self.TOPIC_DESCRIPTION_WORDS,
                    self.progress, current=21, task=task
                )
            else:
                logger.debug('Not enough papers for topics evolution')
                self.evolution_df = None
                self.evolution_kwds = None

    @staticmethod
    def similarity(d):
        return \
            PapersAnalyzer.SIMILARITY_BIBLIOGRAPHIC_COUPLING * np.log1p(d.get('bibcoupling', 0)) + \
            PapersAnalyzer.SIMILARITY_COCITATION * np.log1p(d.get('cocitation', 0)) + \
            PapersAnalyzer.SIMILARITY_CITATION * d.get('citation', 0) + \
            PapersAnalyzer.SIMILARITY_TEXT_CITATION * d.get('text', 0)

    @staticmethod
    def add_to_dataframe(df, data, col, na):
        assert col not in df, f'Column {col} already is in dataframe'
        t = pd.Series(data).reset_index().rename(columns={'index': 'id', 0: col})
        t['id'] = t['id'].astype(str)
        df_merged = pd.merge(df, t, on='id', how='outer')
        df_merged[col] = df_merged[col].fillna(na)
        return df_merged

    def dump(self):
        """
        Dump valuable fields to JSON-serializable dict. Use 'load' to restore analyzer.
        """
        return {
            'df': self.df.to_json(),
            'comp_other': self.comp_other,
            'kwd_df': self.kwd_df.to_json(),
            'citations_graph': json_graph.node_link_data(self.citations_graph),
            'similarity_graph': json_graph.node_link_data(self.similarity_graph),
            'structure_graph': json_graph.node_link_data(self.structure_graph),
            'top_cited_papers': self.top_cited_papers,
            'max_gain_papers': self.max_gain_papers,
            'max_rel_gain_papers': self.max_rel_gain_papers,
        }

    @staticmethod
    def load(fields):
        """
        Load valuable fields from JSON-serializable dict. Use 'dump' to dump analyzer.
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
        kwd_df = pd.read_json(fields['kwd_df'])

        # Extra filter is applied to overcome split behaviour problem: split('') = [''] problem
        kwd_df['kwd'] = [kwd.split(',') if kwd != '' else [] for kwd in kwd_df['kwd']]
        kwd_df['kwd'] = kwd_df['kwd'].apply(lambda x: [el.split(':') for el in x])
        kwd_df['kwd'] = kwd_df['kwd'].apply(lambda x: [(el[0], float(el[1])) for el in x])

        # Restore citation and structure graphs
        citations_graph = json_graph.node_link_graph(fields['citations_graph'])
        similarity_graph = json_graph.node_link_graph(fields['similarity_graph'])
        structure_graph = json_graph.node_link_graph(fields['structure_graph'])

        top_cited_papers = fields['top_cited_papers']
        max_gain_papers = fields['max_gain_papers']
        max_rel_gain_papers = fields['max_rel_gain_papers']

        return {
            'df': df,
            'comp_other': comp_other,
            'kwd_df': kwd_df,
            'citations_graph': citations_graph,
            'similarity_graph': similarity_graph,
            'structure_graph': structure_graph,
            'top_cited_papers': top_cited_papers,
            'max_gain_papers': max_gain_papers,
            'max_rel_gain_papers': max_rel_gain_papers
        }

    def init(self, fields):
        """
        Init analyzer with required fields.
        NOTE: results page doesn't use dump/load for visualization.
        Look for init calls in the codebase to find out useful fields for serialization.
        :param fields: desearialized JSON
        """
        logger.debug(f'Loading\n{fields}')
        loaded = PapersAnalyzer.load(fields)
        logger.debug(f'Loaded\n{loaded}')
        self.df = loaded['df']
        self.comp_other = loaded['comp_other']
        # Used for components naming
        self.kwd_df = loaded['kwd_df']
        # Used for structure visualization
        self.citations_graph = loaded['citations_graph']
        self.similarity_graph = loaded['similarity_graph']
        self.structure_graph = loaded['structure_graph']
        # Used for navigation
        self.top_cited_papers = loaded['top_cited_papers']
        self.max_gain_papers = loaded['max_gain_papers']
        self.max_rel_gain_papers = loaded['max_rel_gain_papers']
