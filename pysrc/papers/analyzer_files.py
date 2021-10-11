import hashlib
import json
import logging
import math
import os
from collections import Counter
from itertools import product
from math import sin, cos, pi, fabs

import jinja2
import networkx as nx
import numpy as np
import pandas as pd
from bokeh.colors import RGB
from bokeh.io import reset_output
from bokeh.models import ColumnDataSource, ColorBar, PrintfTickFormatter, LinearColorMapper
from bokeh.models import GraphRenderer, StaticLayoutProvider, Circle, HoverTool, MultiLine, LabelSet
from bokeh.models.graphs import NodesAndLinkedEdges
from bokeh.plotting import figure, output_file, save
from lazy import lazy
from matplotlib import pyplot as plt
from more_itertools import unique_everseen
from sklearn.manifold import TSNE

from pysrc.papers.analysis.citations import find_top_cited_papers, build_cit_stats_df, merge_citation_stats, \
    build_cocit_grouped_df
from pysrc.papers.analysis.graph import build_papers_graph, \
    sparse_graph, to_weighted_graph
from pysrc.papers.analysis.node2vec import node2vec
from pysrc.papers.analysis.text import get_frequent_tokens
from pysrc.papers.analysis.text import texts_embeddings, vectorize_corpus, preprocess_text, word2vec_tokens
from pysrc.papers.analysis.topics import get_topics_description, compute_topics_similarity_matrix, cluster_and_sort
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.plot.plot_preprocessor import PlotPreprocessor
from pysrc.papers.plot.plotter import Plotter
from pysrc.papers.utils import cut_authors_list

logger = logging.getLogger(__name__)

ANALYSIS_FILES_TYPE = 'files'

# Deployment and development
SEARCH_RESULTS_PATHS = ['/search_results', os.path.expanduser('~/.pubtrends/search_results')]


class AnalyzerFiles(PapersAnalyzer):
    # Increase max number of topics
    TOPICS_MAX_NUMBER = 40

    def __init__(self, loader, config, test=False):
        super(AnalyzerFiles, self).__init__(loader, config)

        self.loader = loader
        self.source = Loaders.source(self.loader, test)

    def total_steps(self):
        return 17

    def teardown(self):
        self.progress.remove_handler()

    def analyze_ids(self, ids, source, query, limit, task=None):
        self.query = query
        self.query_folder = self.query_to_folder(source, query, limit)
        logger.info(f'Query folder: {self.query_folder}')
        self.progress.info(f'Found {len(ids)} papers', current=1, task=task)
        path_ids = os.path.join(self.query_folder, 'ids.txt')
        logger.info(f'Ids saved to {path_ids}')
        with open(path_ids, 'w') as f:
            f.write('\n'.join(ids))

        self.progress.info('Loading publications from database', current=2, task=task)
        self.pub_df = self.loader.load_publications(ids)
        self.pub_types = list(set(self.pub_df['type']))
        self.df = self.pub_df
        if len(self.pub_df) == 0:
            raise SearchError(f'Nothing found in database')
        ids = list(self.pub_df['id'])  # Limit ids to existing papers only!
        self.progress.info(f'Found {len(ids)} papers in database', current=2, task=task)

        self.progress.info('Loading citations statistics for papers', current=3, task=task)
        cits_by_year_df = self.loader.load_citations_by_year(ids)
        self.progress.info(f'Found {len(cits_by_year_df)} records of citations by year', current=3, task=task)

        self.cit_stats_df = build_cit_stats_df(cits_by_year_df, len(ids))
        if len(self.cit_stats_df) == 0:
            logger.warning('No citations of papers were found')
        self.df, self.citation_years = merge_citation_stats(self.pub_df, self.cit_stats_df)

        # Load data about citations between given papers (excluding outer papers)
        # IMPORTANT: cit_df may contain not all the publications for query
        self.progress.info('Loading citations information', current=4, task=task)
        self.cit_df = self.loader.load_citations(ids)
        self.progress.info(f'Found {len(self.cit_df)} citations between papers', current=3, task=task)

        self.progress.info('Identifying top cited papers', current=5, task=task)
        logger.debug('Top cited papers')
        self.top_cited_papers, self.top_cited_df = find_top_cited_papers(self.df, PapersAnalyzer.TOP_CITED_PAPERS)

        self.progress.info('Analyzing title and abstract texts', current=6, task=task)
        self.corpus_terms, self.corpus_counts, self.stems_map = vectorize_corpus(
            self.pub_df,
            max_features=PapersAnalyzer.VECTOR_WORDS,
            min_df=PapersAnalyzer.VECTOR_MIN_DF,
            max_df=PapersAnalyzer.VECTOR_MAX_DF
        )
        logger.debug('Analyzing tokens embeddings')
        self.corpus_tokens_embedding = word2vec_tokens(
            self.pub_df, self.corpus_tokens, self.stems_tokens_map
        )
        logger.debug('Analyzing texts embeddings')
        self.texts_embeddings = texts_embeddings(
            self.corpus_counts, self.corpus_tokens_embedding
        )

        self.progress.info('Analyzing MESH terms', current=7, task=task)
        mesh_counter = Counter()
        for mesh_terms in self.df['mesh']:
            if mesh_terms:
                for mt in mesh_terms.split(','):
                    mesh_counter[mt] += 1

        path_mesh_terms_freqs = os.path.join(self.query_folder, "mesh_terms_freqs.html")
        logger.info(f'Save frequent MESH terms to file {path_mesh_terms_freqs}')
        output_file(filename=path_mesh_terms_freqs, title="Mesh terms")
        save(plot_mesh_terms(mesh_counter))
        reset_output()

        logger.info('Computing mesh terms')
        mesh_corpus_terms, mesh_corpus_counts = vectorize_mesh_terms(self.df, mesh_counter)

        if len(mesh_corpus_terms) > 0:
            logger.info('Analyzing mesh terms timeline')
            freq_meshs = get_frequent_mesh_terms(self.top_cited_df)
            keywords_df, years = PlotPreprocessor.frequent_keywords_data(
                freq_meshs, self.df, mesh_corpus_terms, mesh_corpus_counts, 20
            )
            path_mesh_terms_timeline = os.path.join(self.query_folder, 'timeline_mesh_terms.html')
            logging.info(f'Save frequent mesh terms to file {path_mesh_terms_timeline}')
            output_file(filename=path_mesh_terms_timeline, title="Mesh terms timeline")
            save(Plotter.plot_keywords_timeline(keywords_df, years))
            reset_output()

        self.progress.info('Calculating co-citations for selected papers', current=8, task=task)
        self.cocit_df = self.loader.load_cocitations(ids)
        cocit_grouped_df = build_cocit_grouped_df(self.cocit_df)
        logger.debug(f'Found {len(cocit_grouped_df)} co-cited pairs of papers')
        self.cocit_grouped_df = cocit_grouped_df[
            cocit_grouped_df['total'] >= PapersAnalyzer.SIMILARITY_COCITATION_MIN].copy()
        logger.debug(f'Filtered {len(self.cocit_grouped_df)} co-cited pairs of papers, '
                     f'threshold {PapersAnalyzer.SIMILARITY_COCITATION_MIN}')

        self.progress.info('Processing bibliographic coupling for selected papers', current=9, task=task)
        bibliographic_coupling_df = self.loader.load_bibliographic_coupling(ids)
        logger.debug(f'Found {len(bibliographic_coupling_df)} bibliographic coupling pairs of papers')
        self.bibliographic_coupling_df = bibliographic_coupling_df[
            bibliographic_coupling_df['total'] >= PapersAnalyzer.SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN].copy()
        logger.debug(f'Filtered {len(self.bibliographic_coupling_df)} bibliographic coupling pairs of papers '
                     f'threshold {PapersAnalyzer.SIMILARITY_BIBLIOGRAPHIC_COUPLING_MIN}')

        self.progress.info('Analyzing papers graph', current=10, task=task)
        self.papers_graph = build_papers_graph(
            self.df, self.cit_df, self.cocit_grouped_df, self.bibliographic_coupling_df,
        )
        logger.debug(f'Built papers graph - {self.papers_graph.number_of_nodes()} nodes and '
                     f'{self.papers_graph.number_of_edges()} edges')

        self.progress.info(f'Analyzing papers graph with {self.papers_graph.number_of_nodes()} nodes '
                           f'and {self.papers_graph.number_of_edges()} edges', current=11, task=task)
        logger.debug('Analyzing papers graph embeddings')
        self.weighted_similarity_graph = to_weighted_graph(self.papers_graph, PapersAnalyzer.similarity)
        gs = sparse_graph(self.weighted_similarity_graph)
        self.graph_embeddings = node2vec(self.df['id'], gs)

        logger.debug('Computing aggregated graph and text embeddings for papers')
        self.papers_embeddings = np.concatenate(
            (self.graph_embeddings * PapersAnalyzer.GRAPH_EMBEDDINGS_FACTOR,
             self.texts_embeddings * PapersAnalyzer.TEXT_EMBEDDINGS_FACTOR), axis=1)

        logger.debug('Apply TSNE transformation on papers embeddings')
        tsne_embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(self.papers_embeddings)
        self.df['x'] = tsne_embeddings_2d[:, 0]
        self.df['y'] = tsne_embeddings_2d[:, 1]

        self.progress.info('Extracting topics from papers', current=12, task=task)
        clusters, dendrogram_children = cluster_and_sort(self.papers_embeddings,
                                                         PapersAnalyzer.TOPIC_MIN_SIZE,
                                                         self.TOPICS_MAX_NUMBER)
        self.df['comp'] = clusters
        path_topics_sizes = os.path.join(self.query_folder, 'topics_sizes.html')
        logging.info(f'Save topics ratios to file {path_topics_sizes}')
        output_file(filename=path_topics_sizes, title="Topics sizes")
        save(plot_components_ratio(self.df))
        reset_output()

        similarity_df, topics = topics_similarity_data(
            self.papers_embeddings, self.df['comp']
        )
        similarity_df['type'] = ['Inside' if x == y else 'Outside'
                                 for (x, y) in zip(similarity_df['comp_x'], similarity_df['comp_y'])]
        path_topics_similarity = os.path.join(self.query_folder, 'topics_similarity.html')
        logging.info(f'Save similarity heatmap to file {path_topics_similarity}')
        output_file(filename=path_topics_similarity, title="Topics mean similarity")
        save(heatmap_topics_similarity(similarity_df, topics))
        reset_output()

        self.progress.info('Analyzing topics descriptions', current=13, task=task)
        print('Computing clusters keywords')
        clusters_pids = self.df[['id', 'comp']].groupby('comp')['id'].apply(list).to_dict()
        clusters_description = get_topics_description(
            self.df, clusters_pids,
            self.corpus_terms, self.corpus_counts, self.stems_map,
            n_words=PapersAnalyzer.TOPIC_DESCRIPTION_WORDS
        )

        kwds = [(comp, ','.join([f'{t}:{v:.3f}' for t, v in vs[:20]])) for comp, vs in clusters_description.items()]
        self.kwd_df = pd.DataFrame(kwds, columns=['comp', 'kwd'])
        t = self.kwd_df.copy()
        t['comp'] += 1
        path_tags = os.path.join(self.query_folder, 'tags.csv')
        logger.info(f'Save tags to {path_tags}')
        t.to_csv(path_tags, index=False)
        del t

        self.progress.info('Analyzing topics descriptions with MESH terms', current=14, task=task)
        mesh_clusters_description = get_topics_description(
            self.df, clusters_pids,
            mesh_corpus_terms, mesh_corpus_counts, None,
            n_words=PapersAnalyzer.TOPIC_DESCRIPTION_WORDS
        )

        meshs = [(comp, ','.join([f'{t}:{v:.3f}' for t, v in vs[:20]]))
                 for comp, vs in mesh_clusters_description.items()]
        mesh_df = pd.DataFrame(meshs, columns=['comp', 'kwd'])
        t = mesh_df.copy()
        t['comp'] += 1
        path_mesh = os.path.join(self.query_folder, 'mesh.csv')
        logger.info(f'Save topic mesh terms to {path_mesh}')
        t.to_csv(path_mesh, index=False)
        del t

        max_year, min_year = self.df['year'].max(), self.df['year'].min()
        plot_components, data = PlotPreprocessor.component_size_summary_data(
            self.df, sorted(set(self.df['comp'])), min_year, max_year
        )
        path_topics_by_years = os.path.join(self.query_folder, 'topics_by_years.html')
        logging.info(f'Save topics years to file {path_topics_by_years}')
        output_file(filename=path_topics_by_years, title="Topics by years")
        save(Plotter._topics_years_distribution(self.df, self.kwd_df, plot_components, data, min_year, max_year))
        reset_output()

        if len(set(self.df['comp'])) > 1:
            path_topics = os.path.join(self.query_folder, 'topics.html')
            logging.info(f'Save topics hierarchy with keywords to file {path_topics}')
            output_file(filename=path_topics, title="Topics dendrogram")
            save(topics_hierarchy_with_keywords(self.df, self.kwd_df, clusters, dendrogram_children,
                                                max_words=3, plot_height=1200, plot_width=1200))
            reset_output()

            path_topics_mesh = os.path.join(self.query_folder, 'topics_mesh.html')
            logging.info(f'Save topics hierarchy with mesh keywords to file {path_topics_mesh}')
            output_file(filename=path_topics_mesh, title="Topics dendrogram")
            save(topics_hierarchy_with_keywords(self.df, mesh_df, clusters, dendrogram_children,
                                                max_words=3, plot_height=1200, plot_width=1200))
            reset_output()

        self.df['topic_tags'] = [','.join(t for t, _ in clusters_description[c][:5]) for c in self.df['comp']]
        self.df['topic_meshs'] = [','.join(t for t, _ in mesh_clusters_description[c][:5]) for c in self.df['comp']]
        path_papers = os.path.join(self.query_folder, 'papers.csv')
        logging.info(f'Saving papers and components dataframes {path_papers}')
        t = self.df.copy()
        t['comp'] += 1
        t.to_csv(path_papers, index=False)
        del t

        self.progress.info('Preparing papers graphs', current=15, task=task)
        self.sparse_similarity_graph = sparse_graph(self.weighted_similarity_graph, max_edges_to_nodes=5)
        path_papers_graph = os.path.join(self.query_folder, 'papers.html')
        logging.info(f'Saving papers graph for bokeh {path_papers_graph}')
        output_file(filename=path_papers_graph, title="Papers graph")
        save(papers_graph(self.sparse_similarity_graph, self.df, plot_width=1200, plot_height=1200))
        reset_output()

        path_papers_graph_interactive = os.path.join(self.query_folder, 'papers_interactive.html')
        logging.info(f'Saving papers graph for cytoscape.js {path_papers_graph_interactive}')
        template_path = os.path.realpath(os.path.join(__file__, '../../papers_template.html'))
        save_sim_papers_graph_interactive(self.sparse_similarity_graph, self.df, clusters_description,
                                          mesh_clusters_description, template_path, path_papers_graph_interactive)

        self.progress.info('Other analyses', current=16, task=task)
        plotter = Plotter(self)
        path_timeline = os.path.join(self.query_folder, 'timeline.html')
        logging.info(f'Save timeline to {path_timeline}')
        output_file(filename=path_timeline, title="Timeline")
        save(plotter.papers_by_year())
        reset_output()

        path_terms_timeline = os.path.join(self.query_folder, "timeline_terms.html")
        logging.info(f'Save frequent tokens to file {path_terms_timeline}')
        freq_kwds = get_frequent_tokens(self.top_cited_df, self.stems_map)
        output_file(filename=path_terms_timeline, title="Terms timeline")
        keywords_frequencies = plotter.plot_keywords_frequencies(freq_kwds)
        if keywords_frequencies is not None:
            save(keywords_frequencies)
        reset_output()

        self.progress.done('Done analysis', task=task)

    @lazy
    def search_results_folder(self):
        logger.info('Preparing search results folder')
        for path in SEARCH_RESULTS_PATHS:
            if os.path.exists(path):
                logger.info(f'Search results will be stored at {path}')
                return path
        else:
            raise RuntimeError(f'Search results folder not found among: {SEARCH_RESULTS_PATHS}')

    def query_to_folder(self, source, query, limit, max_folder_length=100):
        folder_name = preprocess_text(f'{source}_{query}_{limit}').replace(' ', '_')
        if len(folder_name) > max_folder_length:
            folder_name = folder_name[:(max_folder_length - 32 - 1)] + '_' + \
                          hashlib.md5(folder_name.encode('utf-8')).hexdigest()
        folder = os.path.join(self.search_results_folder, folder_name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        return folder


def plot_mesh_terms(mesh_counter, top=100, plot_width=1200, plot_height=400):
    mc_terms = mesh_counter.most_common(top)
    terms = [mc[0] for mc in mc_terms]
    numbers = [mc[1] for mc in mc_terms]
    cmap = Plotter.factors_colormap(top)
    colors = [Plotter.color_to_rgb(cmap(i)) for i in range(top)]
    source = ColumnDataSource(data=dict(terms=terms, numbers=numbers, colors=colors))

    p = figure(plot_width=plot_width, plot_height=plot_height,
               toolbar_location="right", tools="save", x_range=terms)
    p.vbar(x='terms', top='numbers', width=0.8, fill_alpha=0.5, color='colors', source=source)
    p.hover.tooltips = [("Term", '@terms'), ("Number", '@numbers')]
    p.sizing_mode = 'stretch_width'
    p.xaxis.axis_label = 'Mesh term'
    p.xaxis.major_label_orientation = math.pi / 4
    p.yaxis.axis_label = 'Numbner of papers'
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    return p


def vectorize_mesh_terms(df, mesh_counter, max_features=1000, min_df=0.01, max_df=0.8):
    logger.debug(f'Vectorization mesh terms min_df={min_df} max_df={max_df} max_features={max_features}')
    features = []
    for mt, count in mesh_counter.most_common():
        if len(features) >= max_features:
            break
        if count < min_df * len(df) or count > max_df * len(df):
            continue
        features.append(mt)
    features_map = dict((f, i) for i, f in enumerate(features))
    logging.debug(f'Total {len(features)} mesh terms filtered')

    counts = np.asmatrix(np.zeros(shape=(len(df), len(features))))
    for i, mesh_terms in enumerate(df['mesh']):
        if mesh_terms:
            for mt in mesh_terms.split(','):
                if mt in features_map:
                    counts[i, features_map[mt]] = 1
    logger.debug(f'Vectorized corpus size {counts.shape}')
    if counts.shape[1] != 0:
        terms_counts = np.asarray(np.sum(counts, axis=0)).reshape(-1)
        terms_freqs = terms_counts / len(df)
        logger.debug(f'Terms frequencies min={terms_freqs.min()}, max={terms_freqs.max()}, '
                     f'mean={terms_freqs.mean()}, std={terms_freqs.std()}')
    return features, counts


def get_frequent_mesh_terms(df, fraction=0.1, min_tokens=20):
    counter = Counter()
    for mesh_terms in df['mesh']:
        if mesh_terms:
            for mt in mesh_terms.split(','):
                counter[mt] += 1
    result = {}
    tokens = len(counter)
    for token, cnt in counter.most_common(max(min_tokens, int(tokens * fraction))):
        result[token] = cnt / tokens
    return result


def components_ratio_data(df):
    comp_sizes = dict(df.groupby('comp')['id'].count())
    comps = list(comp_sizes.keys())
    ratios = [100 * comp_sizes[c] / len(df) for c in comps]

    # c + 1 is used to start numbering from 1
    comps = list(map(str, [c + 1 for c in comps]))
    return comps, ratios


def plot_components_ratio(df, plot_width=1200, plot_height=400):
    comps, ratios = components_ratio_data(df)
    n_comps = len(comps)
    cmap = Plotter.factors_colormap(n_comps)
    colors = [Plotter.color_to_rgb(cmap(i)) for i in range(n_comps)]
    source = ColumnDataSource(data=dict(comps=comps, ratios=ratios, colors=colors))

    p = figure(plot_width=plot_width, plot_height=plot_height,
               toolbar_location="right", tools="save", x_range=comps)
    p.vbar(x='comps', top='ratios', width=0.8, fill_alpha=0.5, color='colors', source=source)
    p.hover.tooltips = [("Topic", '@comps'), ("Amount", '@ratios %')]
    p.sizing_mode = 'stretch_width'
    p.xaxis.axis_label = 'Topic'
    p.yaxis.axis_label = 'Percentage of papers'
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None

    return p


def topics_similarity_data(papers_embeddings, comps):
    similarity_matrix = compute_topics_similarity_matrix(papers_embeddings, comps)

    # c + 1 is used to start numbering with 1
    components = [str(c + 1) for c in sorted(set(comps))]
    n_comps = len(components)
    similarity_topics_df = pd.DataFrame([
        {'comp_x': i, 'comp_y': j, 'similarity': similarity_matrix[i, j]}
        for i, j in product(range(n_comps), range(n_comps))
    ])
    similarity_topics_df['comp_x'] = similarity_topics_df['comp_x'].apply(lambda x: x + 1).astype(str)
    similarity_topics_df['comp_y'] = similarity_topics_df['comp_y'].apply(lambda x: x + 1).astype(str)
    return similarity_topics_df, components


def heatmap_topics_similarity(similarity_df, topics):
    logger.debug('Visualizing topics similarity with heatmap')

    step = 10
    cmap = plt.cm.get_cmap('PuBu', step)
    colors = [RGB(*[round(c * 255) for c in cmap(i)[:3]]) for i in range(step)]
    mapper = LinearColorMapper(palette=colors,
                               low=similarity_df.similarity.min(),
                               high=similarity_df.similarity.max())

    p = figure(x_range=topics, y_range=topics,
               x_axis_location="below", plot_width=600, plot_height=600,
               tools="hover,pan,tap,wheel_zoom,box_zoom,reset,save", toolbar_location="right",
               tooltips=[('Topic 1', '@comp_x'),
                         ('Topic 2', '@comp_y'),
                         ('Similarity', '@similarity')])

    p.sizing_mode = 'stretch_width'
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 0

    p.rect(x="comp_x", y="comp_y", width=1, height=1,
           source=similarity_df,
           fill_color={'field': 'similarity', 'transform': mapper},
           line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="10pt",
                         formatter=PrintfTickFormatter(format="%.2f"),
                         label_standoff=11, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    return p


def compute_clusters_dendrogram_children(clusters, children):
    leaves_map = dict(enumerate(clusters))
    nodes_map = {}
    clusters_children = []
    for i, (u, v) in enumerate(children):
        u_cluster = leaves_map[u] if u in leaves_map else nodes_map[u]
        v_cluster = leaves_map[v] if v in leaves_map else nodes_map[v]
        node = len(leaves_map) + i
        if u_cluster is not None and v_cluster is not None:
            if u_cluster != v_cluster:
                nodes_map[node] = None  # Different clusters
                clusters_children.append((u, v, node))
            else:
                nodes_map[node] = u_cluster
        else:
            nodes_map[node] = None  # Different clusters
            clusters_children.append((u, v, node))

    def rwc(v):
        if v in leaves_map:
            return leaves_map[v]
        elif v in nodes_map:
            res = nodes_map[v]
            return res if res is not None else v
        else:
            return v

    # Rename nodes to clusters
    result = [(rwc(u), rwc(v), rwc(n)) for u, v, n in clusters_children]
    #     logger.debug(f'Clusters based dendrogram children {result}')
    return result


def convert_clusters_dendrogram_to_paths(clusters, children):
    logger.debug('Converting agglomerate clustering clusters dendrogram format to path for visualization')
    paths = [[p] for p in sorted(set(clusters))]
    for i, (u, v, n) in enumerate(children):
        for p in paths:
            if p[i] == u or p[i] == v:
                p.append(n)
            else:
                p.append(p[i])
    #     logger.debug(f'Paths {paths}')
    logger.debug('Radix sort or paths to ensure no overlaps')
    for i in range(len(children)):
        paths.sort(key=lambda p: p[i])
        # Reorder next level to keep order of previous if possible
        if i != len(children):
            order = dict((v, i) for i, v in enumerate(unique_everseen(p[i + 1] for p in paths)))
            for p in paths:
                p[i + 1] = order[p[i + 1]]
    leaves_order = dict((v, i) for i, v in enumerate(unique_everseen(p[0] for p in paths)))
    return paths, leaves_order


def contrast_color(rgb):
    r, g, b = rgb.r, rgb.g, rgb.b
    """
    Light foreground for dark background and vice verse.
    Idea Taken from https://stackoverflow.com/a/1855903/418358
    """
    # Counting the perceptive luminance - human eye favors green color...
    if 1 - (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.5:
        return RGB(0, 0, 0)
    else:
        return RGB(255, 255, 255)


def topics_words(kwd_df, max_words):
    words2show = {}
    for _, row in kwd_df.iterrows():
        comp, kwds = row[0], row[1]
        if kwds != '':  # Correctly process empty freq_kwds encoding
            words2show[comp] = [p.split(':')[0] for p in kwds.split(',')[:max_words]]
    return words2show


def topics_hierarchy_with_keywords(df, kwd_df, clusters, dendrogram_children,
                                   max_words=3, plot_width=1200, plot_height=800):
    comp_sizes = Counter(df['comp'])
    logger.debug('Computing dendrogram for clusters')
    if dendrogram_children is None:
        return None
    clusters_dendrogram = compute_clusters_dendrogram_children(clusters, dendrogram_children)
    paths, leaves_order = convert_clusters_dendrogram_to_paths(clusters, clusters_dendrogram)

    # Configure dimensions
    p = figure(x_range=(-180, 180),
               y_range=(-160, 160),
               tools="save",
               width=plot_width, height=plot_height)
    x_coefficient = 1.2  # Ellipse x coefficient
    y_delta = 40  # Extra space near pi / 2 and 3 * pi / 2
    n_topics = len(leaves_order)
    radius = 100  # Radius of circular dendrogram
    dendrogram_len = len(paths[0])
    d_radius = radius / dendrogram_len
    d_degree = 2 * pi / n_topics

    # Leaves coordinates
    leaves_degrees = dict((v, i * d_degree) for v, i in leaves_order.items())

    # Draw dendrogram - from bottom to top
    ds = leaves_degrees.copy()
    for i in range(1, dendrogram_len):
        next_ds = {}
        for path in paths:
            if path[i] not in next_ds:
                next_ds[path[i]] = []
            next_ds[path[i]].append(ds[path[i - 1]])
        for v, nds in next_ds.items():
            next_ds[v] = np.mean(nds)

        for path in paths:
            current_d = ds[path[i - 1]]
            next_d = next_ds[path[i]]
            p.line([cos(current_d) * d_radius * (dendrogram_len - i),
                    cos(next_d) * d_radius * (dendrogram_len - i - 1)],
                   [sin(current_d) * d_radius * (dendrogram_len - i),
                    sin(next_d) * d_radius * (dendrogram_len - i - 1)],
                   line_color='lightgray')
        ds = next_ds

    # Draw leaves
    n_comps = len(comp_sizes)
    cmap = Plotter.factors_colormap(n_comps)
    topics_colors = dict((i, Plotter.color_to_rgb(cmap(i))) for i in range(n_comps))
    xs = [cos(d) * d_radius * (dendrogram_len - 1) for _, d in leaves_degrees.items()]
    ys = [sin(d) * d_radius * (dendrogram_len - 1) for _, d in leaves_degrees.items()]
    sizes = [20 + int(min(10, math.log(comp_sizes[v]))) for v, _ in leaves_degrees.items()]
    comps = [v + 1 for v, _ in leaves_degrees.items()]
    colors = [topics_colors[v] for v, _ in leaves_degrees.items()]
    ds = ColumnDataSource(data=dict(x=xs, y=ys, size=sizes, comps=comps, color=colors))
    p.circle(x='x', y='y', size='size', fill_color='color', line_color='black', source=ds)

    # Topics labels
    p.text(x=[cos(d) * d_radius * (dendrogram_len - 1) for _, d in leaves_degrees.items()],
           y=[sin(d) * d_radius * (dendrogram_len - 1) for _, d in leaves_degrees.items()],
           text=[str(v + 1) for v, _ in leaves_degrees.items()],
           text_align='center', text_baseline='middle', text_font_size='10pt',
           text_color=[contrast_color(topics_colors[v]) for v, _ in leaves_degrees.items()])

    # Show words for components - most popular words per component
    topics = leaves_order.keys()
    words2show = topics_words(kwd_df, max_words)

    # Visualize words
    for v, d in leaves_degrees.items():
        if v not in words2show:  # No super-specific words for topic
            continue
        words = words2show[v]
        xs = []
        ys = []
        for i, word in enumerate(words):
            wd = d + d_degree * (i - len(words) / 2) / len(words)
            # Make word degree in range 0 - 2 * pi
            if wd < 0:
                wd += 2 * pi
            elif wd > 2 * pi:
                wd -= 2 * pi
            xs.append(cos(wd) * radius * x_coefficient)
            y = sin(wd) * radius
            # Additional vertical space around pi/2 and 3*pi/2
            if pi / 4 <= wd < 3 * pi / 4:
                y += pow(pi / 4 - fabs(pi / 2 - wd), 1.5) * y_delta
            elif 5 * pi / 4 <= wd < 7 * pi / 4:
                y -= pow(pi / 4 - fabs(3 * pi / 2 - wd), 1.5) * y_delta
            ys.append(y)

        # Different text alignment for left | right parts
        p.text(x=[x for x in xs if x > 0], y=[y for i, y in enumerate(ys) if xs[i] > 0],
               text=[w for i, w in enumerate(words) if xs[i] > 0],
               text_align='left', text_baseline='middle', text_font_size='10pt',
               text_color=topics_colors[v])
        p.text(x=[x for x in xs if x <= 0], y=[y for i, y in enumerate(ys) if xs[i] <= 0],
               text=[w for i, w in enumerate(words) if xs[i] <= 0],
               text_align='right', text_baseline='middle', text_font_size='10pt',
               text_color=topics_colors[v])

    p.sizing_mode = 'stretch_width'
    p.axis.major_tick_line_color = None
    p.axis.minor_tick_line_color = None
    p.axis.major_label_text_color = None
    p.axis.major_label_text_font_size = '0pt'
    p.axis.axis_line_color = None
    p.grid.grid_line_color = None
    p.outline_line_color = None
    return p


def papers_graph(g, df, plot_width=600, plot_height=600):
    nodes = df['id']
    graph = GraphRenderer()
    comps = df['comp']
    cmap = Plotter.factors_colormap(len(set(comps)))
    palette = dict(zip(sorted(set(comps)), [Plotter.color_to_rgb(cmap(i)).to_hex()
                                            for i in range(len(set(comps)))]))

    graph.node_renderer.data_source.add(df['id'], 'index')
    graph.node_renderer.data_source.data['id'] = df['id']
    graph.node_renderer.data_source.data['title'] = df['title']
    graph.node_renderer.data_source.data['authors'] = df['authors']
    graph.node_renderer.data_source.data['journal'] = df['journal']
    graph.node_renderer.data_source.data['year'] = df['year']
    graph.node_renderer.data_source.data['type'] = df['type']
    graph.node_renderer.data_source.data['total'] = df['total']
    graph.node_renderer.data_source.data['mesh'] = df['mesh']
    graph.node_renderer.data_source.data['keywords'] = df['keywords']
    graph.node_renderer.data_source.data['topic'] = [c + 1 for c in comps]
    graph.node_renderer.data_source.data['topic_tags'] = df['topic_tags']
    graph.node_renderer.data_source.data['topic_meshs'] = df['topic_meshs']

    # Aesthetics
    graph.node_renderer.data_source.data['size'] = df['total'] * 20 / df['total'].max() + 5
    graph.node_renderer.data_source.data['color'] = [palette[c] for c in comps]

    # Edges
    graph.edge_renderer.data_source.data = dict(start=[u for u, _ in g.edges],
                                                end=[v for _, v in g.edges])

    # start of layout code
    x = df['x']
    y = df['y']
    xrange = max(x) - min(x)
    yrange = max(y) - min(y)
    p = figure(plot_width=plot_width,
               plot_height=plot_height,
               x_range=(min(x) - 0.05 * xrange, max(x) + 0.05 * xrange),
               y_range=(min(y) - 0.05 * yrange, max(y) + 0.05 * yrange),
               tools="pan,tap,wheel_zoom,box_zoom,reset,save")
    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
    p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
    p.grid.grid_line_color = None
    p.outline_line_color = None
    p.sizing_mode = 'stretch_width'

    p.add_tools(HoverTool(tooltips=Plotter._paper_html_tooltips('Pubmed', [
        ("Author(s)", '@authors'),
        ("Journal", '@journal'),
        ("Year", '@year'),
        ("Type", '@type'),
        ("Cited by", '@total paper(s) total'),
        ("Mesh", '@mesh'),
        ("Keywords", '@keywords'),
        ("Topic", '@topic'),
        ("Topic tags", '@topic_tags'),
        ("Topic Mesh", '@topic_meshs'),
    ])))

    graph_layout = dict(zip(nodes, zip(x, y)))
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    graph.node_renderer.glyph = Circle(size='size', fill_alpha=0.7, line_alpha=0.7, fill_color='color')
    graph.node_renderer.hover_glyph = Circle(size='size', fill_alpha=1.0, line_alpha=1.0, fill_color='color')

    graph.edge_renderer.glyph = MultiLine(line_color='lightgrey', line_alpha=0.5, line_width=1)
    graph.edge_renderer.hover_glyph = MultiLine(line_color='grey', line_alpha=1.0, line_width=2)

    graph.inspection_policy = NodesAndLinkedEdges()
    p.renderers.append(graph)

    # Add Labels
    lxs = [df.loc[df['comp'] == c]['x'].mean() for c in sorted(set(comps))]
    lys = [df.loc[df['comp'] == c]['y'].mean() for c in sorted(set(comps))]
    comp_labels = [f"#{c + 1}" for c in sorted(set(comps))]
    source = ColumnDataSource({'x': lxs, 'y': lys, 'name': comp_labels})
    labels = LabelSet(x='x', y='y', text='name', source=source,
                      background_fill_color='white', text_font_size='11px', background_fill_alpha=.9)
    p.renderers.append(labels)

    return p


def save_sim_papers_graph_interactive(gs, df, clusters_description, mesh_clusters_description,
                                      template_path, path):
    logging.info('Saving papers graph for cytoscape.js')

    topics_tags = {c: ','.join(t for t, _ in clusters_description[c][:5]) for c in sorted(set(df['comp']))}
    topics_meshs = {c: ','.join(t for t, _ in mesh_clusters_description[c][:5]) for c in sorted(set(df['comp']))}

    logger.debug('Creating graph')
    gss = nx.Graph()
    for (u, v) in gs.edges():
        gss.add_edge(u, v)
    for n in gs.nodes():
        if not gss.has_node(n):
            gss.add_node(n)

    logger.debug('Collect attributes for nodes')
    attrs = {}
    for node in df['id']:
        sel = df[df['id'] == node]
        attrs[node] = dict(
            title=sel['title'].values[0],
            authors=cut_authors_list(sel['authors'].values[0]),
            journal=sel['journal'].values[0],
            year=int(sel['year'].values[0]),
            cited=int(sel['total'].values[0]),
            topic=int(sel['comp'].values[0]),
            # These can be heavy
            abstract=sel['abstract'].values[0],
            mesh=sel['mesh'].values[0],
            keywords=sel['keywords'].values[0]
        )

    nx.set_node_attributes(gss, attrs)
    graph_cs = nx.cytoscape_data(gss)['elements']

    logger.debug('Layout')
    maxy = df['y'].max()
    for node_cs in graph_cs['nodes']:
        nid = node_cs['data']['id']
        sel = df.loc[df['id'] == nid]
        # Adjust vertical axis with bokeh graph
        node_cs['position'] = dict(x=int(sel['x'].values[0] * 8),
                                   y=int((maxy - sel['y'].values[0]) * 6))

    with open(template_path) as f:
        text = f.read()

    html = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(text).render(
        topics_palette_json=json.dumps(Plotter.topics_palette(df)),
        topics_tags_json=json.dumps(topics_tags),
        topics_meshs_json=json.dumps(topics_meshs),
        graph_cytoscape_json=json.dumps(graph_cs)
    )

    with open(path, 'w') as f:
        f.write(html)

    logger.debug('Done')


FILES_WITH_DESCRIPTIONS = {
    'ids.txt': 'List of ids returned by search request',

    'papers.csv': 'Detailed information about papers used for analysis',
    'mesh.csv': 'Information about MESH terms for each topic',
    'tags.csv': 'Information about keywords for each topic',

    'timeline.html': 'Overall number of papers per year',
    'timeline_terms.html': 'Most popular keywords per year',
    'timeline_mesh_terms.html': 'Most popular MESH terms per year',
    'mesh_terms_freqs.html': 'Frequency of MESH terms used in papers',

    'papers.html': 'Graph representation of papers',
    'papers_interactive.html': 'Interactive version of papers graph with advanced filtering, coloring',
    'topics.html': 'Topics hierarchy',
    'topics_mesh.html': 'Topics hierarchy with MESH terms',
    'topics_by_years.html': 'Topics per year',

    'topics_similarity.html': 'Similarity between papers within topics',
    'topics_sizes.html': 'Topics sizes',
}
