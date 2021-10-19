import json
import json
import logging
import math
import os
from collections import Counter
from itertools import product, chain

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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from pysrc.app.predefined import query_to_folder
from pysrc.papers.analysis.citations import find_top_cited_papers, build_cit_stats_df, merge_citation_stats, \
    build_cocit_grouped_df
from pysrc.papers.analysis.graph import build_papers_graph, \
    sparse_graph, to_weighted_graph
from pysrc.papers.analysis.node2vec import node2vec
from pysrc.papers.analysis.text import get_frequent_tokens
from pysrc.papers.analysis.text import texts_embeddings, vectorize_corpus, word2vec_tokens
from pysrc.papers.analysis.topics import get_topics_description, compute_topics_similarity_matrix, cluster_and_sort
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.db.search_error import SearchError
from pysrc.papers.plot.plot_preprocessor import PlotPreprocessor
from pysrc.papers.plot.plotter import Plotter, PLOT_WIDTH, SHORT_PLOT_HEIGHT, TALL_PLOT_HEIGHT
from pysrc.papers.utils import cut_authors_list
from pysrc.version import VERSION

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

    def analyze_ids(self, ids, source, query, sort, limit, test=False, task=None):
        self.query = query
        self.query_folder = os.path.join(self.search_results_folder, f"{VERSION.replace(' ', '_')}",
                                         query_to_folder(source, query, sort, limit))
        if not os.path.exists(self.query_folder):
            os.makedirs(self.query_folder)
        logger.info(f'Query folder: {self.query_folder}')
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
        self.corpus, self.corpus_tokens, self.corpus_counts = vectorize_corpus(
            self.pub_df,
            max_features=PapersAnalyzer.VECTOR_WORDS,
            min_df=PapersAnalyzer.VECTOR_MIN_DF,
            max_df=PapersAnalyzer.VECTOR_MAX_DF,
            test=test
        )
        logger.debug('Analyzing tokens embeddings')
        self.corpus_tokens_embedding = word2vec_tokens(
            self.corpus, self.corpus_tokens, test=test
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
        mesh_corpus_tokens, mesh_corpus_counts = vectorize_mesh_tokens(self.df, mesh_counter)

        if len(mesh_corpus_tokens) > 0:
            logger.info('Analyzing mesh terms timeline')
            freq_meshs = get_frequent_mesh_terms(self.top_cited_df)
            keywords_df, years = PlotPreprocessor.frequent_keywords_data(
                freq_meshs, self.df, mesh_corpus_tokens, mesh_corpus_counts, 20
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
        papers_embeddings = np.concatenate(
            (self.graph_embeddings * PapersAnalyzer.GRAPH_EMBEDDINGS_FACTOR,
             self.texts_embeddings * PapersAnalyzer.TEXT_EMBEDDINGS_FACTOR), axis=1)

        logger.debug('Computing PCA projection')
        pca = PCA(n_components=min(len(papers_embeddings), PapersAnalyzer.PCA_COMPONENTS))
        t = StandardScaler().fit_transform(papers_embeddings)
        self.pca_coords = pca.fit_transform(t)
        logger.debug('Explained variation', int(np.sum(pca.explained_variance_ratio_) * 100), '%')

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

        self.progress.info('Extracting topics from papers', current=12, task=task)
        clusters, dendrogram = cluster_and_sort(
            self.pca_coords, PapersAnalyzer.TOPIC_MIN_SIZE, self.TOPICS_MAX_NUMBER
        )
        self.df['comp'] = clusters
        path_topics_sizes = os.path.join(self.query_folder, 'topics_sizes.html')
        logging.info(f'Save topics ratios to file {path_topics_sizes}')
        output_file(filename=path_topics_sizes, title="Topics sizes")
        save(plot_components_ratio(self.df))
        reset_output()

        similarity_df, topics = topics_similarity_data(
            self.pca_coords, self.df['comp']
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
            self.corpus, self.corpus_tokens, self.corpus_counts,
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
        mesh_corpus = [
            [[mesh_corpus_tokens[i]] * int(mc) for i, mc in
             enumerate(np.asarray(mesh_corpus_counts[pid, :]).reshape(-1)) if mc > 0]
            for pid in range(mesh_corpus_counts.shape[0])
        ]
        mesh_clusters_description = get_topics_description(
            self.df, clusters_pids,
            mesh_corpus, mesh_corpus_tokens, mesh_corpus_counts,
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
            save(Plotter._topics_hierarchy_with_keywords(self.df, self.kwd_df, clusters, dendrogram, max_words=3))
            reset_output()

            path_topics_mesh = os.path.join(self.query_folder, 'topics_mesh.html')
            logging.info(f'Save topics hierarchy with mesh keywords to file {path_topics_mesh}')
            output_file(filename=path_topics_mesh, title="Topics dendrogram")
            save(Plotter._topics_hierarchy_with_keywords(self.df, mesh_df, clusters, dendrogram, max_words=3))
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
        logger.debug('Prepare sparse graph for visualization')
        self.sparse_papers_graph = self.prepare_sparse_papers_graph(self.papers_graph, self.weighted_similarity_graph)
        path_papers_graph = os.path.join(self.query_folder, 'papers.html')
        logging.info(f'Saving papers graph for bokeh {path_papers_graph}')
        output_file(filename=path_papers_graph, title="Papers graph")
        save(Plotter._plot_papers_graph(source, self.sparse_papers_graph, self.df,
                                        clusters_description, mesh_clusters_description, add_callback=False,
                                        plot_width=PLOT_WIDTH, plot_height=PLOT_WIDTH))
        reset_output()

        path_papers_graph_interactive = os.path.join(self.query_folder, 'papers_interactive.html')
        logging.info(f'Saving papers graph for cytoscape.js {path_papers_graph_interactive}')
        template_path = os.path.realpath(os.path.join(__file__, '../../papers_template.html'))
        save_sim_papers_graph_interactive(self.sparse_papers_graph, self.df, clusters_description,
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
        freq_kwds = get_frequent_tokens(chain(*chain(*self.corpus)))
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


def plot_mesh_terms(mesh_counter, top=100, plot_width=PLOT_WIDTH, plot_height=TALL_PLOT_HEIGHT):
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


def vectorize_mesh_tokens(df, mesh_counter, max_features=1000,
                          min_df=0.1, max_df=PapersAnalyzer.VECTOR_MAX_DF):
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
    for i, mesh_tokens in enumerate(df['mesh']):
        if mesh_tokens:
            for mt in mesh_tokens.split(','):
                if mt in features_map:
                    counts[i, features_map[mt]] = 1
    logger.debug(f'Vectorized corpus size {counts.shape}')
    if counts.shape[1] != 0:
        tokens_counts = np.asarray(np.sum(counts, axis=0)).reshape(-1)
        tokens_freqs = tokens_counts / len(df)
        logger.debug(f'Terms frequencies min={tokens_freqs.min()}, max={tokens_freqs.max()}, '
                     f'mean={tokens_freqs.mean()}, std={tokens_freqs.std()}')
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


def plot_components_ratio(df, plot_width=PLOT_WIDTH, plot_height=SHORT_PLOT_HEIGHT):
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


def heatmap_topics_similarity(similarity_df, topics, plot_width=PLOT_WIDTH, plot_height=TALL_PLOT_HEIGHT):
    logger.debug('Visualizing topics similarity with heatmap')

    step = 10
    cmap = plt.cm.get_cmap('PuBu', step)
    colors = [RGB(*[round(c * 255) for c in cmap(i)[:3]]) for i in range(step)]
    mapper = LinearColorMapper(palette=colors,
                               low=similarity_df.similarity.min(),
                               high=similarity_df.similarity.max())

    p = figure(x_range=topics, y_range=topics,
               x_axis_location="below", plot_width=plot_width, plot_height=plot_height,
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
