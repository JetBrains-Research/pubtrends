import logging
import math
from itertools import product as cart_product
from string import Template

import holoviews as hv
import ipywidgets as widgets
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import display
from bokeh.colors import RGB
from bokeh.core.properties import value
from bokeh.io import push_notebook
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CDSView, GroupFilter, CustomJS, HTMLTemplateFormatter
from bokeh.models import GraphRenderer, StaticLayoutProvider
# Tools used: hover,pan,tap,wheel_zoom,box_zoom,reset,save
from bokeh.models import HoverTool, PanTool, WheelZoomTool, BoxZoomTool, ResetTool, SaveTool
from bokeh.models import LinearColorMapper, PrintfTickFormatter, ColorBar
from bokeh.models import Plot, Range1d, MultiLine, Circle, Span
from bokeh.models.widgets.tables import DataTable, TableColumn
from bokeh.palettes import Category20
from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap
from holoviews import dim
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from .utils import PUBMED_ARTICLE_BASE_URL, SEMANTIC_SCHOLAR_BASE_URL, get_word_cloud_data, \
    get_most_common_ngrams

TOOLS = "hover,pan,tap,wheel_zoom,box_zoom,reset,save"
hv.extension('bokeh')


class Plotter:
    def __init__(self, analyzer):
        self.analyzer = analyzer

        n_comps = len(self.analyzer.components)
        self.comp_palette = []
        for color in Category20[20][:n_comps]:
            rgb_values = []
            for pos in range(1, 7, 2):
                rgb_values.append(int(color[pos:pos + 2], 16))
            self.comp_palette.append(RGB(*rgb_values))

    @staticmethod
    def pubmed_callback(source, db):
        if db == 'semantic':
            base = SEMANTIC_SCHOLAR_BASE_URL
        elif db == 'pubmed':
            base = PUBMED_ARTICLE_BASE_URL
        else:
            raise ValueError("Wrong value of db")
        return CustomJS(args=dict(source=source, base=base), code="""
            var data = source.data, selected = source.selected.indices;
            if (selected.length == 1) {
                // only consider case where one glyph is selected by user
                selected_id = data['id'][selected[0]]
                for (var i = 0; i < data['id'].length; ++i){
                    if(data['id'][i] == selected_id){
                        window.open(base + data['id'][i], '_blank');
                    }
                }
            }
        """)

    def chord_diagram_components(self):
        logging.info('Visualizing components with Chord diagram')

        G = nx.Graph()
        # Using merge left keeps order
        gdf = pd.merge(pd.Series(self.analyzer.CG.nodes(), dtype=object).reset_index().rename(
            columns={0: 'id'}),
            self.analyzer.df[['id', 'title', 'authors', 'year', 'total', 'comp']],
            how='left'
        ).sort_values(by='total', ascending=False)

        for c in range(len(self.analyzer.components)):
            for n in gdf[gdf['comp'] == c]['id']:
                # NOTE: we use nodes id as String to avoid problems str keys in jsonify
                # during graph visualization
                G.add_node(str(n))

        edge_starts = []
        edge_ends = []
        edge_weights = []
        for start, end, data in self.analyzer.CG.edges(data=True):
            edge_starts.append(start)
            edge_ends.append(end)
            edge_weights.append(min(data['weight'], 20))

        # Show with Bokeh
        plot = Plot(plot_width=800, plot_height=800,
                    x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
        plot.title.text = 'Co-citations graph'

        graph = GraphRenderer()
        graph.node_renderer.data_source.add(list(G.nodes), 'index')
        graph.edge_renderer.data_source.data = dict(start=edge_starts,
                                                    end=edge_ends)

        # Start of layout code
        circ = [i * 2 * math.pi / len(G.nodes()) for i in range(len(G.nodes()))]
        x = [math.cos(i) for i in circ]
        y = [math.sin(i) for i in circ]
        graph_layout = dict(zip(list(G.nodes()), zip(x, y)))
        graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

        # Draw quadratic bezier paths
        def bezier(start, end, steps=10, c=1.5):
            return [(1 - s) ** c * start + s ** c * end for s in np.linspace(0, 1, steps)]

        xs, ys = [], []
        edge_colors = []
        edge_alphas = []
        for edge_start, edge_end in zip(edge_starts, edge_ends):
            sx, sy = graph_layout[edge_start]
            ex, ey = graph_layout[edge_end]
            xs.append(bezier(sx, ex))
            ys.append(bezier(sy, ey))
            if self.analyzer.pm[edge_start] == self.analyzer.pm[edge_end]:
                edge_colors.append(self.comp_palette[self.analyzer.pm[edge_start]])
                edge_alphas.append(0.1)
            else:
                edge_colors.append('grey')
                edge_alphas.append(0.05)

        # Paths for edges
        graph.edge_renderer.data_source.data['xs'] = xs
        graph.edge_renderer.data_source.data['ys'] = ys

        # Style for edges
        graph.edge_renderer.data_source.data['edge_colors'] = edge_colors
        graph.edge_renderer.data_source.data['edge_alphas'] = edge_alphas
        graph.edge_renderer.data_source.data['edge_weights'] = edge_weights

        # TODO: use ColumnDatasource
        # Nodes data for rendering
        graph.node_renderer.data_source.data['id'] = list(G.nodes())
        graph.node_renderer.data_source.data['colors'] = [self.comp_palette[self.analyzer.pm[n]] for n in
                                                          G.nodes()]
        graph.node_renderer.data_source.data['title'] = gdf['title']
        graph.node_renderer.data_source.data['authors'] = gdf['authors']
        graph.node_renderer.data_source.data['year'] = gdf['year'].replace(np.nan, "Undefined")
        graph.node_renderer.data_source.data['total'] = gdf['total']
        log_total = np.log(gdf['total'])
        graph.node_renderer.data_source.data['size'] = (log_total / np.max(log_total)) * 5 + 5
        graph.node_renderer.data_source.data['topic'] = \
            [f'#{self.analyzer.pm[n]}{" OTHER" if self.analyzer.pm[n] == 0 and self.analyzer.components_merged else ""}'
             for n in G.nodes()]

        # node rendering
        graph.node_renderer.glyph = Circle(size='size', line_color='colors', fill_color='colors',
                                           line_alpha=0.5, fill_alpha=0.8)
        # edge rendering
        graph.edge_renderer.glyph = MultiLine(
            line_color='edge_colors',
            line_alpha='edge_alphas',
            line_width='edge_weights')

        # add tools to the plot
        # hover,pan,tap,wheel_zoom,box_zoom,reset,save
        plot.add_tools(HoverTool(tooltips=self._html_tooltips([
            ("Author(s)", '@authors'),
            ("Year", '@year'),
            ("Cited by", '@total paper(s) total'),
            ("Subtopic", '@topic')])),
            PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), SaveTool())

        plot.renderers.append(graph)
        return plot

    def heatmap_clusters(self):
        logging.info('Visualizing components with heatmap')

        clusters = list(map(str, self.analyzer.components))
        n_comps = len(clusters)

        links = pd.DataFrame(self.analyzer.CG.edges(data=True), columns=['source', 'target', 'value'])
        links['value'] = links['value'].apply(lambda data: data['weight'])

        cluster_edges = links.merge(self.analyzer.df[['id', 'comp']], how='left', left_on='source',
                                    right_on='id').merge(self.analyzer.df[['id', 'comp']], how='left', left_on='target',
                                                         right_on='id')

        are_swapped = (cluster_edges['comp_x'] <= cluster_edges['comp_y'])
        cluster_edges = cluster_edges.loc[are_swapped].rename(columns={'comp_x': 'comp_y', 'comp_y': 'comp_x'})
        cluster_edges = cluster_edges.groupby(['comp_x', 'comp_y'])['value'].sum().reset_index()

        connectivity_matrix = [[0] * n_comps for _ in range(n_comps)]
        for index, row in cluster_edges.iterrows():
            connectivity_matrix[row['comp_x']][row['comp_y']] = row['value']
            connectivity_matrix[row['comp_y']][row['comp_x']] = row['value']

        cluster_edges = pd.DataFrame([{'comp_x': i, 'comp_y': j, 'value': connectivity_matrix[i][j]}
                                      for i, j in cart_product(range(n_comps), range(n_comps))])

        def get_density(row):
            return row['value'] / (self.analyzer.pmcomp_sizes[row['comp_x']] *
                                   self.analyzer.pmcomp_sizes[row['comp_y']])

        cluster_edges['density'] = cluster_edges.apply(lambda row: get_density(row), axis=1)
        cluster_edges['comp_x'] = cluster_edges['comp_x'].astype(str)
        cluster_edges['comp_y'] = cluster_edges['comp_y'].astype(str)

        step = 30
        cmap = plt.cm.get_cmap('PuBu', step)
        colors = [RGB(*[round(c * 255) for c in cmap(i)[:3]]) for i in range(step)]
        mapper = LinearColorMapper(palette=colors, low=cluster_edges.density.min(),
                                   high=cluster_edges.density.max())

        p = figure(title="Density between different clusters",
                   x_range=clusters, y_range=clusters,
                   x_axis_location="below", plot_width=960, plot_height=400,
                   tools=TOOLS, toolbar_location='above',
                   tooltips=[('subtopic1', '#@comp_x'), ('subtopic2', '#@comp_y'),
                             ('density', '@density, @value co-citations')])

        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "10pt"
        p.axis.major_label_standoff = 0

        p.rect(x="comp_x", y="comp_y", width=1, height=1,
               source=cluster_edges,
               fill_color={'field': 'density', 'transform': mapper},
               line_color=None)

        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="10pt",
                             formatter=PrintfTickFormatter(format="%.2f"),
                             label_standoff=11, border_line_color=None, location=(0, 0))
        p.add_layout(color_bar, 'right')
        return p

    def cocitations_clustering(self, max_chord_diagram_size=500):
        if self.analyzer.df.shape[0] > max_chord_diagram_size:
            self.clusters_info_message = """
            Heatmap is used to show which subtopics are related to each other.
            Density is based on co-citations between clusters
            and depends on the size of the clusters."""
            return self.heatmap_clusters()

        self.clusters_info_message = """
        Chord diagram is used to show papers as graph nodes,
        edges demonstrate co-citations."""
        return self.chord_diagram_components()

    def component_size_summary(self):
        logging.info('Summary component detailed info visualization')

        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        n_comps = len(self.analyzer.components)

        components = [str(i) for i in range(n_comps)]
        years = list(range(min_year - 1, max_year + 1))
        data = {'years': years}
        for c in range(n_comps):
            data[str(c)] = [
                len(self.analyzer.df[np.logical_and(self.analyzer.df['comp'] == c,
                                                    self.analyzer.df['year'] == y)])
                for y in range(min_year, max_year)]

        p = figure(x_range=[min_year, max_year], plot_width=960, plot_height=300,
                   title="Components by Year",
                   toolbar_location=None, tools="hover", tooltips="Subtopic #$name: @$name")

        p.vbar_stack(components, x='years', width=0.9, color=self.comp_palette, source=data, alpha=0.5,
                     legend=[value(c) for c in components])

        p.y_range.start = 0
        p.xgrid.grid_line_color = None
        p.axis.minor_tick_line_color = None
        p.outline_line_color = None
        p.legend.location = "top_left"
        p.legend.orientation = "horizontal"

        return p

    def subtopic_timeline_graphs(self):
        logging.info('Per component detailed info visualization')

        # Prepare layouts
        n_comps = len(self.analyzer.components)
        self.colors = dict(enumerate(self.comp_palette))
        ds = [None] * n_comps
        p = [None] * n_comps

        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        for c in range(n_comps):
            # Scatter layout for articles from subtopic
            title = f'Subtopic #{c}{" OTHER" if c == 0 and self.analyzer.components_merged else ""}'

            ds[c] = self.__build_data_source(self.analyzer.df[self.analyzer.df['comp'] == c],
                                             width=700)
            plot = self.__serve_scatter_article_layout(source=ds[c],
                                                       year_range=[min_year, max_year],
                                                       title=title, width=760)

            plot.circle(x='year', y='pos', fill_alpha=0.8, source=ds[c], size='size',
                        line_color=self.colors[c], fill_color=self.colors[c])

            # Word cloud description of subtopic by titles and abstracts
            kwds = get_word_cloud_data(self.analyzer.df_kwd, c)
            color = (self.colors[c].r, self.colors[c].g, self.colors[c].b)
            wc = WordCloud(background_color="white", width=200, height=400,
                           color_func=lambda *args, **kwargs: color,
                           max_words=20, max_font_size=40)
            wc.generate_from_frequencies(kwds)

            image = wc.to_array()
            desc = figure(title="", toolbar_location="above",
                          plot_width=200, plot_height=400,
                          x_range=[0, 10], y_range=[0, 10], tools=[])
            desc.axis.visible = False

            img = np.empty((400, 200), dtype=np.uint32)
            view = img.view(dtype=np.uint8).reshape((400, 200, 4))
            view[:, :, 0:3] = image[::-1, :, :]
            view[:, :, 3] = 255

            desc.image_rgba(image=[img], x=[0], y=[0], dw=[10], dh=[10])

            p[c] = row(desc, plot)

        return p

    def top_cited_papers(self):
        n_comps = len(self.analyzer.components)
        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        ds = self.__build_data_source(self.analyzer.top_cited_df, width=700)
        plot = self.__serve_scatter_article_layout(source=ds,
                                                   year_range=[min_year, max_year],
                                                   title=f'{len(self.analyzer.top_cited_df)} top cited papers',
                                                   width=960)

        for c in range(n_comps):
            view = CDSView(source=ds, filters=[GroupFilter(column_name='comp',
                                                           group=str(c))])
            plot.circle(x='year', y='pos', fill_alpha=0.8, source=ds, view=view,
                        size='size', line_color=self.colors[c], fill_color=self.colors[c])

        return plot

    def max_gain_papers(self):
        logging.info('Different colors encode different papers')
        cols = ['year', 'id', 'title', 'authors', 'paper_year', 'count']
        max_gain_df = self.analyzer.max_gain_df[cols].replace(np.nan, "Undefined")
        ds_max = ColumnDataSource(max_gain_df)

        factors = self.analyzer.max_gain_df['id'].unique()
        cmap = plt.cm.get_cmap('jet', len(factors))
        palette = [RGB(*[round(c * 255) for c in cmap(i)[:3]]) for i in range(len(factors))]
        colors = factor_cmap('id', palette=palette, factors=factors)

        year_range = [self.analyzer.min_year - 1, self.analyzer.max_year + 1]
        p = figure(tools=TOOLS, toolbar_location="above",
                   plot_width=960, plot_height=300, x_range=year_range,
                   title='Max gain of citations per year')
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Number of citations'
        p.hover.tooltips = self._html_tooltips([
            ("Author(s)", '@authors'),
            ("Year", '@paper_year'),
            ("Cited by", '@count papers in @year')
        ])
        p.js_on_event('tap', self.pubmed_callback(ds_max, self.analyzer.source))
        p.vbar(x='year', width=0.8, top='count', fill_alpha=0.5, source=ds_max, fill_color=colors,
               line_color=colors)
        return p

    def max_relative_gain_papers(self):
        logging.info('Top papers in relative gain for each year')
        logging.info('Relative gain (year) = Citation Gain (year) / Citations before year')
        logging.info('Different colors encode different papers')
        cols = ['year', 'id', 'title', 'authors', 'paper_year', 'rel_gain']
        max_rel_gain_df = self.analyzer.max_rel_gain_df[cols].replace(np.nan, "Undefined")
        ds_max = ColumnDataSource(max_rel_gain_df)

        factors = self.analyzer.max_rel_gain_df['id'].astype(str).unique()
        cmap = plt.cm.get_cmap('jet', len(factors))
        palette = [RGB(*[round(c * 255) for c in cmap(i)[:3]]) for i in range(len(factors))]
        colors = factor_cmap('id', palette=palette, factors=factors)

        year_range = [self.analyzer.min_year - 1, self.analyzer.max_year + 1]
        p = figure(tools=TOOLS, toolbar_location="above",
                   plot_width=960, plot_height=300, x_range=year_range,
                   title='Max relative gain of citations per year')
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Relative Gain of Citations'
        p.hover.tooltips = self._html_tooltips([
            ("Author(s)", '@authors'),
            ("Year", '@paper_year'),
            ("Relative Gain", '@rel_gain in @year')])
        p.js_on_event('tap', self.pubmed_callback(ds_max, self.analyzer.source))

        p.vbar(x='year', width=0.8, top='rel_gain', fill_alpha=0.5, source=ds_max,
               fill_color=colors, line_color=colors)
        return p

    def article_citation_dynamics(self):
        logging.info('Choose ID to get detailed citations timeline '
                     'for top cited / max gain or relative gain papers')
        highlight_papers = sorted(self.analyzer.top_cited_papers.union(
            self.analyzer.max_gain_papers, self.analyzer.max_rel_gain_papers))

        def update(b):
            id = dropdown.value
            data = self.analyzer.df[self.analyzer.df['id'] == id]

            x = self.analyzer.years
            y = data[x].values[0]

            bar.data_source.data = {'x': x, 'y': y}
            span.location = data['year'].values[0]

            push_notebook(handle=h)

        title = "Number of Citations per Year"
        d = ColumnDataSource(data=dict(x=[], y=[]))

        p = figure(tools=TOOLS, toolbar_location="above", plot_width=960,
                   plot_height=300, title=title)
        bar = p.vbar(x='x', width=0.8, top='y', source=d, color='#A6CEE3', line_width=3)
        span = Span(location=None, dimension='height', line_color='red',
                    line_dash='dashed', line_width=3)
        p.xaxis.axis_label = "Year"
        p.yaxis.axis_label = "Number of citations"
        p.hover.tooltips = [
            ("Year", "@x"),
            ("Cited by", "@y paper(s) in @x"),
        ]
        p.renderers.append(span)

        dropdown = widgets.Dropdown(
            options=list(highlight_papers),
            description='ID:',
            disabled=False
        )

        button = widgets.Button(
            description='Show',
            disabled=False,
            button_style='info',
            tooltip='Show'
        )
        button.on_click(update)

        panel = widgets.HBox([dropdown, button])

        display(panel)
        h = show(p, notebook_handle=True)

    def papers_statistics(self):
        cols = ['year', 'id', 'title', 'authors']
        df_stats = self.analyzer.df[cols].groupby(['year']).size().reset_index(name='counts')
        ds_stats = ColumnDataSource(df_stats)

        year_range = [self.analyzer.min_year - 1, self.analyzer.max_year + 1]
        p = figure(tools=TOOLS, toolbar_location="above",
                   plot_width=760, plot_height=400, x_range=year_range, title='Amount of articles per year')
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Amount of articles'
        p.hover.tooltips = [("Amount", '@counts'), ("Year", '@year')]

        p.vbar(x='year', width=0.8, top='counts', fill_alpha=0.5, source=ds_stats)

        kwds = {}
        for ngram, count in get_most_common_ngrams(self.analyzer.top_cited_df['title'],
                                                   self.analyzer.top_cited_df['abstract']).items():
            for word in ngram.split(' '):
                kwds[word] = float(count) + kwds.get(word, 0)
        wc = WordCloud(background_color="white", width=200, height=400,
                       color_func=lambda *args, **kwargs: 'black',
                       max_words=20, max_font_size=40)
        wc.generate_from_frequencies(kwds)

        image = wc.to_array()
        desc = figure(title="", toolbar_location="above",
                      plot_width=200, plot_height=400,
                      x_range=[0, 10], y_range=[0, 10], tools=[])
        desc.axis.visible = False

        img = np.empty((400, 200), dtype=np.uint32)
        view = img.view(dtype=np.uint8).reshape((400, 200, 4))
        view[:, :, 0:3] = image[::-1, :, :]
        view[:, :, 3] = 255

        desc.image_rgba(image=[img], x=[0], y=[0], dw=[10], dh=[10])
        p = row(desc, p)
        return p

    def subtopic_evolution(self):

        def sort_nodes_key(node):
            y, c = node[0].split(' ')
            return int(y), -int(c)

        cols = self.analyzer.evolution_df.columns[2:]
        pairs = list(zip(cols, cols[1:]))
        nodes = set()
        edges = []

        for now, then in pairs:
            nodes_now = [f'{now} {c}' for c in self.analyzer.evolution_df[now].unique()]
            nodes_then = [f'{then} {c}' for c in self.analyzer.evolution_df[then].unique()]

            inner = {node: 0 for node in nodes_then}
            changes = {node: inner.copy() for node in nodes_now}
            for pmid, comp in self.analyzer.evolution_df.iterrows():
                c_now, c_then = comp[now], comp[then]
                changes[f'{now} {c_now}'][f'{then} {c_then}'] += 1

            for v in nodes_now:
                for u in nodes_then:
                    if changes[v][u] > 0:
                        edges.append((v, u, changes[v][u]))
                        nodes.add(v)
                        nodes.add(u)

        n_steps = len(self.analyzer.evolution_df.columns) - 2
        value_dim = hv.Dimension('Amount', unit=None)

        # One step is not enough to analyze evolution
        if n_steps < 2:
            return None

        nodes_data = []

        for node in nodes:
            year, c = node.split(' ')
            if int(c) >= 0:
                if n_steps < 4:
                    label = f"{year} {', '.join(self.analyzer.evolution_kwds[int(year)][int(c)][:5])}"
                else:
                    label = node
            else:
                label = f"Published after {year}"
            nodes_data.append((node, label))
        nodes_data = sorted(nodes_data, key=sort_nodes_key, reverse=True)

        nodes_ds = hv.Dataset(nodes_data, 'index', 'label')

        topic_evolution = hv.Sankey((edges, nodes_ds), ['From', 'To'], vdims=value_dim)
        topic_evolution.opts(labels='label', width=960, height=600, show_values=False, cmap='tab20',
                             edge_color=dim('To').str(), node_color=dim('index').str())

        if n_steps > 3:
            years = []
            subtopics = []
            keywords = []

            for year, comps in self.analyzer.evolution_kwds.items():
                for c, kwd in comps.items():
                    if c >= 0:
                        years.append(year)
                        subtopics.append(c)
                        keywords.append(', '.join(kwd))

            data = dict(
                years=years,
                subtopics=subtopics,
                keywords=keywords
            )

            source = ColumnDataSource(data)

            columns = [
                TableColumn(field="years", title="Year", width=50),
                TableColumn(field="subtopics", title="Subtopic", width=50),
                TableColumn(field="keywords", title="Keywords", width=800),
            ]

            subtopic_keywords = DataTable(source=source, columns=columns, width=900)

            return column(hv.render(topic_evolution, backend='bokeh'), subtopic_keywords)

        return hv.render(topic_evolution, backend='bokeh')

    def author_statistics(self):
        data = dict(
            author=self.analyzer.author_stats['author'],
            sum=self.analyzer.author_stats['sum'],
            subtopics=self.analyzer.author_stats['comp'].apply(lambda comp: self._to_colored_circle(comp))
        )

        template = """<p <%= subtopics %> </p>"""
        formatter = HTMLTemplateFormatter(template=template)

        source = ColumnDataSource(data)

        columns = [
            TableColumn(field="author", title="Author", width=500),
            TableColumn(field="sum", title="Number of articles", width=100),
            TableColumn(field="subtopics", title="Subtopics", formatter=formatter, width=100, sortable=False),
        ]

        author_stats = DataTable(source=source, columns=columns, width=700)

        return author_stats

    def journal_statistics(self):
        data = dict(
            journal=self.analyzer.journal_stats['journal'],
            sum=self.analyzer.journal_stats['sum'],
            subtopics=self.analyzer.journal_stats['comp'].apply(lambda comp: self._to_colored_circle(comp))
        )

        template = """<p <%= subtopics %> </p>"""
        formatter = HTMLTemplateFormatter(template=template)

        source = ColumnDataSource(data)

        columns = [
            TableColumn(field="journal", title="Journal", width=500),
            TableColumn(field="sum", title='Number of articles', width=100),
            TableColumn(field="subtopics", title='Subtopics', formatter=formatter, width=100, sortable=False),
        ]

        journal_stats = DataTable(source=source, columns=columns, width=700)
        return journal_stats

    def _to_colored_circle(self, components):
        # html code to generate circles corresponding to the 3 most popular subtopics
        return ' '.join(
            map(lambda i: f'''<i class="fas fa-circle" style="color:{self.colors[i]}"></i> ''', components[:3]))

    def __build_data_source(self, df, width=760):
        # Sort papers from the same year with total number of citations as key, use rank as y-pos
        ranks = df.groupby('year')['total'].rank(ascending=False, method='first')

        # Calculate max size of circles to avoid overlapping along x-axis
        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        max_radius_screen_units = width / (max_year - min_year + 1)
        size_scaling_coefficient = max_radius_screen_units / np.log(df['total']).max()

        # NOTE: 'comp' column is used as string because GroupFilter supports
        #       only categorical values (needed to color top cited papers by components)
        d = ColumnDataSource(data=dict(id=df['id'], title=df['title'], authors=df['authors'],
                                       year=df['year'].replace(np.nan, "Undefined"),
                                       total=df['total'], comp=df['comp'].astype(str), pos=ranks,
                                       size=np.log(df['total']) * size_scaling_coefficient))
        return d

    def __serve_scatter_article_layout(self, source, year_range, title, width=960):
        min_year, max_year = year_range
        p = figure(tools=TOOLS, toolbar_location="above",
                   plot_width=width, plot_height=400,
                   x_range=(min_year - 1, max_year + 1),
                   title=title)
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Amount of articles'
        p.y_range.start = 0

        p.hover.tooltips = self._html_tooltips([
            ("Author(s)", '@authors'),
            ("Year", '@year'),
            ("Cited by", '@total paper(s) total')])
        p.js_on_event('tap', self.pubmed_callback(source, self.analyzer.source))

        return p

    def _add_pmid(self, tips_list):
        if self.analyzer.source == "pubmed":
            tips_list.insert(0, ("PMID", '@id'))

        return tips_list

    def _html_tooltips(self, tips_list):
        tips_list = self._add_pmid(tips_list)

        style_caption = Template('<span style="font-size: 12px;color:dodgerblue;">$caption:</span>')
        style_value = Template('<span style="font-size: 11px;">$value</span>')

        tips_list_html = '\n'.join([f'''
            <div> {style_caption.substitute(caption=tip[0])} {style_value.substitute(value=tip[1])} </div>'''
                                    for tip in tips_list])

        html_tooltips_str = f'''
               <div style="max-width: 320px">
                   <div>
                       <span style="font-size: 13px; font-weight: bold;">@title</span>
                   </div>
                   {tips_list_html}
               </div>
            '''
        return html_tooltips_str
