import logging
from string import Template

import holoviews as hv
import ipywidgets as widgets
import numpy as np
from IPython.display import display
from bokeh.colors import RGB
from bokeh.core.properties import value
from bokeh.io import push_notebook
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS, HTMLTemplateFormatter, NodesAndLinkedEdges, TapTool
from bokeh.models import GraphRenderer, StaticLayoutProvider
# Tools used: hover,pan,tap,wheel_zoom,box_zoom,reset,save
from bokeh.models import HoverTool, PanTool, WheelZoomTool, BoxZoomTool, ResetTool, SaveTool
from bokeh.models import LinearColorMapper, PrintfTickFormatter, ColorBar
from bokeh.models import NumeralTickFormatter
from bokeh.models import Plot, Range1d, MultiLine, Circle, Span
from bokeh.models.widgets.tables import DataTable, TableColumn
from bokeh.palettes import Category20
from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap
from holoviews import dim
from matplotlib import pyplot as plt
from pandas import RangeIndex
from wordcloud import WordCloud

from .utils import PUBMED_ARTICLE_BASE_URL, SEMANTIC_SCHOLAR_BASE_URL, get_word_cloud_data, \
    get_most_common_ngrams
from .visualization_data import PlotPreprocessor

TOOLS = "hover,pan,tap,wheel_zoom,box_zoom,reset,save"
hv.extension('bokeh')


class Plotter:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.comp_palette = []
        n_comps = len(self.analyzer.components)
        for color in Category20[20][:n_comps]:
            self.comp_palette.append(RGB(*PlotPreprocessor.hex2rgb(color)))

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
                    if (data['id'][i] == selected_id) {
                        window.open(base + data['id'][i], '_blank');
                        // avoid opening multiple tabs with the same article
                        break;
                    }
                }
            }
        """)

    @staticmethod
    def subtopic_callback(source):
        return CustomJS(args=dict(source=source), code="""
            var data = source.data, selected = source.selected.indices;
            if (selected.length == 1) {
                // only consider case where one glyph is selected by user
                selected_comp = data['comps'][selected[0]];
                window.location.hash = '#subtopic-' + selected_comp;
            }
            source.selected.indices = [];
        """)

    def chord_diagram_components(self):
        logging.info('Visualizing components with Chord diagram')

        layout, node_data_source, edge_data_source = PlotPreprocessor.chord_diagram_data(
            self.analyzer.CG, self.analyzer.df, self.analyzer.pm, self.analyzer.comp_other, self.comp_palette
        )

        plot = Plot(plot_width=800, plot_height=800,
                    x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
        plot.title.text = 'Co-citations graph'

        graph = GraphRenderer()

        # Data for nodes & edges
        # Assignment `graph.node_renderer.data_source = node_data_source` leads to an empty graph, do not use it!
        graph.node_renderer.data_source.data = node_data_source.data
        graph.edge_renderer.data_source.data = edge_data_source.data

        # Node layout
        graph.layout_provider = StaticLayoutProvider(graph_layout=layout)

        # Node rendering
        graph.node_renderer.glyph = Circle(size='size', line_color='colors', fill_color='colors',
                                           line_alpha=0.5, fill_alpha=0.8)
        graph.node_renderer.selection_glyph = Circle(size='size', line_color='#91C82F', fill_color='#91C82F',
                                                     line_alpha=0.5, fill_alpha=0.8)

        # Edge rendering
        graph.edge_renderer.glyph = MultiLine(line_color='edge_colors',
                                              line_alpha='edge_alphas',
                                              line_width='edge_weights')
        graph.edge_renderer.selection_glyph = MultiLine(line_color='#91C82F',
                                                        line_alpha=0.5,
                                                        line_width='edge_weights')

        graph.selection_policy = NodesAndLinkedEdges()

        # Add tools to the plot
        # hover,pan,tap,wheel_zoom,box_zoom,reset,save
        plot.add_tools(HoverTool(tooltips=self._html_tooltips([
            ("Author(s)", '@authors'),
            ("Year", '@year'),
            ("Cited by", '@total paper(s) total'),
            ("Subtopic", '@topic')])),
            PanTool(), TapTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), SaveTool())

        plot.renderers.append(graph)
        return plot

    def heatmap_clusters(self):
        logging.info('Visualizing components with heatmap')

        cluster_edges, clusters = PlotPreprocessor.heatmap_clusters_data(
            self.analyzer.CG, self.analyzer.df, self.analyzer.pmcomp_sizes
        )

        step = 30
        cmap = plt.cm.get_cmap('PuBu', step)
        colors = [RGB(*[round(c * 255) for c in cmap(i)[:3]]) for i in range(step)]
        mapper = LinearColorMapper(palette=colors,
                                   low=cluster_edges.density.min(),
                                   high=cluster_edges.density.max())

        p = figure(title="Density between different clusters",
                   x_range=clusters, y_range=clusters,
                   x_axis_location="below", plot_width=960, plot_height=400,
                   tools=TOOLS, toolbar_location='above',
                   tooltips=[('Subtopic 1', '#@comp_x'), ('Subtopic 2', '#@comp_y'),
                             ('Density', '@density, @value co-citations')])

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
        if len(self.analyzer.df) > max_chord_diagram_size:
            self.clusters_info_message = """
            Heatmap is used to show which subtopics are related to each other.
            Density is based on co-citations between clusters
            and depends on the size of the clusters."""
            return self.heatmap_clusters()

        self.clusters_info_message = """
        Chord diagram is used to show papers as graph nodes,
        edges demonstrate co-citations. Click on any node
        to highlight all incident edges."""
        return self.chord_diagram_components()

    def component_size_summary(self):
        logging.info('Summary component detailed info visualization')

        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        components, data = PlotPreprocessor.component_size_summary_data(
            self.analyzer.df, self.analyzer.components, min_year, max_year
        )

        p = figure(x_range=[min_year - 1, max_year + 1], plot_width=960, plot_height=300,
                   title="Components by Year", toolbar_location="right", tools=TOOLS,
                   tooltips=[('Subtopic', '$name'), ('Amount', '@$name')])

        # NOTE: VBar is invisible (alpha = 0) to provide tooltips on hover as stacked area does not support them
        p.vbar_stack(components, x='years', width=0.9, color=self.comp_palette, source=data, alpha=0,
                     legend=[f'{c} OTHER' if int(c) == self.analyzer.comp_other else value(c)
                             for c in components])

        # VArea is actually displayed
        p.varea_stack(stackers=components, x='years', color=self.comp_palette, source=data, alpha=0.5,
                      legend=[f'{c} OTHER' if int(c) == self.analyzer.comp_other else value(c)
                              for c in components])

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
            title = f'Subtopic #{c}{" OTHER" if c == self.analyzer.comp_other else ""}'

            ds[c] = PlotPreprocessor.article_view_data_source(self.analyzer.df[self.analyzer.df['comp'] == c],
                                                              min_year, max_year, width=700)
            plot = self.__serve_scatter_article_layout(source=ds[c],
                                                       year_range=[min_year, max_year],
                                                       title=title, width=760)

            plot.circle(x='year', y='total', fill_alpha=0.5, source=ds[c], size='size',
                        line_color=self.colors[c], fill_color=self.colors[c])

            # Word cloud description of subtopic by titles and abstracts
            kwds = get_word_cloud_data(self.analyzer.df_kwd, c)
            color = (self.colors[c].r, self.colors[c].g, self.colors[c].b)
            wc = WordCloud(background_color="white", width=200, height=400,
                           color_func=lambda *args, **kwargs: color,
                           max_words=20, max_font_size=40)
            wc.generate_from_frequencies(kwds)

            image = wc.to_array()
            desc = figure(title="Word Cloud", toolbar_location="above",
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

    def component_ratio(self):
        comps, source = PlotPreprocessor.component_ratio_data(
            self.analyzer.df, self.comp_palette
        )

        p = figure(plot_width=900, plot_height=50 * len(comps), toolbar_location="above", tools=TOOLS, y_range=comps)
        p.hbar(y='comps', right='ratios', height=0.9, fill_alpha=0.5, color='colors', source=source)
        p.hover.tooltips = [("Subtopic", '@comps'), ("Amount", '@ratios %')]

        p.x_range.start = 0
        p.xaxis.axis_label = 'Percentage of articles'
        p.yaxis.axis_label = 'Subtopic'
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.axis.minor_tick_line_color = None
        p.outline_line_color = None
        p.js_on_event('tap', self.subtopic_callback(source))

        return p

    def top_cited_papers(self):
        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        ds = PlotPreprocessor.article_view_data_source(
            self.analyzer.top_cited_df, min_year, max_year, width=700
        )
        plot = self.__serve_scatter_article_layout(source=ds,
                                                   year_range=[min_year, max_year],
                                                   title=f'{len(self.analyzer.top_cited_df)} top cited papers',
                                                   width=960)

        plot.circle(x='year', y='total', fill_alpha=0.5, source=ds,
                    size='size', line_color='blue')

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
        ds_stats = PlotPreprocessor.papers_statistics_data(self.analyzer.df)

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
        desc = figure(title="Word Cloud", toolbar_location="above",
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
        n_steps = len(self.analyzer.evolution_df.columns) - 2

        # One step is not enough to analyze evolution
        if n_steps < 2:
            return None

        edges, nodes_data = PlotPreprocessor.subtopic_evolution_data(
            self.analyzer.evolution_df, self.analyzer.evolution_kwds, n_steps
        )

        value_dim = hv.Dimension('Amount', unit=None)
        nodes_ds = hv.Dataset(nodes_data, 'index', 'label')
        topic_evolution = hv.Sankey((edges, nodes_ds), ['From', 'To'], vdims=value_dim)
        topic_evolution.opts(labels='label', width=960, height=600, show_values=False, cmap='tab20',
                             edge_color=dim('To').str(), node_color=dim('index').str())

        if n_steps > 3:
            columns, source = PlotPreprocessor.subtopic_evolution_keywords_data(self.analyzer.evolution_kwds)
            subtopic_keywords = DataTable(source=source, columns=columns, width=900, index_position=None)

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
        source.add(RangeIndex(start=1, stop=self.analyzer.author_stats.shape[0], step=1), 'index')

        columns = [
            TableColumn(field="index", title="#", width=20),
            TableColumn(field="author", title="Author", width=500),
            TableColumn(field="sum", title="Number of articles", width=100),
            TableColumn(field="subtopics", title="Subtopics", formatter=formatter, width=100, sortable=False),
        ]

        author_stats = DataTable(source=source, columns=columns, width=700, index_position=None)
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
        source.add(RangeIndex(start=1, stop=self.analyzer.journal_stats.shape[0], step=1), 'index')

        columns = [
            TableColumn(field="index", title="#", width=20),
            TableColumn(field="journal", title="Journal", width=500),
            TableColumn(field="sum", title='Number of articles', width=100),
            TableColumn(field="subtopics", title='Subtopics', formatter=formatter, width=100, sortable=False),
        ]

        journal_stats = DataTable(source=source, columns=columns, width=700, index_position=None)
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
                   title=title,
                   y_axis_type="log")
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Number of citations'
        p.yaxis.formatter = NumeralTickFormatter(format='0,0')

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
