import html
import json
import logging
from string import Template

import holoviews as hv
import numpy as np
from bokeh.colors import RGB
from bokeh.core.properties import value
from bokeh.embed import components
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS, NodesAndLinkedEdges, TapTool
from bokeh.models import GraphRenderer, StaticLayoutProvider
# Tools used: hover,pan,tap,wheel_zoom,box_zoom,reset,save
from bokeh.models import HoverTool, PanTool, WheelZoomTool, BoxZoomTool, ResetTool, SaveTool
from bokeh.models import LinearColorMapper, PrintfTickFormatter, ColorBar
from bokeh.models import NumeralTickFormatter
from bokeh.models import Plot, Range1d, MultiLine, Circle
from bokeh.models.widgets.tables import DataTable
from bokeh.palettes import Category20
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from holoviews import dim
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from .utils import LOCAL_BASE_URL, get_word_cloud_data, \
    get_most_common_ngrams, cut_authors_list, ZOOM_OUT, ZOOM_IN, zoom_name
from .visualization_data import PlotPreprocessor

TOOLS = "hover,pan,tap,wheel_zoom,box_zoom,reset,save"
hv.extension('bokeh')


def visualize_analysis(analyzer):
    # Initialize plotter after completion of analysis
    plotter = Plotter(analyzer=analyzer)
    # Order is important here!
    paper_statistics, zoom_out_callback = plotter.papers_statistics_and_zoom_out_callback()
    result = {
        'log': html.unescape(analyzer.log()),
        'experimental': analyzer.config.experimental,
        'n_papers': analyzer.n_papers,
        'n_citations': int(analyzer.df['total'].sum()),
        'n_subtopics': len(analyzer.components),
        'comp_other': analyzer.comp_other,
        'cocitations_clusters': [components(plotter.cocitations_clustering())],
        'component_size_summary': [components(plotter.component_size_summary())],
        'component_years_summary_boxplots': [components(plotter.component_years_summary_boxplots())],
        'subtopics_infos_and_zoom_in_callbacks':
            [(components(p), zoom_in_callback) for
             (p, zoom_in_callback) in plotter.subtopics_infos_and_zoom_in_callbacks()],
        'top_cited_papers': [components(plotter.top_cited_papers())],
        'max_gain_papers': [components(plotter.max_gain_papers())],
        'max_relative_gain_papers': [components(plotter.max_relative_gain_papers())],
        'component_sizes': plotter.component_sizes(),
        'component_ratio': [components(plotter.component_ratio())],
        'papers_stats': [components(paper_statistics)],
        'papers_zoom_out_callback': zoom_out_callback,
        'clusters_info_message': html.unescape(plotter.clusters_info_message),
        'author_statistics': plotter.author_statistics(),
        'journal_statistics': plotter.journal_statistics()
    }
    # Experimental features
    if analyzer.config.experimental:
        subtopic_evolution = plotter.subtopic_evolution()
        # Pass subtopic evolution only if not None
        if subtopic_evolution:
            result['subtopic_evolution'] = [components(subtopic_evolution)]
    return result


class Plotter:
    def __init__(self, analyzer=None):
        self.analyzer = analyzer

        if self.analyzer:
            n_comps = len(self.analyzer.components)
            if n_comps > 20:
                raise ValueError(f'Too big number of components {n_comps}')
            self.comp_palette = [RGB(*PlotPreprocessor.hex2rgb(c)) for c in Category20[20][:n_comps]]
            self.comp_colors = dict(enumerate(self.comp_palette))

            n_pub_types = len(self.analyzer.pub_types)
            pub_types_cmap = plt.cm.get_cmap('jet', n_pub_types)
            self.pub_types_colors_map = dict(
                zip(self.analyzer.pub_types,
                    [RGB(*[round(c * 255) for c in pub_types_cmap(i)[:3]]) for i in range(n_pub_types)]))

    @staticmethod
    def paper_callback(source, db):
        if db in ['Semantic Scholar', 'Pubmed']:
            base = LOCAL_BASE_URL.substitute(source=db)
        else:
            raise ValueError("Wrong value of db")
        return CustomJS(args=dict(source=source, base=base), code="""
            var data = source.data, selected = source.selected.indices;

            // Decode jobid from URL, which is supposed to be last
            var tokens = window.location.href.split('&');
            var jobid = tokens[tokens.length - 1];

            // Max amount of papers to be opened, others will be ignored
            var MAX_AMOUNT = 3;

            for (var i = 0; i < Math.min(MAX_AMOUNT, selected.length); i++){
                window.open(base + data['id'][selected[i]] + '&' + jobid, "_blank");
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

    @staticmethod
    def zoom_callback(id_list, source, zoom, query):
        # check zoom value
        zoom_name(zoom)
        # submit list of ids and database name to the main page using invisible form
        # IMPORTANT: no double quotes!
        return f"""
        var form = document.createElement('form');
        document.body.appendChild(form);
        form.method = 'post';
        form.action = '/process_ids';
        form.target = '_blank'

        var input = document.createElement('input');
        input.type = 'hidden';
        input.name = 'id_list';
        input.value = {json.dumps(id_list).replace('"', "'")};
        form.appendChild(input);

        var input = document.createElement('input');
        input.type = 'hidden';
        input.name = 'source';
        input.value = '{source}';
        form.appendChild(input);

        var input = document.createElement('input');
        input.type = 'hidden';
        input.name = 'zoom';
        input.value = '{zoom}';
        form.appendChild(input);

        var input = document.createElement('input');
        input.type = 'hidden';
        input.name = 'query';
        input.value = {json.dumps(query).replace('"', "'")};
        form.appendChild(input);

        form.submit();
        """

    def chord_diagram_components(self):
        logging.info('Visualizing components with Chord diagram')

        layout, node_data_source, edge_data_source = PlotPreprocessor.chord_diagram_data(
            self.analyzer.CG, self.analyzer.df, self.analyzer.partition, self.analyzer.comp_other, self.comp_palette
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
            self.analyzer.CG, self.analyzer.df, self.analyzer.comp_sizes
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
            Density is based on co-citations between clusters and depends on the size of the clusters."""
            return self.heatmap_clusters()

        self.clusters_info_message = """
        Chord diagram is used to show papers as graph nodes, and edges demonstrate co-citations.
        Click on any node to highlight all incident edges."""
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
                     legend=[f'{c} OTHER' if int(c) - 1 == self.analyzer.comp_other else value(c)
                             for c in components])

        # VArea is actually displayed
        p.varea_stack(stackers=components, x='years', color=self.comp_palette, source=data, alpha=0.5,
                      legend=[f'{c} OTHER' if int(c) - 1 == self.analyzer.comp_other else value(c)
                              for c in components])

        p.y_range.start = 0
        p.xgrid.grid_line_color = None
        p.axis.minor_tick_line_color = None
        p.outline_line_color = None
        p.legend.location = "top_left"
        p.legend.orientation = "horizontal"

        return p

    def component_years_summary_boxplots(self):
        logging.info('Summary component year detailed info visualization on boxplot')

        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        components, data = PlotPreprocessor.component_size_summary_data(
            self.analyzer.df, self.analyzer.components, min_year, max_year
        )
        labels = []
        values = []
        for c in components:
            vs = data[c]
            expanded_vs = []
            for i, y in enumerate(range(min_year, max_year + 1)):
                expanded_vs.extend([y for _ in range(vs[i])])
            labels.extend([c for _ in range(len(expanded_vs))])
            values.extend(expanded_vs)
        boxwhisker = hv.BoxWhisker((labels, values), 'Topic', 'Publications year')
        boxwhisker.opts(width=960, height=300, box_fill_color=dim('Topic').str(), cmap='tab20')
        return hv.render(boxwhisker, backend='bokeh')

    def subtopics_infos_and_zoom_in_callbacks(self):
        logging.info('Per component detailed info visualization')

        # Prepare layouts
        n_comps = len(self.analyzer.components)
        result = []

        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        for c in range(n_comps):
            comp_source = self.analyzer.df[self.analyzer.df['comp'] == c]
            ds = PlotPreprocessor.article_view_data_source(
                comp_source, min_year, max_year, True, width=700
            )
            # Add type coloring
            ds.add([self.pub_types_colors_map[t] for t in comp_source['type']], 'color')
            plot = self.__serve_scatter_article_layout(source=ds,
                                                       year_range=[min_year, max_year],
                                                       title="Publications", width=760)
            plot.circle(x='year', y='total_fixed', fill_alpha=0.5, source=ds, size='size',
                        line_color='color', fill_color='color', legend='type')
            plot.legend.location = "top_left"

            # Word cloud description of subtopic by titles and abstracts
            kwds = get_word_cloud_data(self.analyzer.df_kwd, c)
            color = (self.comp_colors[c].r, self.comp_colors[c].g, self.comp_colors[c].b)
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

            # Create Zoom In callback
            zoom_in_callback = self.zoom_callback(list(comp_source['id']), self.analyzer.source,
                                                  zoom=ZOOM_IN,
                                                  query=self.analyzer.query)
            result.append((row(desc, plot), zoom_in_callback))

        return result

    def component_sizes(self):
        assigned_comps = self.analyzer.df[self.analyzer.df['comp'] >= 0]
        d = dict(assigned_comps.groupby('comp')['id'].count())
        return [int(d[k]) for k in range(len(d))]

    def component_ratio(self):
        comps, source = PlotPreprocessor.component_ratio_data(
            self.analyzer.df, self.comp_palette
        )

        p = figure(plot_width=900, plot_height=50 * len(comps), toolbar_location="above", tools=TOOLS, y_range=comps)
        p.hbar(y='comps', right='ratios', height=0.9, fill_alpha=0.5, color='colors', source=source)
        p.hover.tooltips = [("Subtopic", '@comps'), ("Amount", '@ratios %')]

        p.x_range.start = 0
        p.xaxis.axis_label = 'Percentage of papers'
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
            self.analyzer.top_cited_df, min_year, max_year, False, width=700
        )
        # Add type coloring
        ds.add([self.pub_types_colors_map[t] for t in self.analyzer.top_cited_df['type']], 'color')

        plot = self.__serve_scatter_article_layout(source=ds,
                                                   year_range=[min_year, max_year],
                                                   title=f'{len(self.analyzer.top_cited_df)} top cited papers',
                                                   width=960)

        plot.circle(x='year', y='total_fixed', fill_alpha=0.5, source=ds, size='size',
                    line_color='color', fill_color='color', legend='type')
        plot.legend.location = "top_left"
        return plot

    def max_gain_papers(self):
        logging.info('Different colors encode different papers')
        cols = ['year', 'id', 'title', 'authors', 'paper_year', 'count']
        max_gain_df = self.analyzer.max_gain_df[cols].replace(np.nan, "Undefined")
        max_gain_df['authors'] = max_gain_df['authors'].apply(lambda authors: cut_authors_list(authors))
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
        p.js_on_event('tap', self.paper_callback(ds_max, self.analyzer.source))
        p.vbar(x='year', width=0.8, top='count', fill_alpha=0.5, source=ds_max, fill_color=colors,
               line_color=colors)
        return p

    def max_relative_gain_papers(self):
        logging.info('Top papers in relative gain for each year')
        logging.info('Relative gain (year) = Citation Gain (year) / Citations before year')
        logging.info('Different colors encode different papers')
        cols = ['year', 'id', 'title', 'authors', 'paper_year', 'rel_gain']
        max_rel_gain_df = self.analyzer.max_rel_gain_df[cols].replace(np.nan, "Undefined")
        max_rel_gain_df['authors'] = max_rel_gain_df['authors'].apply(lambda authors: cut_authors_list(authors))
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
        p.js_on_event('tap', self.paper_callback(ds_max, self.analyzer.source))

        p.vbar(x='year', width=0.8, top='rel_gain', fill_alpha=0.5, source=ds_max,
               fill_color=colors, line_color=colors)
        return p

    @staticmethod
    def article_citation_dynamics(df, pid):
        d = PlotPreprocessor.article_citation_dynamics_data(df, pid)

        p = figure(tools=TOOLS, toolbar_location="above", plot_width=960,
                   plot_height=300, title="Number of Citations per Year")
        p.vbar(x='x', width=0.8, top='y', source=d, color='#A6CEE3', line_width=3)
        p.xaxis.axis_label = "Year"
        p.yaxis.axis_label = "Number of citations"
        p.hover.tooltips = [
            ("Year", "@x"),
            ("Cited by", "@y paper(s) in @x"),
        ]

        return p

    def papers_statistics_and_zoom_out_callback(self):
        ds_stats = PlotPreprocessor.papers_statistics_data(self.analyzer.df)

        year_range = [self.analyzer.min_year - 1, self.analyzer.max_year + 1]
        p = figure(tools=TOOLS, toolbar_location="above",
                   plot_width=760, plot_height=400, x_range=year_range, title='Amount of papers per year')
        p.y_range.start = 0
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Amount of papers'
        p.hover.tooltips = [("Amount", '@counts'), ("Year", '@year')]

        # NOTE: VBar is invisible (alpha=0) to provide tooltips, as in self.component_size_summary()
        p.vbar(x='year', width=0.8, top='counts', fill_alpha=0, line_alpha=0, source=ds_stats)

        # VArea is actually displayed
        ds_stats.data['bottom'] = [0] * len(ds_stats.data['year'])
        p.varea(x='year', y1='bottom', y2='counts', fill_alpha=0.5, source=ds_stats)

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

        # Create Zoom Out callback
        zoom_out_callback = self.zoom_callback(list(self.analyzer.df['id']), self.analyzer.source,
                                               zoom=ZOOM_OUT, query=self.analyzer.query)

        return row(desc, p), zoom_out_callback

    def subtopic_evolution(self):
        """
        Sankey diagram of subtopic evolution
        :return:
            if self.analyzer.evolution_df is None: None, as no evolution can be observed in 1 step
            if number of steps < 3: Sankey diagram
            else: Sankey diagram + table with keywords
        """
        # Subtopic evolution analysis failed, one step is not enough to analyze evolution
        if self.analyzer.evolution_df is None or not self.analyzer.evolution_kwds:
            return None

        n_steps = len(self.analyzer.evolution_df.columns) - 2

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
        author = self.analyzer.author_stats['author']
        sum = self.analyzer.author_stats['sum']
        subtopics = self.analyzer.author_stats.apply(
            lambda row: self._to_colored_circle(row['comp'], row['counts'], row['sum']), axis=1)
        return list(zip(author, sum, subtopics))

    def journal_statistics(self):
        journal = self.analyzer.journal_stats['journal']
        sum = self.analyzer.journal_stats['sum']
        subtopics = self.analyzer.journal_stats.apply(
            lambda row: self._to_colored_circle(row['comp'], row['counts'], row['sum']), axis=1)
        return list(zip(journal, sum, subtopics))

    def _to_colored_circle(self, components, counts, sum):
        # html code to generate circles corresponding to the 3 most popular subtopics
        top = 3
        return ' '.join(
            map(lambda topic: f'''<a class="fas fa-circle" style="color:{self.comp_colors[topic[0]]}"
                                     href="#subtopic-{topic[0] + 1}"></a>
                                  <span class="bk" style="color:black">{int(topic[1] / sum * 100)}%</span> ''',
                zip(components[:top], counts[:top])))

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
                   title=title, y_axis_type="log")
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Number of citations'
        p.yaxis.formatter = NumeralTickFormatter(format='0,0')

        p.hover.tooltips = self._html_tooltips([
            ("Author(s)", '@authors'),
            ("Year", '@year'),
            ("Type", '@type'),
            ("Cited by", '@total paper(s) total')])
        p.js_on_event('tap', self.paper_callback(source, self.analyzer.source))

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
