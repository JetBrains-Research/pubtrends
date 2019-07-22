import logging
import math
from string import Template

import ipywidgets as widgets
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import display
from bokeh.colors import RGB
from bokeh.core.properties import value
from bokeh.io import push_notebook
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, CDSView, GroupFilter, CustomJS
from bokeh.models import GraphRenderer, StaticLayoutProvider
# Tools used: hover,pan,tap,wheel_zoom,box_zoom,reset,save
from bokeh.models import HoverTool, PanTool, WheelZoomTool, BoxZoomTool, ResetTool, SaveTool
from bokeh.models import Plot, Range1d, MultiLine, Circle, Span
from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from .utils import PUBMED_ARTICLE_BASE_URL, SEMANTIC_SCHOLAR_BASE_URL, get_word_cloud_data, get_most_common_ngrams

TOOLS = "hover,pan,tap,wheel_zoom,box_zoom,reset,save"


class Plotter:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.index = analyzer.loader.index

    @staticmethod
    def pubmed_callback(source, index):
        if index == 'ssid':
            base = SEMANTIC_SCHOLAR_BASE_URL
        elif index == 'pmid':
            base = PUBMED_ARTICLE_BASE_URL
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
        gdf = pd.merge(pd.Series(self.analyzer.CG.nodes(), dtype=object).reset_index().rename(columns={0: 'id'}),
                       self.analyzer.df[['id', 'title', 'authors', 'year', 'total', 'comp']], how='left'
                       ).sort_values(by='total', ascending=False)

        for c in range(len(self.analyzer.components)):
            for n in gdf[gdf['comp'] == c]['id']:
                # NOTE: we use nodes id as String to avoid problems str keys in jsonify during graph visualization
                G.add_node(str(n))

        edge_starts = []
        edge_ends = []
        edge_weights = []
        for start, end, data in self.analyzer.CG.edges(data=True):
            edge_starts.append(start)
            edge_ends.append(end)
            edge_weights.append(min(data['weight'], 20))

        n_comps = len(self.analyzer.components)
        cmap = plt.cm.get_cmap('jet', n_comps)
        comp_palette = [RGB(*[round(c * 255) for c in cmap(i)[:3]]) for i in range(n_comps)]

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
                edge_colors.append(comp_palette[self.analyzer.pm[edge_start]])
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
        graph.node_renderer.data_source.data['colors'] = [comp_palette[self.analyzer.pm[n]] for n in G.nodes()]
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
                                           line_alpha=0.5, fill_alpha=0.5)
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

    def component_size_summary(self):
        logging.info('Summary component detailed info visualization')

        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        n_comps = len(self.analyzer.components)
        cmap = plt.cm.get_cmap('jet', n_comps)
        palette = [RGB(*[round(c * 255) for c in cmap(i)[:3]]) for i in range(n_comps)]

        components = [str(i) for i in range(n_comps)]
        years = list(range(min_year - 1, max_year + 1))
        data = {'years': years}
        for c in range(n_comps):
            data[str(c)] = [
                len(self.analyzer.df[np.logical_and(self.analyzer.df['comp'] == c, self.analyzer.df['year'] == y)])
                for y in range(min_year, max_year)]

        p = figure(x_range=[min_year, max_year], plot_width=960, plot_height=300, title="Components by Year",
                   toolbar_location=None, tools="hover", tooltips="Subtopic #$name: @$name")

        p.vbar_stack(components, x='years', width=0.9, color=palette, source=data, alpha=0.5,
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

        #        # Reorder subtopics by importance descending
        #        KEY = 'citations' # 'size' or 'citations'
        #
        #        if KEY == 'size':
        #            order = self.analyzer.df.groupby('comp')['pmid'].count().sort_values(ascending=False).index.values
        #        elif KEY == 'citations':
        #            order = self.analyzer.df.groupby('comp')['total'].sum().sort_values(ascending=False).index.values

        # Prepare layouts
        n_comps = len(self.analyzer.components)
        cmap = plt.cm.get_cmap('jet', n_comps)
        self.colors = {c: RGB(*[int(round(ch * 255)) for ch in cmap(c)[:3]]) \
                       for c in self.analyzer.components}
        ds = [None] * n_comps
        p = [None] * n_comps

        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        for c in range(n_comps):
            # Scatter layout for articles from subtopic
            title = f'Subtopic #{c}{" OTHER" if c == 0 and self.analyzer.components_merged else ""}'

            ds[c] = self.__build_data_source(self.analyzer.df[self.analyzer.df['comp'] == c], width=700)
            plot = self.__serve_scatter_article_layout(source=ds[c],
                                                       year_range=[min_year, max_year],
                                                       title=title, width=760)

            plot.circle(x='year', y='pos', fill_alpha=0.5, source=ds[c], size='size',
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
                                                   width=760)

        for c in range(n_comps):
            view = CDSView(source=ds, filters=[GroupFilter(column_name='comp',
                                                           group=str(c))])
            plot.circle(x='year', y='pos', fill_alpha=0.5, source=ds, view=view,
                        size='size', line_color=self.colors[c], fill_color=self.colors[c])

        # Word cloud description of subtopic by titles and abstracts
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

        p = row(desc, plot)
        return p

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
                   plot_width=960, plot_height=300, x_range=year_range, title='Max gain of citations per year')
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Number of citations'
        p.hover.tooltips = self._html_tooltips([
            ("Author(s)", '@authors'),
            ("Year", '@paper_year'),
            ("Cited by", '@count papers in @year')
        ])
        p.js_on_event('tap', self.pubmed_callback(ds_max, self.index))
        p.vbar(x='year', width=0.8, top='count', fill_alpha=0.5, source=ds_max, fill_color=colors, line_color=colors)
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
                   plot_width=960, plot_height=300, x_range=year_range, title='Max relative gain of citations per year')
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Relative Gain of Citations'
        p.hover.tooltips = self._html_tooltips([
            ("Author(s)", '@authors'),
            ("Year", '@paper_year'),
            ("Relative Gain", '@rel_gain in @year')])
        p.js_on_event('tap', self.pubmed_callback(ds_max, self.index))

        p.vbar(x='year', width=0.8, top='rel_gain', fill_alpha=0.5, source=ds_max, fill_color=colors, line_color=colors)
        return p

    def article_citation_dynamics(self):
        logging.info('Choose ID to get detailed citations timeline for top cited / max gain or relative gain papers')
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

    def subtopic_evolution(self):
        plt.rcParams['figure.figsize'] = 20, 10
        pd.plotting.parallel_coordinates(self.analyzer.evolution_df,
                                         'current', sort_labels=True)
        plt.xlabel('Year')
        plt.ylabel('Component ID')
        plt.grid(b=True, which='both', linestyle='--')
        plt.legend().set_visible(False)
        return plt

    def __build_data_source(self, df, width=760):
        ARTICLE_PLOT_WIDTH = width  # Width of the plot (without axis borders)

        # Sort papers from the same year with total number of citations as key, use rank as y-pos
        ranks = df.groupby('year')['total'].rank(ascending=False, method='first')

        # Calculate max size of circles to avoid overlapping along x-axis
        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        max_radius_screen_units = ARTICLE_PLOT_WIDTH / (max_year - min_year + 1)
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
        p.js_on_event('tap', self.pubmed_callback(source, self.index))

        return p

    def _add_pmid(self, tips_list):
        if self.index == "pmid":
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
