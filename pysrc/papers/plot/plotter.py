import json
import logging
import re
from collections import Counter
from string import Template
import math
from math import pi, sin, cos, fabs

import holoviews as hv
import networkx as nx
import numpy as np
from bokeh.colors import RGB
from bokeh.embed import components
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models import GraphRenderer, StaticLayoutProvider, Circle, HoverTool, MultiLine
from bokeh.models import LinearColorMapper, PrintfTickFormatter, ColorBar
from bokeh.models import NumeralTickFormatter
from bokeh.models.graphs import NodesAndLinkedEdges
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from holoviews import dim
from holoviews import opts
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from pysrc.papers.analysis.graph import local_sparse
from pysrc.papers.analysis.text import get_frequent_tokens, get_topic_word_cloud_data
from pysrc.papers.analysis.topics import convert_clusters_dendrogram_to_paths, compute_clusters_dendrogram_children
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.plot.plot_preprocessor import PlotPreprocessor
from pysrc.papers.utils import cut_authors_list, ZOOM_OUT, ZOOM_IN, zoom_name, trim, rgb2hex, MAX_TITLE_LENGTH, \
    contrast_color

TOOLS = "hover,pan,tap,wheel_zoom,box_zoom,reset,save"
hv.extension('bokeh')

logger = logging.getLogger(__name__)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

MAX_AUTHOR_LENGTH = 100
MAX_JOURNAL_LENGTH = 100
MAX_LINEAR_AXIS = 100

PLOT_WIDTH = 870
PAPERS_PLOT_WIDTH = 670

SHORT_PLOT_HEIGHT = 300
TALL_PLOT_HEIGHT = 600
PLOT_HEIGHT = 375

WORD_CLOUD_WIDTH = 200
WORD_CLOUD_HEIGHT = 300

TOPIC_WORD_CLOUD_KEYWORDS = 20
TOPIC_KEYWORDS = 5


def visualize_analysis(analyzer):
    # Initialize plotter after completion of analysis
    plotter = Plotter(analyzer=analyzer)
    freq_kwds = get_frequent_tokens(analyzer.top_cited_df, query=analyzer.query)
    word_cloud, zoom_out_callback = plotter.papers_word_cloud_and_callback(freq_kwds)
    export_name = re.sub('_{2,}', '_', re.sub('["\':,. ]', '_', f'{analyzer.query}'.lower())).strip('_')
    result = dict(
        topics_analyzed=False,
        n_papers=analyzer.n_papers,
        n_topics=len(analyzer.components),
        export_name=export_name,
        top_cited_papers=[components(plotter.top_cited_papers())],
        most_cited_per_year_papers=[components(plotter.most_cited_per_year_papers())],
        fastest_growth_per_year_papers=[components(plotter.fastest_growth_per_year_papers())],
        papers_stats=[components(plotter.papers_by_year())],
        papers_word_cloud=Plotter.word_cloud_prepare(word_cloud),
        papers_zoom_out_callback=zoom_out_callback,
        keywords_frequencies=[components(plotter.plot_keywords_frequencies(freq_kwds))]
    )

    if analyzer.similarity_graph.nodes():
        result.update(dict(
            topics_analyzed=True,
            components_similarity=[components(plotter.heatmap_topics_similarity())],
            component_size_summary=[components(plotter.topic_years_distribution())],
            topics_info_and_word_cloud_and_callback=[
                (components(p), Plotter.word_cloud_prepare(wc), "true" if is_empty else "false", zoom_in_callback) for
                (p, wc, is_empty, zoom_in_callback) in plotter.topics_info_and_word_cloud_and_callback()],
            component_sizes=plotter.component_sizes(),
            structure_graph=[components(plotter.structure_graph())]
        ))

        topics_hierarchy = plotter.topics_hierarchy_with_keywords()
        if topics_hierarchy:
            result['topics_hierarchy'] = [components(topics_hierarchy)]

    # Configure additional features
    result.update(dict(
        feature_authors_enabled=PUBTRENDS_CONFIG.feature_authors_enabled,
        feature_journals_enabled=PUBTRENDS_CONFIG.feature_journals_enabled,
        feature_numbers_enabled=PUBTRENDS_CONFIG.feature_numbers_enabled,
        feature_evolution_enabled=PUBTRENDS_CONFIG.feature_evolution_enabled,
        feature_review_enabled=PUBTRENDS_CONFIG.feature_review_enabled
    ))

    if PUBTRENDS_CONFIG.feature_authors_enabled:
        result['author_statistics'] = plotter.author_statistics()
        if len(analyzer.authors_similarity_graph.nodes()) > 0:
            result['authors_graph'] = [components(plotter.authors_graph())]

    if PUBTRENDS_CONFIG.feature_journals_enabled:
        result['journal_statistics'] = plotter.journal_statistics()

    if PUBTRENDS_CONFIG.feature_numbers_enabled:
        _, url_prefix = Loaders.get_loader_and_url_prefix(analyzer.source, analyzer.config)
        if analyzer.numbers_df is not None:
            result['numbers'] = [
                (row['id'], url_prefix + row['id'], trim(row['title'], MAX_TITLE_LENGTH), row['numbers'])
                for _, row in analyzer.numbers_df.iterrows()
            ]

    if PUBTRENDS_CONFIG.feature_evolution_enabled:
        evolution_result = plotter.topic_evolution()
        if evolution_result is not None:
            evolution_data, keywords_data = evolution_result
            result['topic_evolution'] = [components(evolution_data)]
            result['topic_evolution_keywords'] = keywords_data

    return result


class Plotter:
    def __init__(self, analyzer=None):
        self.analyzer = analyzer

        if self.analyzer:
            if self.analyzer.similarity_graph.nodes():
                self.comp_colors = Plotter.topics_palette_rgb(self.analyzer.df)
                self.comp_palette = list(self.comp_colors.values())

            n_pub_types = len(self.analyzer.pub_types)
            pub_types_cmap = plt.cm.get_cmap('jet', n_pub_types)
            self.pub_types_colors_map = dict(
                zip(self.analyzer.pub_types, [Plotter.color_to_rgb(pub_types_cmap(i)) for i in range(n_pub_types)])
            )

    @staticmethod
    def paper_callback(ds):
        return CustomJS(args=dict(ds=ds), code="""
            var data = ds.data, selected = ds.selected.indices;

            // Decode params from URL
            const jobid = new URL(window.location).searchParams.get('jobid');
            const source = new URL(window.location).searchParams.get('source');

            // Max number of papers to be opened, others will be ignored
            var MAX_PAPERS = 3;

            for (var i = 0; i < Math.min(MAX_PAPERS, selected.length); i++){
                window.open('/paper?source=' + source + '&id=' + data['id'][selected[i]] + '&jobid=' + jobid, '_blank');
            }
        """)

    @staticmethod
    def author_callback(ds):
        return CustomJS(args=dict(ds=ds), code="""
            var data = ds.data, selected = ds.selected.indices;

            // Decode params from URL
            const jobid = new URL(window.location).searchParams.get('jobid');
            const source = new URL(window.location).searchParams.get('source');
            const query = new URL(window.location).searchParams.get('query');
            const limit = new URL(window.location).searchParams.get('limit');
            const sort = new URL(window.location).searchParams.get('sort');

            // Max number of authors to be opened, others will be ignored
            var MAX_AUTHORS = 3;

            for (var i = 0; i < Math.min(MAX_AUTHORS, selected.length); i++){
                window.open('/papers?&query=' + query + '&source=' + source + '&limit=' + limit + '&sort=' + sort + 
                '&author=' + data['id'][selected[i]] + '&jobid=' + jobid, '_blank');
            }
        """)

    @staticmethod
    def topic_callback(source):
        return CustomJS(args=dict(source=source), code="""
            var data = source.data, selected = source.selected.indices;
            if (selected.length == 1) {
                // only consider case where one glyph is selected by user
                selected_comp = data['comps'][selected[0]];
                window.location.hash = '#topic-' + selected_comp;
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

    def heatmap_topics_similarity(self):
        logger.debug('Visualizing topics similarity with heatmap')

        similarity_df, topics = PlotPreprocessor.topics_similarity_data(
            self.analyzer.similarity_graph, self.analyzer.partition
        )

        step = 30
        cmap = plt.cm.get_cmap('PuBu', step)
        colors = [RGB(*[round(c * 255) for c in cmap(i)[:3]]) for i in range(step)]
        mapper = LinearColorMapper(palette=colors,
                                   low=similarity_df.similarity.min(),
                                   high=similarity_df.similarity.max())

        p = figure(x_range=topics, y_range=topics,
                   x_axis_location="below", plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT,
                   tools=TOOLS, toolbar_location="right",
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

    def topic_years_distribution(self):
        logger.debug('Topics publications year distribution visualization')
        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        plot_components, data = PlotPreprocessor.component_size_summary_data(
            self.analyzer.df, self.analyzer.components, min_year, max_year
        )
        source = ColumnDataSource(data=dict(x=[min_year - 1] + data['years'] + [max_year + 1]))
        plot_titles = []
        words2show = PlotPreprocessor.topics_words(self.analyzer.kwd_df, TOPIC_KEYWORDS, self.analyzer.components)
        for c in self.analyzer.components:
            percent = int(100 * self.analyzer.comp_sizes[int(c)] / len(self.analyzer.df))
            plot_titles.append(f'#{c + 1} [{percent if percent > 0 else "<1"}%] {",".join(words2show[c])}')
        # Fake additional y levels
        p = figure(y_range=list(reversed(plot_titles)) + [' ', '  ', '   '],
                   plot_width=PLOT_WIDTH, plot_height=50 * (len(plot_components) + 2),
                   x_range=(min_year - 1, max_year + 1), toolbar_location=None)
        topics_colors = Plotter.topics_palette_rgb(self.analyzer.df)
        max_papers_per_year = max(max(data[pc]) for pc in plot_components)
        for i, (pc, pt) in enumerate(zip(plot_components, plot_titles)):
            source.add([(pt, 0)] + [(pt, 3 * d / max_papers_per_year) for d in data[pc]] + [(pt, 0)], pt)
            p.patch('x', pt, color=topics_colors[i], alpha=0.6, line_color="black", source=source)
        p.sizing_mode = 'stretch_width'
        p.outline_line_color = None
        p.axis.minor_tick_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.axis_line_color = None
        return p

    def topics_info_and_word_cloud_and_callback(self):
        logger.debug('Per component detailed info visualization')

        # Prepare layouts
        n_comps = len(self.analyzer.components)
        result = []

        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        for comp in range(n_comps):
            df_comp = self.analyzer.df[self.analyzer.df['comp'] == comp]
            if len(df_comp) == 0:
                continue
            ds = ColumnDataSource(PlotPreprocessor.article_view_data_source(
                df_comp, min_year, max_year, True, width=PAPERS_PLOT_WIDTH
            ))
            # Add type coloring
            ds.add([self.pub_types_colors_map[t] for t in df_comp['type']], 'color')
            plot = self.__serve_scatter_article_layout(
                ds=ds, year_range=[min_year, max_year], width=PAPERS_PLOT_WIDTH
            )
            plot.circle(x='year', y='y', fill_alpha=0.5, source=ds, size='size',
                        line_color='color', fill_color='color', legend_field='type')
            plot.legend.location = "top_left"

            # Word cloud description of topic by titles and abstracts
            kwds = get_topic_word_cloud_data(self.analyzer.kwd_df, comp)
            is_empty = len(kwds) == 0
            if is_empty:
                kwds = {'N/A': 1}
            kwds[f'#{comp + 1}'] = 1  # Artificial tag for scale
            color = (self.comp_colors[comp].r, self.comp_colors[comp].g, self.comp_colors[comp].b)
            wc = WordCloud(background_color="white", width=WORD_CLOUD_WIDTH, height=WORD_CLOUD_HEIGHT,
                           color_func=lambda *args, **kwargs: color,
                           max_words=TOPIC_WORD_CLOUD_KEYWORDS, min_font_size=10, max_font_size=30)
            wc.generate_from_frequencies(kwds)

            # Create Zoom In callback
            id_list = list(df_comp['id'])
            zoom_in_callback = self.zoom_callback(id_list, self.analyzer.source,
                                                  zoom=ZOOM_IN,
                                                  query=self.analyzer.query)

            result.append((plot, wc, is_empty, zoom_in_callback))

        return result

    def component_sizes(self):
        assigned_comps = self.analyzer.df[self.analyzer.df['comp'] >= 0]
        d = dict(assigned_comps.groupby('comp')['id'].count())
        return [int(d[k]) for k in range(len(d))]

    def top_cited_papers(self):
        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        ds = ColumnDataSource(PlotPreprocessor.article_view_data_source(
            self.analyzer.top_cited_df, min_year, max_year, False, width=PAPERS_PLOT_WIDTH
        ))
        # Add type coloring
        ds.add([self.pub_types_colors_map[t] for t in self.analyzer.top_cited_df['type']], 'color')

        plot = self.__serve_scatter_article_layout(
            ds=ds, year_range=[min_year, max_year], width=PLOT_WIDTH
        )

        plot.circle(x='year', y='y', fill_alpha=0.5, source=ds, size='size',
                    line_color='color', fill_color='color', legend_field='type')
        plot.legend.location = "top_left"
        return plot

    def most_cited_per_year_papers(self):
        logger.debug('Different colors encode different papers')
        cols = ['year', 'id', 'title', 'authors', 'journal', 'paper_year', 'count']
        most_cited_per_year_df = self.analyzer.max_gain_df[cols].replace(np.nan, "Undefined")
        most_cited_per_year_df['authors'] = most_cited_per_year_df['authors'].apply(
            lambda authors: cut_authors_list(authors)
        )
        ds_max = ColumnDataSource(most_cited_per_year_df)

        factors = self.analyzer.max_gain_df['id'].unique()
        colors = self.factor_colors(factors)

        year_range = [self.analyzer.min_year - 1, self.analyzer.max_year + 1]
        p = figure(tools=TOOLS, toolbar_location="right",
                   plot_width=PLOT_WIDTH, plot_height=SHORT_PLOT_HEIGHT, x_range=year_range,
                   y_axis_type="log" if self.analyzer.max_gain_df['count'].max() > MAX_LINEAR_AXIS else "linear")
        p.sizing_mode = 'stretch_width'
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Number of citations'
        p.yaxis.formatter = NumeralTickFormatter(format='0,0')
        p.hover.tooltips = self._html_tooltips([
            ("Author(s)", '@authors'),
            ("Journal", '@journal'),
            ("Year", '@paper_year'),
            ("Cited by", '@count papers in @year')
        ])
        p.js_on_event('tap', self.paper_callback(ds_max))
        # Use explicit bottom for log scale as workaround
        # https://github.com/bokeh/bokeh/issues/6536
        bottom = self.analyzer.max_gain_df['count'].min() - 0.01 if len(self.analyzer.max_gain_df) else 0.0
        p.vbar(x='year', width=0.8, top='count', bottom=bottom,
               fill_alpha=0.5, source=ds_max, fill_color=colors, line_color=colors)
        return p

    def fastest_growth_per_year_papers(self):
        logger.debug('Fastest growing papers per year')
        logger.debug('Growth(year) = Citation delta (year) / Citations previous year')
        logger.debug('Different colors encode different papers')
        cols = ['year', 'id', 'title', 'authors', 'journal', 'paper_year', 'rel_gain']
        fastest_growth_per_year_df = self.analyzer.max_rel_gain_df[cols].replace(np.nan, "Undefined")
        fastest_growth_per_year_df['authors'] = fastest_growth_per_year_df['authors'].apply(
            lambda authors: cut_authors_list(authors)
        )
        ds_max = ColumnDataSource(fastest_growth_per_year_df)

        factors = self.analyzer.max_rel_gain_df['id'].astype(str).unique()
        colors = self.factor_colors(factors)

        year_range = [self.analyzer.min_year - 1, self.analyzer.max_year + 1]
        p = figure(tools=TOOLS, toolbar_location="right",
                   plot_width=PLOT_WIDTH, plot_height=SHORT_PLOT_HEIGHT, x_range=year_range,
                   y_axis_type="log" if self.analyzer.max_rel_gain_df['rel_gain'].max() > MAX_LINEAR_AXIS else "linear"
                   )
        p.sizing_mode = 'stretch_width'
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Relative Gain of Citations'
        p.yaxis.formatter = NumeralTickFormatter(format='0,0')
        p.hover.tooltips = self._html_tooltips([
            ("Author(s)", '@authors'),
            ("Journal", '@journal'),
            ("Year", '@paper_year'),
            ("Relative Gain", '@rel_gain in @year')])
        p.js_on_event('tap', self.paper_callback(ds_max))
        # Use explicit bottom for log scale as workaround
        # https://github.com/bokeh/bokeh/issues/6536
        bottom = self.analyzer.max_rel_gain_df['rel_gain'].min() - 0.01 if len(self.analyzer.max_rel_gain_df) else 0.0
        p.vbar(x='year', width=0.8, top='rel_gain', bottom=bottom, source=ds_max,
               fill_alpha=0.5, fill_color=colors, line_color=colors)
        return p

    @staticmethod
    def paper_citations_per_year(df, pid):
        d = ColumnDataSource(PlotPreprocessor.article_citation_dynamics_data(df, pid))

        p = figure(tools=TOOLS, toolbar_location="right", plot_width=PLOT_WIDTH,
                   plot_height=SHORT_PLOT_HEIGHT)
        p.vbar(x='x', width=0.8, top='y', source=d, color='#A6CEE3', line_width=3)
        p.sizing_mode = 'stretch_width'
        p.xaxis.axis_label = "Year"
        p.yaxis.axis_label = "Number of citations"
        p.hover.tooltips = [
            ("Year", "@x"),
            ("Cited by", "@y paper(s) in @x"),
        ]

        return p

    def papers_by_year(self):
        ds_stats = ColumnDataSource(PlotPreprocessor.papers_statistics_data(self.analyzer.df))
        year_range = [self.analyzer.min_year - 1, self.analyzer.max_year + 1]
        p = figure(tools=TOOLS, toolbar_location="right",
                   plot_width=PAPERS_PLOT_WIDTH, plot_height=PLOT_HEIGHT,
                   x_range=year_range)
        p.sizing_mode = 'stretch_width'
        p.y_range.start = 0
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Number of papers'
        p.hover.tooltips = [("Papers", '@counts'), ("Year", '@year')]
        if self.analyzer.min_year != self.analyzer.max_year:
            # NOTE: VBar is invisible (alpha=0) to provide tooltips, as in self.component_size_summary()
            p.vbar(x='year', width=0.8, top='counts', fill_alpha=0, line_alpha=0, source=ds_stats)
            # VArea is actually displayed
            ds_stats.data['bottom'] = [0] * len(ds_stats.data['year'])
            p.varea(x='year', y1='bottom', y2='counts', fill_alpha=0.5, source=ds_stats)
        else:
            # NOTE: VBar is invisible (alpha=0) to provide tooltips, as in self.component_size_summary()
            p.vbar(x='year', width=0.8, top='counts', source=ds_stats)
        return p

    def papers_word_cloud_and_callback(self, freq_kwds):
        # Build word cloud, size is proportional to token frequency
        wc = WordCloud(background_color="white", width=WORD_CLOUD_WIDTH, height=WORD_CLOUD_HEIGHT,
                       color_func=lambda *args, **kwargs: 'black',
                       max_words=TOPIC_WORD_CLOUD_KEYWORDS, min_font_size=10, max_font_size=30)
        wc.generate_from_frequencies(freq_kwds)

        # Create Zoom Out callback
        id_list = list(self.analyzer.df['id'])
        zoom_out_callback = self.zoom_callback(id_list, self.analyzer.source,
                                               zoom=ZOOM_OUT, query=self.analyzer.query)

        return wc, zoom_out_callback

    def plot_keywords_frequencies(self, freq_kwds, n=20):
        keywords_df, years = PlotPreprocessor.frequent_keywords_data(
            freq_kwds, self.analyzer.df, self.analyzer.corpus_terms, self.analyzer.corpus_counts, n
        )

        # Define the value dimensions
        max_numbers = keywords_df['number'].max()
        vdim = hv.Dimension('number', range=(-10, max_numbers + 10))

        # Define the dataset
        ds = hv.Dataset(keywords_df, vdims=vdim)
        curves = ds.to(hv.Curve, 'year', groupby='keyword').overlay().redim(
            year=dict(range=(min(years) - 1, max(years) + 5)))

        # Define a function to get the text annotations
        max_year = ds['year'].max()
        label_df = keywords_df[keywords_df.year == max_year].copy().reset_index(drop=True)

        # Update layout for better labels representation
        label_df.sort_values(by='number', inplace=True)
        if len(label_df) > 1:
            label_df['number'] = [i * max_numbers / (len(label_df) - 1) for i in range(len(label_df))]
        label_df.sort_values(by='keyword', inplace=True)
        labels = hv.Labels(label_df, ['year', 'number'], 'keyword')

        overlay = curves * labels

        cmap = Plotter.factors_colormap(len(label_df))
        palette = [Plotter.color_to_rgb(cmap(i)).to_hex() for i in range(len(label_df))]
        overlay.opts(
            opts.Curve(show_frame=False, labelled=[], tools=['hover'],
                       width=PLOT_WIDTH, height=TALL_PLOT_HEIGHT, show_legend=False,
                       xticks=list(reversed(range(max(years), min(years), -5))),
                       color=hv.Cycle(values=palette), alpha=0.8, line_width=2, show_grid=True),
            opts.Labels(text_color='keyword', cmap=palette, text_align='left'),
            opts.NdOverlay(batched=False,
                           gridstyle={'grid_line_dash': [6, 4], 'grid_line_width': 1, 'grid_bounds': (0, 100)})
        )
        p = hv.render(overlay, backend='bokeh')
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Number of papers'
        p.sizing_mode = 'stretch_width'
        return p

    def author_statistics(self):
        authors = self.analyzer.author_stats['author']
        sums = self.analyzer.author_stats['sum']
        if self.analyzer.similarity_graph.nodes():
            topics = self.analyzer.author_stats.apply(
                lambda row: self._to_colored_circle(row['comp'], row['counts'], row['sum']), axis=1)
        else:
            topics = [' '] * len(self.analyzer.author_stats)  # Ignore topics
        return list(zip([trim(a, MAX_AUTHOR_LENGTH) for a in authors], sums, topics))

    def journal_statistics(self):
        journals = self.analyzer.journal_stats['journal']
        sums = self.analyzer.journal_stats['sum']
        if self.analyzer.similarity_graph.nodes():
            topics = self.analyzer.journal_stats.apply(
                lambda row: self._to_colored_circle(row['comp'], row['counts'], row['sum']), axis=1)
        else:
            topics = [' '] * len(self.analyzer.journal_stats)  # Ignore topics
        return list(zip([trim(j, MAX_JOURNAL_LENGTH) for j in journals], sums, topics))

    def _to_colored_circle(self, components, counts, sum, top=3):
        # html code to generate circles corresponding to the most popular topics
        return ' '.join([
            f'<a class="fas fa-circle" style="color:{self.comp_colors[comp]}" href="#topic-{comp + 1}"></a>'
            f'<span class="bk" style="color:black">{int(count / sum * 100)}%</span>'
            for comp, count in zip(components[:top], counts[:top])
        ])

    def __serve_scatter_article_layout(self, ds, year_range, width=PLOT_WIDTH):
        min_year, max_year = year_range
        p = figure(tools=TOOLS, toolbar_location="right",
                   plot_width=width, plot_height=PLOT_HEIGHT,
                   x_range=(min_year - 1, max_year + 1), y_axis_type="log")
        p.sizing_mode = 'stretch_width'
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Number of citations'
        p.yaxis.formatter = NumeralTickFormatter(format='0,0')

        p.hover.tooltips = self._html_tooltips([
            ("Author(s)", '@authors'),
            ("Journal", '@journal'),
            ("Year", '@year'),
            ("Type", '@type'),
            ("Cited by", '@total paper(s) total')])
        p.js_on_event('tap', self.paper_callback(ds))

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

    def dump_citations_graph_cytoscape(self):
        return PlotPreprocessor.dump_citations_graph_cytoscape(self.analyzer.df, self.analyzer.citations_graph)

    def topics_hierarchy_with_keywords(self):
        kwd_df = self.analyzer.kwd_df
        comp_sizes = Counter(self.analyzer.df['comp'])
        logger.debug('Computing dendrogram for clusters')
        if self.analyzer.dendrogram_children is None:
            return None
        clusters_dendrogram_children = compute_clusters_dendrogram_children(self.analyzer.clusters,
                                                                            self.analyzer.dendrogram_children)
        paths, leaves_order = convert_clusters_dendrogram_to_paths(self.analyzer.clusters,
                                                                   clusters_dendrogram_children)

        # Configure dimensions
        p = figure(x_range=[-190, 190],
                   y_range=[-160, 160],
                   tools="save",
                   width=PLOT_WIDTH, height=int(PLOT_WIDTH * 0.8))
        x_coefficient = 1.5  # Ellipse x coefficient
        y_delta = 60  # Extra space near pi / 2 and 3 * pi / 2
        n_topics = len(leaves_order)
        radius = 80  # Radius of circular dendrogram
        dendrogram_len = len(paths[0])
        d_radius = radius / dendrogram_len
        d_degree = 2 * pi / n_topics
        delta = 3  # Space between dendrogram and text
        max_words = min(5, max(1, int(120 / n_topics)))

        # Leaves coordinates
        leaves_degrees = dict((v, i * d_degree) for v, i in leaves_order.items())

        # Draw levels
        for i in range(1, dendrogram_len):
            p.ellipse(0, 0, fill_alpha=0, line_color='lightgray', line_alpha=0.5,
                      width=2 * d_radius * i,
                      height=2 * d_radius * i,
                      line_dash='dotted')

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

        # Draw center
        p.circle(x=0, y=0, size=2, fill_color='gray', line_color='gray')

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

        def contrast_color_rbg(rgb):
            cr, cg, cb = contrast_color(rgb.r, rgb.g, rgb.b)
            return RGB(cr, cg, cb)

        # Topics labels
        p.text(x=[cos(d) * d_radius * (dendrogram_len - 1) for _, d in leaves_degrees.items()],
               y=[sin(d) * d_radius * (dendrogram_len - 1) for _, d in leaves_degrees.items()],
               text=[str(v + 1) for v, _ in leaves_degrees.items()],
               text_align='center', text_baseline='middle', text_font_size='10pt',
               text_color=contrast_color_rbg(topics_colors[v]))

        # Show words for components - most popular words per component
        topics = leaves_order.keys()
        words2show = PlotPreprocessor.topics_words(kwd_df, max_words, topics)

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
                xs.append(cos(wd) * (radius * x_coefficient + delta))
                y = sin(wd) * (radius + delta)
                # Additional vertical space around pi/2 and 3*pi/2
                if pi / 4 <= wd < 3 * pi / 4:
                    y += (pi / 4 - fabs(pi / 2 - wd)) * y_delta
                elif 5 * pi / 4 <= wd < 7 * pi / 4:
                    y -= (pi / 4 - fabs(3 * pi / 2 - wd)) * y_delta
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

    def structure_graph(self):
        g = local_sparse(self.analyzer.similarity_graph, 0.5)
        df = self.analyzer.df
        nodes = df['id']
        comps = df['comp']
        graph = GraphRenderer()
        cmap = Plotter.factors_colormap(len(set(comps)))
        palette = dict(zip(sorted(set(comps)), [Plotter.color_to_rgb(cmap(i)).to_hex()
                                                for i in range(len(set(comps)))]))

        graph.node_renderer.data_source.add(df['id'], 'index')
        graph.node_renderer.data_source.data['id'] = df['id']
        graph.node_renderer.data_source.data['title'] = df['title']
        graph.node_renderer.data_source.data['authors'] = \
            df['authors'].apply(lambda authors: cut_authors_list(authors))
        graph.node_renderer.data_source.data['journal'] = df['journal']
        graph.node_renderer.data_source.data['year'] = df['year']
        graph.node_renderer.data_source.data['cited'] = df['total']
        # Limit size
        graph.node_renderer.data_source.data['size'] = df['total'] * 20 / df['total'].max() + 5
        graph.node_renderer.data_source.data['topic'] = [c + 1 for c in comps]
        graph.node_renderer.data_source.data['color'] = [palette[c] for c in comps]

        graph.edge_renderer.data_source.data = dict(start=[u for u, _ in g.edges],
                                                    end=[v for _, v in g.edges])

        # start of layout code
        x, y = df['x'], df['y']
        xrange = max(x) - min(x)
        yrange = max(y) - min(y)
        p = figure(width=PLOT_WIDTH,
                   height=TALL_PLOT_HEIGHT,
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

        tooltip = """
        <div style="max-width: 500px">
            <div>
                <span style="font-size: 12px; font-weight: bold;">@title</span>
            </div>
            <div>
                <span style="font-size: 11px; font-weight: bold;">Author(s)</span>
                <span style="font-size: 10px;">@authors</span>
            </div>
            <div>
                <span style="font-size: 11px; font-weight: bold;">Journal</span>
                <span style="font-size: 10px;">@journal</span>
            </div>
            <div>
                <span style="font-size: 11px; font-weight: bold;">Year</span>
                <span style="font-size: 10px;">@year</span>
            </div>
            <div>
                <span style="font-size: 11px; font-weight: bold;">Cited</span>
                <span style="font-size: 10px;">@cited</span>
            </div>
            <div>
                <span style="font-size: 11px; font-weight: bold;">Topic</span>
                <span style="font-size: 10px;">@topic</span>
            </div>
        </div>
        """

        p.add_tools(HoverTool(tooltips=tooltip))
        p.js_on_event('tap', self.paper_callback(graph.node_renderer.data_source))

        graph_layout = dict(zip(nodes, zip(x, y)))
        graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

        graph.node_renderer.glyph = Circle(size='size', fill_alpha=0.7, line_alpha=0.7, fill_color='color')
        graph.node_renderer.hover_glyph = Circle(size='size', fill_alpha=1.0, line_alpha=1.0, fill_color='color')

        graph.edge_renderer.glyph = MultiLine(line_color='grey', line_alpha=0.1, line_width=1)
        graph.edge_renderer.hover_glyph = MultiLine(line_color='grey', line_alpha=1.0, line_width=2)

        graph.inspection_policy = NodesAndLinkedEdges()

        p.renderers.append(graph)
        return p

    def authors_graph(self, authors_per_group=20):
        authors_df = self.analyzer.authors_df
        if authors_df is None:
            return None
        authors_similarity_graph = self.analyzer.authors_similarity_graph
        logger.debug(f'Collecting top productive authors only ({authors_per_group}) per group')
        top_authors = set([])
        for group in set(authors_df['cluster']):
            group_authors = authors_df.loc[authors_df['cluster'] == group]
            top = group_authors.sort_values(by=['productivity'], ascending=False).head(authors_per_group)['author']
            logger.debug(f'#{group} ({len(group_authors)}) {", ".join(top)}, ...')
            top_authors.update(top)
        top_authors_df = authors_df.loc[authors_df['author'].isin(top_authors)]

        logger.debug(f'Filtering authors graph authors_per_group={authors_per_group}')
        top_authors_graph = nx.Graph()
        for (a1, a2, d) in authors_similarity_graph.edges(data=True):
            if a1 in top_authors and a2 in top_authors:
                top_authors_graph.add_edge(a1, a2, **d)
        logger.debug(f'Built filtered top authors graph - '
                     f'{len(top_authors_graph.nodes())} nodes and {len(top_authors_graph.edges())} edges')

        graph = GraphRenderer()
        clusters = top_authors_df['cluster']
        cmap = Plotter.factors_colormap(len(set(clusters)))
        palette = dict(zip(set(clusters), [Plotter.color_to_rgb(cmap(i)).to_hex() for i in range(len(set(clusters)))]))

        authors = top_authors_df['author']
        graph.node_renderer.data_source.add(authors, 'index')
        graph.node_renderer.data_source.data['id'] = authors
        graph.node_renderer.data_source.data['cited'] = top_authors_df['cited']
        graph.node_renderer.data_source.data['papers'] = top_authors_df['papers']
        graph.node_renderer.data_source.data['size'] = \
            top_authors_df['productivity'] * 20 / top_authors_df['productivity'].max() + 5
        graph.node_renderer.data_source.data['cluster'] = clusters
        graph.node_renderer.data_source.data['color'] = [palette[c] for c in clusters]
        graph.edge_renderer.data_source.data = dict(start=[a for a, _ in top_authors_graph.edges],
                                                    end=[a for _, a in top_authors_graph.edges])

        # start of layout code
        x, y = top_authors_df['x'], top_authors_df['y']
        xrange = max(x) - min(x)
        yrange = max(y) - min(y)
        p = figure(title="",
                   width=PLOT_WIDTH,
                   height=TALL_PLOT_HEIGHT,
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

        tooltip = """
        <div style="max-width: 500px">
            <div>
                <span style="font-size: 12px; font-weight: bold;">@id</span>
            </div>
            <div>
                <span style="font-size: 11px; font-weight: bold;">Papers</span>
                <span style="font-size: 10px;">@papers</span>
            </div>
            <div>
                <span style="font-size: 11px; font-weight: bold;">Cited</span>
                <span style="font-size: 10px;">@cited</span>
            </div>
            <div>
                <span style="font-size: 11px; font-weight: bold;">Cluster</span>
                <span style="font-size: 10px;">@cluster</span>
            </div>
        </div>
        """

        p.add_tools(HoverTool(tooltips=tooltip))
        p.js_on_event('tap', self.author_callback(graph.node_renderer.data_source))

        graph_layout = dict(zip(authors, zip(x, y)))
        graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

        graph.node_renderer.glyph = Circle(size='size', fill_alpha=0.7, line_alpha=0.7, fill_color='color')
        graph.node_renderer.hover_glyph = Circle(size='size', fill_alpha=1.0, line_alpha=1.0, fill_color='color')

        graph.edge_renderer.glyph = MultiLine(line_color='grey', line_alpha=0.1, line_width=1)
        graph.edge_renderer.hover_glyph = MultiLine(line_color='grey', line_alpha=1.0, line_width=2)

        graph.inspection_policy = NodesAndLinkedEdges()
        p.renderers.append(graph)
        return p

    def topic_evolution(self):
        """
        Sankey diagram of topic evolution
        :return:
            if self.analyzer.evolution_df is None: None, as no evolution can be observed in 1 step
            Sankey diagram + table with keywords
        """
        # Topic evolution analysis failed, one step is not enough to analyze evolution
        if self.analyzer.evolution_df is None or not self.analyzer.evolution_kwds:
            logger.debug(f'Topic evolution failure, '
                         f'evolution_df is None: {self.analyzer.evolution_df is None}, '
                         f'evolution_kwds is None: {self.analyzer.evolution_kwds is None}')
            return None

        n_steps = len(self.analyzer.evolution_df.columns) - 2

        edges, nodes_data = PlotPreprocessor.topic_evolution_data(
            self.analyzer.evolution_df, self.analyzer.evolution_kwds, n_steps
        )

        value_dim = hv.Dimension('Papers', unit=None)
        nodes_ds = hv.Dataset(nodes_data, 'index', 'label')
        topic_evolution = hv.Sankey((edges, nodes_ds), ['From', 'To'], vdims=value_dim)
        topic_evolution.opts(labels='label',
                             width=PLOT_WIDTH, height=max(TALL_PLOT_HEIGHT, len(self.analyzer.components) * 30),
                             show_values=False, cmap='tab20',
                             edge_color=dim('To').str(), node_color=dim('index').str())

        p = hv.render(topic_evolution, backend='bokeh')
        p.sizing_mode = 'stretch_width'
        kwds_data = PlotPreprocessor.topic_evolution_keywords_data(
            self.analyzer.evolution_kwds
        )
        return p, kwds_data

    @staticmethod
    def word_cloud_prepare(wc):
        return json.dumps([(word, int(position[0]), int(position[1]),
                            int(font_size), orientation is not None,
                            rgb2hex(color))
                           for (word, count), font_size, position, orientation, color in wc.layout_])

    @staticmethod
    def color_to_rgb(v):
        return RGB(*[int(c * 255) for c in v[:3]])

    @staticmethod
    def factor_colors(factors):
        cmap = Plotter.factors_colormap(len(factors))
        palette = [Plotter.color_to_rgb(cmap(i)).to_hex() for i in range(len(factors))]
        colors = factor_cmap('id', palette=palette, factors=factors)
        return colors

    @staticmethod
    def topics_palette_rgb(df):
        n_comps = len(set(df['comp']))
        cmap = Plotter.factors_colormap(n_comps)
        return dict((i, Plotter.color_to_rgb(cmap(i))) for i in range(n_comps))

    @staticmethod
    def factors_colormap(n):
        if n <= 10:
            return plt.cm.get_cmap('tab10', n)
        if n <= 20:
            return plt.cm.get_cmap('tab20', n)
        else:
            return plt.cm.get_cmap('nipy_spectral', n)

    @staticmethod
    def topics_palette(df):
        return dict((k, v.to_hex()) for k, v in Plotter.topics_palette_rgb(df).items())
