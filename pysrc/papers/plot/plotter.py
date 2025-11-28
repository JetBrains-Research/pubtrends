import logging
from collections import Counter
from math import fabs, log, sin, cos, pi
from string import Template

import holoviews as hv
import numpy as np
from bokeh.embed import components
from bokeh.models import ColumnDataSource, CustomJS, LabelSet
from bokeh.models import GraphRenderer, StaticLayoutProvider, Circle, HoverTool, MultiLine, Label
from bokeh.models import NumeralTickFormatter
from bokeh.models.graphs import NodesAndLinkedEdges
from bokeh.plotting import figure
from holoviews import opts
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from wordcloud import WordCloud

from pysrc.config import PAPERS_PLOT_WIDTH, WORD_CLOUD_WIDTH, WORD_CLOUD_KEYWORDS, WORD_CLOUD_HEIGHT, \
    PLOT_WIDTH, SHORT_PLOT_HEIGHT, MAX_LINEAR_AXIS, PLOT_HEIGHT, MAX_JOURNAL_LENGTH, MAX_AUTHOR_LENGTH, \
    TALL_PLOT_HEIGHT, VISUALIZATION_GRAPH_EDGES
from pysrc.papers.analysis.graph import sparse_graph
from pysrc.papers.analysis.topics import get_topics_description
from pysrc.papers.plot.geometry import compute_component_boundaries, rescaled_comp_corrds, \
    shapely_to_bokeh_multipolygons
from pysrc.papers.plot.plot_preprocessor import PlotPreprocessor
from pysrc.papers.utils import cut_authors_list, contrast_color, \
    topics_palette_rgb, color_to_rgb, factor_colors, factors_colormap, trim

TOOLS = "hover,pan,tap,wheel_zoom,box_zoom,reset,save"
hv.extension('bokeh')

logger = logging.getLogger(__name__)


def exception_handler(func):
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.info(f'{func.__name__} raised {e}')
            return []

    return inner_function


@exception_handler
def components_list(plot):
    return [components(plot)]


@exception_handler
def topics_info_and_word_cloud(plotter):
    return [
        (components(p), PlotPreprocessor.word_cloud_prepare(wc))
        for (p, wc) in plotter.plot_topics_info_and_word_cloud()
    ]


class Plotter:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.pub_types = list(set(self.data.df['type']))
        self.topics_description = get_topics_description(
            self.data.df,
            self.data.corpus, self.data.corpus_tokens, self.data.corpus_counts,
            n_words=max(self.config.topic_description_words, WORD_CLOUD_KEYWORDS),
        )

        if self.data:
            if self.data.papers_graph.nodes():
                self.comp_colors = topics_palette_rgb(self.data.df)

            n_pub_types = len(self.pub_types)
            pub_types_cmap = plt.cm.get_cmap('jet', n_pub_types)
            self.pub_types_colors_map = dict(
                zip(self.pub_types, [color_to_rgb(pub_types_cmap(i)) for i in range(n_pub_types)])
            )

    def plot_topic_years_distribution(self):
        logger.debug('Processing topic_years_distribution')
        logger.debug('Topics publications year distribution visualization')
        min_year, max_year = self.data.df['year'].min(), self.data.df['year'].max()
        plot_components, data = PlotPreprocessor.component_size_summary_data(
            self.data.df, set(self.data.df['comp']), min_year, max_year
        )
        kwd_df = PlotPreprocessor.compute_kwds(self.topics_description, self.config.topic_description_words)
        return self._plot_topics_years_distribution(
            self.data.df, kwd_df, plot_components, data, self.config.topic_description_words, min_year, max_year
        )

    def plot_topics_info_and_word_cloud(self):
        logger.debug('Processing topics_info_and_word_cloud')

        # Prepare layouts
        result = []

        min_year, max_year = self.data.df['year'].min(), self.data.df['year'].max()
        for comp in sorted(set(self.data.df['comp'])):
            df_comp = self.data.df[self.data.df['comp'] == comp]
            ds = ColumnDataSource(PlotPreprocessor.article_view_data_source(
                df_comp, min_year, max_year, True, width=PAPERS_PLOT_WIDTH
            ))
            # Add type coloring
            ds.add([self.pub_types_colors_map[t] for t in df_comp['type']], 'color')
            plot = Plotter._plot_scatter_papers_layout(
                self.data.source, ds, [min_year, max_year], PAPERS_PLOT_WIDTH
            )
            plot.scatter(x='year', y='y', fill_alpha=0.5, source=ds, size='size',
                         line_color='color', fill_color='color', legend_field='type')
            plot.legend.location = "top_left"

            # Word cloud description of topic by titles and abstracts
            kwds = PlotPreprocessor.get_topic_word_cloud_data(
                self.topics_description, comp, WORD_CLOUD_KEYWORDS
            )
            kwds[f'#{comp + 1}'] = 1  # Artificial tag for scale
            color = (self.comp_colors[comp].r, self.comp_colors[comp].g, self.comp_colors[comp].b)
            wc = WordCloud(background_color="white", width=WORD_CLOUD_WIDTH, height=WORD_CLOUD_HEIGHT,
                           color_func=lambda *args, **kwargs: color,
                           max_words=WORD_CLOUD_KEYWORDS, min_font_size=10, max_font_size=30)
            wc.generate_from_frequencies(kwds)

            result.append((plot, wc))

        return result

    def plot_top_cited_papers(self):
        logger.debug('Processing top_cited_papers')
        min_year, max_year = self.data.df['year'].min(), self.data.df['year'].max()
        ds = ColumnDataSource(PlotPreprocessor.article_view_data_source(
            self.data.top_cited_df, min_year, max_year, False, width=PAPERS_PLOT_WIDTH
        ))
        # Add type coloring
        ds.add([self.pub_types_colors_map[t] for t in self.data.top_cited_df['type']], 'color')

        plot = Plotter._plot_scatter_papers_layout(
            self.data.source, ds, [min_year, max_year], PLOT_WIDTH
        )

        plot.scatter(x='year', y='y', fill_alpha=0.5, source=ds, size='size',
                     line_color='color', fill_color='color', legend_field='type')
        plot.legend.location = "top_left"
        Plotter.remove_wheel_zoom_tool(plot)
        return plot

    def plot_most_cited_per_year_papers(self):
        logger.debug('Processing most_cited_per_year_papers')
        cols = ['year', 'id', 'title', 'authors', 'journal', 'paper_year', 'count']
        most_cited_per_year_df = self.data.max_gain_df[cols].replace(np.nan, "Undefined")
        most_cited_per_year_df['authors'] = most_cited_per_year_df['authors'].apply(
            lambda authors: cut_authors_list(authors)
        )
        factors = self.data.max_gain_df['id'].unique()
        colors = factor_colors(factors)

        most_cited_counts = most_cited_per_year_df['count']
        min_year, max_year = self.data.df['year'].min(), self.data.df['year'].max()
        p = figure(tools=TOOLS, toolbar_location="right",
                   width=PLOT_WIDTH, height=SHORT_PLOT_HEIGHT,
                   x_range=(min_year - 1, max_year + 1),
                   y_axis_type="log" if most_cited_counts.max() > MAX_LINEAR_AXIS else "linear")
        p.sizing_mode = 'stretch_width'
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Number of citations'
        p.yaxis.formatter = NumeralTickFormatter(format='0,0')
        p.hover.tooltips = self._paper_html_tooltips(self.data.source, [
            ("Author(s)", '@authors'),
            ("Journal", '@journal'),
            ("Year", '@paper_year'),
            ("Cited by", '@count papers in @year')
        ])
        most_cited_per_year_df['count'] = most_cited_per_year_df['count'].astype(float)
        ds = ColumnDataSource(most_cited_per_year_df)
        p.js_on_event('tap', self._paper_callback(ds))
        # Use explicit bottom for log scale as workaround
        # https://github.com/bokeh/bokeh/issues/6536
        bottom = most_cited_counts.min() - 0.01
        p.vbar(x='year', width=0.8, top='count', bottom=bottom,
               fill_alpha=0.5, source=ds, fill_color=colors, line_color=colors)
        Plotter.remove_wheel_zoom_tool(p)
        return p

    def plot_fastest_growth_per_year_papers(self):
        logger.debug('Processing fastest_growth_per_year_papers')
        logger.debug('Growth(year) = Citation delta (year) / Citations previous year')
        logger.debug('Different colors encode different papers')
        cols = ['year', 'id', 'title', 'authors', 'journal', 'paper_year', 'rel_gain']
        fastest_growth_per_year_df = self.data.max_rel_gain_df[cols].replace(np.nan, "Undefined")
        fastest_growth_per_year_df['authors'] = fastest_growth_per_year_df['authors'].apply(
            lambda authors: cut_authors_list(authors)
        )

        factors = self.data.max_rel_gain_df['id'].astype(str).unique()
        colors = factor_colors(factors)

        fastest_rel_gains = fastest_growth_per_year_df['rel_gain']
        min_year, max_year = self.data.df['year'].min(), self.data.df['year'].max()
        p = figure(tools=TOOLS, toolbar_location="right",
                   width=PLOT_WIDTH, height=SHORT_PLOT_HEIGHT,
                   x_range=(min_year - 1, max_year + 1),
                   y_axis_type="log" if fastest_rel_gains.max() > MAX_LINEAR_AXIS else "linear")
        p.sizing_mode = 'stretch_width'
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Relative Gain of Citations'
        p.yaxis.formatter = NumeralTickFormatter(format='0,0')
        p.hover.tooltips = self._paper_html_tooltips(self.data.source, [
            ("Author(s)", '@authors'),
            ("Journal", '@journal'),
            ("Year", '@paper_year'),
            ("Relative Gain", '@rel_gain in @year')])
        ds = ColumnDataSource(fastest_growth_per_year_df)
        p.js_on_event('tap', self._paper_callback(ds))
        # Use explicit bottom for log scale as workaround
        # https://github.com/bokeh/bokeh/issues/6536
        bottom = fastest_rel_gains.min() - 0.01
        p.vbar(x='year', width=0.8, top='rel_gain', bottom=bottom, source=ds,
               fill_alpha=0.5, fill_color=colors, line_color=colors)
        Plotter.remove_wheel_zoom_tool(p)
        return p

    def plot_papers_by_year(self):
        logger.debug('Processing papers_by_year')
        ds_stats = ColumnDataSource(PlotPreprocessor.papers_statistics_data(self.data.df))
        min_year, max_year = self.data.df['year'].min(), self.data.df['year'].max()
        p = figure(tools=TOOLS, toolbar_location="right",
                   width=PAPERS_PLOT_WIDTH, height=PLOT_HEIGHT,
                   x_range=(min_year - 1, max_year + 1))
        p.sizing_mode = 'stretch_width'
        p.y_range.start = 0
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Number of papers'
        p.hover.tooltips = [("Papers", '@counts'), ("Year", '@year')]
        if min_year != max_year:
            ds_stats.data['bottom'] = [0] * len(ds_stats.data['year'])
            p.varea(x='year', y1='bottom', y2='counts', fill_alpha=0.5, source=ds_stats)
        Plotter.remove_wheel_zoom_tool(p)
        return p

    def plot_keywords_frequencies(self, freq_kwds, n=20):
        logger.debug('Processing plot_keywords_frequencies')
        keywords_df, years = PlotPreprocessor.frequent_keywords_data(
            freq_kwds, self.data.df, self.data.corpus_tokens, self.data.corpus_counts, n
        )
        if len(years) <= 3:
            return None
        return self._plot_keywords_timeline(keywords_df, years)

    def author_statistics(self):
        logger.debug('Processing author_statistics')
        authors = self.data.author_stats['author']
        sums = self.data.author_stats['sum']
        if self.data.papers_graph.nodes():
            topics = self.data.author_stats.apply(
                lambda row: self._to_colored_square(row['comp'], row['counts'], row['sum']), axis=1)
        else:
            topics = [' '] * len(self.data.author_stats)  # Ignore topics
        return list(zip([trim(a, MAX_AUTHOR_LENGTH) for a in authors], sums, topics))

    def journal_statistics(self):
        logger.debug('Processing journal_statistics')
        journals = self.data.journal_stats['journal']
        sums = self.data.journal_stats['sum']
        if self.data.papers_graph.nodes():
            topics = self.data.journal_stats.apply(
                lambda row: self._to_colored_square(row['comp'], row['counts'], row['sum']), axis=1)
        else:
            topics = [' '] * len(self.data.journal_stats)  # Ignore topics
        return list(zip([trim(j, MAX_JOURNAL_LENGTH) for j in journals], sums, topics))

    def topics_hierarchy_with_keywords(self):
        if self.data.dendrogram is None:
            return None
        kwd_df = PlotPreprocessor.compute_kwds(self.topics_description, self.config.topic_description_words)
        return Plotter._plot_topics_hierarchy_with_keywords(
            self.data.df, kwd_df, self.data.df['comp'], self.data.dendrogram
        )

    def plot_papers_graph(self, interactive=True):
        logger.debug('Prepare sparse graph to visualize with reduced number of edges')
        visualize_graph = sparse_graph(self.data.papers_graph, VISUALIZATION_GRAPH_EDGES)
        return Plotter._plot_papers_graph(
            self.data.search_ids, self.data.source, visualize_graph, self.data.df,
            self.config.topic_description_words, topics_tags=self.topics_description,
            interactive=interactive
        )

    def _to_colored_square(self, components, counts, sum, top=3):
        # html code to generate circles corresponding to the most popular topics
        return ' '.join([
            f'<a class="fas fa-square" style="color:{self.comp_colors[comp]}" href="#topic-{comp + 1}"></a>'
            f'<span class="bk" style="color:black">{int(count / sum * 100)}%</span>'
            for comp, count in zip(components[:top], counts[:top])
        ])

    @staticmethod
    def _plot_topics_years_distribution(df, kwd_df, plot_components, data, topic_keywords, min_year, max_year):
        source = ColumnDataSource(data=dict(x=[min_year - 1] + data['years'] + [max_year + 1]))
        plot_titles = []
        words2show = PlotPreprocessor.topics_words(kwd_df, topic_keywords)
        comp_sizes = Counter(df['comp'])
        for c in sorted(set(df['comp'])):
            percent = int(100 * comp_sizes[int(c)] / len(df))
            plot_titles.append(f'#{c + 1} [{percent if percent > 0 else "<1"}%] {",".join(words2show[c])}')
        # Fake additional y levels
        p = figure(y_range=list(reversed(plot_titles)) + [' ', '  ', '   '],
                   width=PLOT_WIDTH, height=50 * (len(plot_components) + 2),
                   x_range=(min_year - 1, max_year + 1), toolbar_location=None)
        topics_colors = topics_palette_rgb(df)
        max_papers_per_year = max(max(data[pc]) for pc in plot_components)
        for i, (pc, pt) in enumerate(zip(plot_components, plot_titles)):
            source.add([(pt, 0)] + [(pt, 3 * d / max_papers_per_year) for d in data[pc]] + [(pt, 0)], pt)
            p.patch('x', pt, color=topics_colors[i], alpha=0.6, line_color="black", source=source)
        p.sizing_mode = 'stretch_width'
        p.outline_line_color = None
        p.axis.minor_tick_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.axis_line_color = None
        Plotter.remove_wheel_zoom_tool(p)
        return p

    @staticmethod
    def _plot_paper_citations_per_year(df, pid):
        logger.debug('Processing paper_citations_per_year')
        ds = ColumnDataSource(PlotPreprocessor.article_citation_dynamics_data(df, pid))
        p = figure(tools=TOOLS, toolbar_location="right", width=PLOT_WIDTH,
                   height=SHORT_PLOT_HEIGHT)
        p.vbar(x='x', width=0.8, top='y', source=ds, color='#A6CEE3', line_width=3)
        p.sizing_mode = 'stretch_width'
        p.xaxis.axis_label = "Year"
        p.yaxis.axis_label = "Number of citations"
        p.hover.tooltips = [
            ("Year", "@x"),
            ("Cited by", "@y paper(s) in @x"),
        ]
        Plotter.remove_wheel_zoom_tool(p)
        return p

    @staticmethod
    def _papers_word_cloud(freq_kwds):
        # Build word cloud, size is proportional to token frequency
        wc = WordCloud(background_color="white", width=WORD_CLOUD_WIDTH, height=WORD_CLOUD_HEIGHT,
                       color_func=lambda *args, **kwargs: 'black',
                       max_words=WORD_CLOUD_KEYWORDS, min_font_size=10, max_font_size=30)
        wc.generate_from_frequencies(freq_kwds)
        return wc

    @staticmethod
    def _plot_keywords_timeline(keywords_df, years):
        logger.debug('Processing plot_keywords_timeline')
        # Define the value dimensions
        max_numbers = keywords_df['number'].max()
        vdim = hv.Dimension('number', range=(-2, max_numbers + 2))
        # Define the dataset
        ds = hv.Dataset(keywords_df, vdims=vdim)
        # Dynamically compute extra years to the right based on label length and plot width
        max_year = ds['year'].max()
        min_year = ds['year'].min()
        # longest keyword length in characters
        max_label_len = keywords_df['keyword'].map(lambda x: len(str(x))).max()
        # rough pixel width per character for the current font
        label_px = max_label_len * 9
        # pixels to years conversion factor
        years_per_px = (max_year - min_year + 5) / float(PLOT_WIDTH)
        # compute extra years needed to fit the longest label inside initial viewport
        extra_years = int(label_px * years_per_px) + 2
        # keep within reasonable bounds
        extra_years = max(5, min(30, extra_years))
        curves = ds.to(hv.Curve, 'year', groupby='keyword').overlay().redim(
            year=dict(range=(min_year - 1, max_year + extra_years)))
        # Define a function to get the text annotations
        label_df = keywords_df[keywords_df['year'] == max_year].copy().reset_index(drop=True)
        label_df['year'] += 1
        # Update layout for better labels representation
        label_df.sort_values(by='number', inplace=True)
        if len(label_df) > 1:
            label_df['number'] = [i * max_numbers / (len(label_df) - 1) for i in range(len(label_df))]
        label_df.sort_values(by='keyword', inplace=True)
        labels = hv.Labels(label_df, ['year', 'number'], 'keyword')
        overlay = curves * labels
        cmap = factors_colormap(len(label_df))
        palette = [color_to_rgb(cmap(i)).to_hex() for i in range(len(label_df))]
        overlay.opts(
            opts.Curve(show_frame=False, labelled=[], tools=['hover'],
                       width=PLOT_WIDTH, height=len(label_df) * 25, show_legend=False,
                       xticks=list(reversed(range(max_year, min_year, -5))),
                       color=hv.Cycle(values=palette), alpha=0.8, line_width=2, show_grid=True),
            opts.Labels(text_color='keyword', cmap=palette, text_align='left'),
            opts.NdOverlay(batched=False,
                           gridstyle={'grid_line_dash': [6, 4], 'grid_line_width': 1, 'grid_bounds': (0, 100)})
        )
        p = hv.render(overlay, backend='bokeh')
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Number of papers'
        p.sizing_mode = 'stretch_width'
        Plotter.remove_wheel_zoom_tool(p)
        return p

    @staticmethod
    def _plot_topics_hierarchy_with_keywords(df, kwd_df, clusters, dendrogram_children,
                                             width=PLOT_WIDTH, height=int(PLOT_WIDTH * 3 / 4)):
        comp_sizes = Counter(df['comp'])
        max_words = 3 if len(comp_sizes) >= 20 else 5
        logger.debug('Computing dendrogram for clusters')
        if dendrogram_children is None:
            return None
        clusters_dendrogram = PlotPreprocessor.compute_clusters_dendrogram_children(clusters, dendrogram_children)
        paths, leaves_order = PlotPreprocessor.convert_clusters_dendrogram_to_paths(clusters, clusters_dendrogram)

        # Configure dimensions, keep range ratios to keep circles round
        mx = 180
        # Hacky coefficients to make circular dendrogram look good
        my = int(1.05 * mx * height / width)
        p = figure(x_range=(-mx, mx),
                   y_range=(-my, my),
                   tools="save",
                   width=width, height=height)
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
                d = ds[path[i - 1]]
                if path[i] not in next_ds:
                    next_ds[path[i]] = []
                next_ds[path[i]].append(d)

                # Draw current level connections
                p.line([cos(d) * d_radius * (dendrogram_len - i), cos(d) * d_radius * (dendrogram_len - i - 1)],
                       [sin(d) * d_radius * (dendrogram_len - i), sin(d) * d_radius * (dendrogram_len - i - 1)],
                       line_color='grey')

            # Compute next connections and draw arcs
            for v, nds in next_ds.items():
                next_ds[v] = np.mean(nds)
                if nds[0] != nds[-1]:
                    # Draw elliptical arc that accounts for non-1:1 aspect ratio.
                    # Approximate the arc with a polyline for robustness across Bokeh versions.
                    r = d_radius * (dendrogram_len - i - 1)
                    start = nds[0]
                    end = nds[-1]
                    Plotter.arc(p, start, end, r)

            ds = next_ds

        # Draw leaves
        n_comps = len(comp_sizes)
        cmap = factors_colormap(n_comps)
        topics_colors = dict((i, color_to_rgb(cmap(i))) for i in range(n_comps))
        xs = [cos(d) * d_radius * (dendrogram_len - 1) for _, d in leaves_degrees.items()]
        ys = [sin(d) * d_radius * (dendrogram_len - 1) for _, d in leaves_degrees.items()]
        # noinspection PyTypeChecker
        sizes = [8 + int(min(5, log(comp_sizes[v]))) for v, _ in leaves_degrees.items()]
        comps = [v + 1 for v, _ in leaves_degrees.items()]
        colors = [topics_colors[v] for v, _ in leaves_degrees.items()]
        ds = ColumnDataSource(data=dict(x=xs, y=ys, size=sizes, comps=comps, color=colors))
        p.rect(x='x', y='y', width='size', height='size', fill_color='color', line_color='black', source=ds)

        # Topics labels
        p.text(x=[cos(d) * d_radius * (dendrogram_len - 1) for _, d in leaves_degrees.items()],
               y=[sin(d) * d_radius * (dendrogram_len - 1) for _, d in leaves_degrees.items()],
               text=[str(v + 1) for v, _ in leaves_degrees.items()],
               text_align='center', text_baseline='middle', text_font_size='10pt',
               text_color=[contrast_color(topics_colors[v]) for v, _ in leaves_degrees.items()])

        # Show words for components - most popular words per component
        words2show = PlotPreprocessor.topics_words(kwd_df, max_words)

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

        # Make the plot responsive to container width while preserving aspect ratio
        p.sizing_mode = 'stretch_width'
        # Keep circles round on resize by matching x/y scale
        p.match_aspect = True
        p.axis.major_tick_line_color = None
        p.axis.minor_tick_line_color = None
        p.axis.major_label_text_color = None
        p.axis.major_label_text_font_size = '0pt'
        p.axis.axis_line_color = None
        p.grid.grid_line_color = None
        p.outline_line_color = None
        Plotter.remove_wheel_zoom_tool(p)
        return p

    @staticmethod
    def arc(p: figure, start, end, r: float, nseg=32):
        # handle wrap-around direction (ensure increasing parameter)
        if end < start:
            end += 2 * pi
        # Sample points along the arc on a circle in data space;
        # the plot's non-1.0 aspect will naturally render this as an ellipse on screen.
        ts = np.linspace(start, end, nseg)
        xs_arc = [r * cos(t) for t in ts]
        ys_arc = [r * sin(t) for t in ts]
        p.line(xs_arc, ys_arc, line_color='grey')

    @staticmethod
    def _plot_scatter_papers_layout(source, ds, year_range, width=PLOT_WIDTH):
        min_year, max_year = year_range
        p = figure(tools=TOOLS, toolbar_location="right",
                   width=width, height=PLOT_HEIGHT,
                   x_range=(min_year - 1, max_year + 1), y_axis_type="log")
        p.sizing_mode = 'stretch_width'
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Number of citations'
        p.yaxis.formatter = NumeralTickFormatter(format='0,0')

        p.hover.tooltips = Plotter._paper_html_tooltips(source, [
            ("Author(s)", '@authors'),
            ("Journal", '@journal'),
            ("Year", '@year'),
            ("Type", '@type'),
            ("Cited by", '@total paper(s) total')])
        p.js_on_event('tap', Plotter._paper_callback(ds))
        Plotter.remove_wheel_zoom_tool(p)
        return p

    @staticmethod
    def _paper_html_tooltips(source, tips_list, idname='id'):
        if source == "pubmed":
            tips_list.insert(0, ("PMID", f'@{idname}'))
        else:
            tips_list.insert(0, ("ID", f'@{idname}'))
        tips_list = tips_list

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

    @staticmethod
    def _plot_papers_graph(
            search_ids, source, gs, df, topic_words,
            shown_pid=None, topics_tags=None, topics_meshs=None, add_callback=True,
            width=PLOT_WIDTH, height=TALL_PLOT_HEIGHT,
            interactive=True):
        logger.debug('Processing plot_papers_graph')
        if search_ids is not None:
            search_ids = [str(p) for p in search_ids]
        if shown_pid is not None:
            shown_pid = str(shown_pid)
        pids = df['id']
        pim = dict((p, i) for i, p in enumerate(pids))
        comps = df['comp']
        connections = [len(list(gs.neighbors(p))) + 1 for p in pids]
        graph = GraphRenderer()
        cmap = factors_colormap(len(set(comps)))
        palette = dict(zip(sorted(set(comps)), [color_to_rgb(cmap(i)).to_hex() for i in range(len(set(comps)))]))

        pids_int = [pim[p] for p in pids]
        # Graph API requires id to be integer
        graph.node_renderer.data_source.add(pids_int, 'index')
        graph.node_renderer.data_source.data['source'] = [source] * len(pids)
        graph.node_renderer.data_source.data['id'] = pids_int
        graph.node_renderer.data_source.data['pid'] = pids
        graph.node_renderer.data_source.data['title'] = df['title']
        graph.node_renderer.data_source.data['authors'] = \
            df['authors'].apply(lambda authors: cut_authors_list(authors))
        graph.node_renderer.data_source.data['journal'] = df['journal']
        graph.node_renderer.data_source.data['year'] = df['year']
        graph.node_renderer.data_source.data['total'] = df['total']
        graph.node_renderer.data_source.data['type'] = df['type']
        graph.node_renderer.data_source.data['mesh'] = df['mesh']
        graph.node_renderer.data_source.data['keywords'] = df['keywords']
        graph.node_renderer.data_source.data['topic'] = [c + 1 for c in comps]
        graph.node_renderer.data_source.data['connections'] = connections
        if topics_tags is not None:
            graph.node_renderer.data_source.data['topic_tags'] = \
                [','.join(t for t, _ in topics_tags[c][:topic_words]) for c in comps]
        if topics_meshs is not None:
            graph.node_renderer.data_source.data['topic_meshs'] = \
                [','.join(t for t, _ in topics_meshs[c][:topic_words]) for c in comps]

        # Aesthetics
        graph.node_renderer.data_source.data['radius'] = \
            minmax_scale([c / max(connections) * 0.5 for c in connections]) + 0.3
        graph.node_renderer.data_source.data['color'] = [palette[c] for c in comps]
        # Show search / shown ids
        highlight_search_ids = search_ids is not None and len(search_ids) < len(df)
        highlight_shown_pid = shown_pid is not None
        if highlight_search_ids or shown_pid is not None:
            graph.node_renderer.data_source.data['line_width'] = \
                [5.0 if highlight_search_ids and str(p) in search_ids else
                 3.0 if highlight_shown_pid and str(p) == shown_pid else
                 1 for p in pids]
            graph.node_renderer.data_source.data['alpha'] = \
                [1.0 if highlight_search_ids and str(p) in search_ids or
                        highlight_shown_pid and str(p) == shown_pid else
                 0.5 for p in pids]
        else:
            graph.node_renderer.data_source.data['line_width'] = [1] * len(pids)
            graph.node_renderer.data_source.data['alpha'] = [0.7] * len(pids)

        # Edges
        graph.edge_renderer.data_source.data = dict(start=[pim[u] for u, _ in gs.edges],
                                                    end=[pim[v] for _, v in gs.edges])

        # start of layout code
        xs, ys = minmax_scale(df['x']) * 100, minmax_scale(df['y']) * 100
        p = figure(width=width,
                   height=height,
                   x_range=(-5, 105),
                   y_range=(-5, 105),
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

        graph_layout = dict(zip(pids_int, zip(xs, ys)))
        graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

        graph.node_renderer.glyph = Circle(
            radius='radius', fill_alpha='alpha', line_alpha='alpha', line_width='line_width', fill_color='color'
        )
        graph.node_renderer.hover_glyph = Circle(
            radius='radius', fill_alpha=1.0, line_alpha=1.0, line_width='line_width', fill_color='color'
        )

        graph.edge_renderer.glyph = MultiLine(line_color='lightgrey', line_alpha=0.2, line_width=0.5)
        graph.edge_renderer.hover_glyph = MultiLine(line_color='grey', line_alpha=1.0, line_width=2)

        graph.inspection_policy = NodesAndLinkedEdges()

        p.renderers.append(graph)

        # Compute component boundaries and centers
        boundaries = compute_component_boundaries(rescaled_comp_corrds(df))

        # Plot boundaries and compute centers
        lxs = []
        lys = []
        labels = []
        for comp, polygon in boundaries.items():
            labels.append(f"#{comp + 1}")
            xc, yx = polygon.centroid.xy
            lxs.append(xc)
            lys.append(yx)
            xs, ys = shapely_to_bokeh_multipolygons(polygon)  # each returns lists with one record
            src = ColumnDataSource({"comp": [comp], "xs": xs, "ys": ys, "alpha": [0.05], "color": [palette[comp]]})
            p.multi_polygons(xs="xs", ys="ys", source=src, color="color", fill_alpha="alpha", line_width=0.1)

        # Plot labels in centers
        source = ColumnDataSource({'x': lxs, 'y': lys, 'name': labels})
        labels = LabelSet(x='x', y='y', text='name', source=source,
                          background_fill_color='white',
                          text_font_size='15px',
                          background_fill_alpha=.9)
        p.renderers.append(labels)

        # Add topic tags in the top left corner
        if topics_tags is not None:
            for i, c in enumerate(sorted(set(comps))):
                p.rect(x=-3, y=103 - i * 2, width=1, height=2, fill_color=palette[c], line_color=None)
                p.add_layout(Label(
                    x=-2, y=102 - i * 2,
                    text=f"#{c + 1}: {', '.join(t for t, _ in topics_tags[c][:5])}",
                    text_font_size='11px',
                    text_align="left",
                    background_fill_color="white",
                    background_fill_alpha=0.7,
                ))

        Plotter.remove_wheel_zoom_tool(p)
        if interactive:
            hover_tags = [
                ("Author(s)", '@authors'),
                ("Journal", '@journal'),
                ("Year", '@year'),
                ("Type", '@type'),
                ("Cited by", '@total paper(s) total'),
                ("Source", '@source'),
                ("Mesh", '@mesh'),
                ("Keywords", '@keywords'),
                ("Topic", '@topic'),
            ]
            if topics_tags is not None:
                hover_tags.append(("Topic tags", '@topic_tags'))
            if topics_meshs is not None:
                hover_tags.append(("Topic Mesh tags", '@topic_meshs'))
            p.add_tools(HoverTool(tooltips=Plotter._paper_html_tooltips(source, hover_tags, idname='pid'),
                                  renderers=[graph]))
            if add_callback:
                p.js_on_event('tap', Plotter._paper_callback(graph.node_renderer.data_source, idname='pid'))

        return p

    @staticmethod
    def _paper_callback(ds, idname='id'):
        return CustomJS(args=dict(ds=ds), code=f"""
            var data = ds.data, selected = ds.selected.indices;

            // Decode params from URL
            const jobid = new URL(window.location).searchParams.get('jobid');
            const query = new URL(window.location).searchParams.get('query');

            // Max number of papers to be opened, others will be ignored
            var MAX_PAPERS = 3;

            for (var i = 0; i < Math.min(MAX_PAPERS, selected.length); i++){{
                window.open('/paper?query=' + query + '&id=' + data['{idname}'][selected[i]] + '&jobid=' + jobid, '_blank');
            }}
        """)

    @staticmethod
    def remove_wheel_zoom_tool(p):
        p.tools = [tool for tool in p.tools if tool.__class__.__name__ != 'WheelZoomTool']
