import json
import logging
from string import Template

import holoviews as hv
import numpy as np
from bokeh.colors import RGB
from bokeh.core.properties import value
from bokeh.embed import components
from bokeh.models import ColumnDataSource, CustomJS, Legend, LegendItem
# Tools used: hover,pan,tap,wheel_zoom,box_zoom,reset,save
from bokeh.models import LinearColorMapper, PrintfTickFormatter, ColorBar
from bokeh.models import NumeralTickFormatter
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from holoviews import dim
from matplotlib import pyplot as plt
from more_itertools import unique_everseen
from wordcloud import WordCloud

from pysrc.papers.plot_preprocessor import PlotPreprocessor
from pysrc.papers.utils import LOCAL_BASE_URL, get_topic_word_cloud_data, \
    get_frequent_tokens, cut_authors_list, ZOOM_OUT, ZOOM_IN, zoom_name, trim, rgb2hex

TOOLS = "hover,pan,tap,wheel_zoom,box_zoom,reset,save"
hv.extension('bokeh')

logger = logging.getLogger(__name__)

MAX_AUTHOR_LENGTH = 100
MAX_JOURNAL_LENGTH = 100
MAX_LINEAR_AXIS = 100

PLOT_WIDTH = 870
SHORT_PLOT_HEIGHT = 300
TALL_PLOT_HEIGHT = 600

PAPERS_PLOT_WIDTH = 670
PAPERS_PLOT_HEIGHT = 400

WORD_CLOUD_WIDTH = 200
WORD_CLOUD_HEIGHT = 300
MAX_WORDS = 20


def visualize_analysis(analyzer):
    # Initialize plotter after completion of analysis
    plotter = Plotter(analyzer=analyzer)
    # Order is important here!
    paper_statistics, word_cloud, zoom_out_callback = plotter.papers_statistics_and_word_cloud_and_callback()
    if analyzer.similarity_graph.nodes():
        topics_hierarchy = plotter.topics_hierarchy()
        return {
            'topics_analyzed': True,
            'n_papers': analyzer.n_papers,
            'n_citations': int(analyzer.df['total'].sum()),
            'n_topics': len(analyzer.components),
            'comp_other': analyzer.comp_other,
            'components_similarity': [components(plotter.heatmap_topics_similarity())],
            'component_size_summary': [components(plotter.component_size_summary())],
            'component_years_summary_boxplots': [components(plotter.component_years_summary_boxplots())],
            'topics_info_and_word_cloud_and_callback':
                [(components(p), Plotter.word_cloud_prepare(wc), zoom_in_callback) for
                 (p, wc, zoom_in_callback) in plotter.topics_info_and_word_cloud_and_callback()],
            'top_cited_papers': [components(plotter.top_cited_papers())],
            'max_gain_papers': [components(plotter.max_gain_papers())],
            'max_relative_gain_papers': [components(plotter.max_relative_gain_papers())],
            'component_sizes': plotter.component_sizes(),
            'component_ratio': [components(plotter.component_ratio())],
            'papers_stats': [components(paper_statistics)],
            'papers_word_cloud': Plotter.word_cloud_prepare(word_cloud),
            'papers_zoom_out_callback': zoom_out_callback,
            'author_statistics': plotter.author_statistics(),
            'journal_statistics': plotter.journal_statistics(),
            'topics_hierarchy': [components(topics_hierarchy)] if topics_hierarchy is not None else []
        }
    else:
        return {
            'topics_analyzed': False,
            'n_papers': analyzer.n_papers,
            'n_citations': int(analyzer.df['total'].sum()),
            'n_topics': 0,
            'top_cited_papers': [components(plotter.top_cited_papers())],
            'max_gain_papers': [components(plotter.max_gain_papers())],
            'max_relative_gain_papers': [components(plotter.max_relative_gain_papers())],
            'papers_stats': [components(paper_statistics)],
            'papers_word_cloud': Plotter.word_cloud_prepare(word_cloud),
            'papers_zoom_out_callback': zoom_out_callback,
            'author_statistics': plotter.author_statistics(),
            'journal_statistics': plotter.journal_statistics(),
        }


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
    def paper_callback(ds, source):
        if source in ['Semantic Scholar', 'Pubmed']:
            base = LOCAL_BASE_URL.substitute(source=source)
        else:
            raise ValueError(f"Wrong value of source: {source}")
        return CustomJS(args=dict(ds=ds, base=base), code="""
            var data = ds.data, selected = ds.selected.indices;

            // Decode jobid from URL
            const jobid = new URL(window.location).searchParams.get('jobid');

            // Max amount of papers to be opened, others will be ignored
            var MAX_AMOUNT = 3;

            for (var i = 0; i < Math.min(MAX_AMOUNT, selected.length); i++){
                window.open(base + data['id'][selected[i]] + '&jobid=' + jobid, '_blank');
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
            self.analyzer.similarity_graph, self.analyzer.df, self.analyzer.comp_sizes
        )

        step = 30
        cmap = plt.cm.get_cmap('PuBu', step)
        colors = [RGB(*[round(c * 255) for c in cmap(i)[:3]]) for i in range(step)]
        mapper = LinearColorMapper(palette=colors,
                                   low=similarity_df.similarity.min(),
                                   high=similarity_df.similarity.max())

        p = figure(title="Similarity between topics",
                   x_range=topics, y_range=topics,
                   x_axis_location="below", plot_width=PLOT_WIDTH, plot_height=PAPERS_PLOT_HEIGHT,
                   tools=TOOLS, toolbar_location='above',
                   tooltips=[('Topic 1', '@comp_x'),
                             ('Topic 2', '@comp_y'),
                             ('Similarity', '@similarity')])

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

    def component_size_summary(self):
        logger.debug('Summary component detailed info visualization')

        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        components, data = PlotPreprocessor.component_size_summary_data(
            self.analyzer.df, self.analyzer.components, min_year, max_year
        )

        p = figure(x_range=[min_year - 1, max_year + 1], plot_width=PLOT_WIDTH, plot_height=SHORT_PLOT_HEIGHT,
                   title="Topics by Year", toolbar_location="right", tools=TOOLS,
                   tooltips=[('Topic', '$name'), ('Amount', '@$name')])

        # NOTE: VBar is invisible (alpha = 0) to provide tooltips on hover as stacked area does not support them
        p.vbar_stack(components, x='years', width=0.9, color=self.comp_palette, source=data, alpha=0)

        # VArea is actually displayed
        p.varea_stack(stackers=components, x='years', color=self.comp_palette, source=data, alpha=0.5)

        # these are a dummy glyphs to help draw the legend
        dummy_for_legend = [p.line(x=[1, 1], y=[1, 1], line_width=15, color=c, name='dummy_for_legend')
                            for c in self.comp_palette]
        legend = Legend(items=[
            LegendItem(label=f'{c} OTHER' if int(c) - 1 == self.analyzer.comp_other else value(c),
                       renderers=[dummy_for_legend[i]],
                       index=i) for i, c in enumerate(components)
        ])
        p.add_layout(legend)

        p.y_range.start = 0
        p.xgrid.grid_line_color = None
        p.axis.minor_tick_line_color = None
        p.outline_line_color = None
        p.legend.location = "top_left"
        p.legend.orientation = "horizontal"

        return p

    def component_years_summary_boxplots(self):
        logger.debug('Summary component year detailed info visualization on boxplot')

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
        boxwhisker.opts(width=PLOT_WIDTH, height=SHORT_PLOT_HEIGHT,
                        box_fill_color=dim('Topic').str(), cmap=Plotter.factors_colormap(len(components)))
        return hv.render(boxwhisker, backend='bokeh')

    def topics_info_and_word_cloud_and_callback(self):
        logger.debug('Per component detailed info visualization')

        # Prepare layouts
        n_comps = len(self.analyzer.components)
        result = []

        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        for comp in range(n_comps):
            df_comp = self.analyzer.df[self.analyzer.df['comp'] == comp]
            ds = ColumnDataSource(PlotPreprocessor.article_view_data_source(
                df_comp, min_year, max_year, True, width=PAPERS_PLOT_WIDTH
            ))
            # Add type coloring
            ds.add([self.pub_types_colors_map[t] for t in df_comp['type']], 'color')
            plot = self.__serve_scatter_article_layout(ds=ds,
                                                       year_range=[min_year, max_year],
                                                       title="Publications", width=PAPERS_PLOT_WIDTH)
            plot.circle(x='year', y='y', fill_alpha=0.5, source=ds, size='size',
                        line_color='color', fill_color='color', legend_field='type')
            plot.legend.location = "top_left"

            # Word cloud description of topic by titles and abstracts
            kwds = get_topic_word_cloud_data(self.analyzer.df_kwd, comp)
            color = (self.comp_colors[comp].r, self.comp_colors[comp].g, self.comp_colors[comp].b)
            wc = WordCloud(background_color="white", width=WORD_CLOUD_WIDTH, height=WORD_CLOUD_HEIGHT,
                           color_func=lambda *args, **kwargs: color,
                           max_words=MAX_WORDS, min_font_size=10, max_font_size=30)
            wc.generate_from_frequencies(kwds)

            # Create Zoom In callback
            id_list = list(df_comp['id'])
            zoom_in_callback = self.zoom_callback(id_list, self.analyzer.source,
                                                  zoom=ZOOM_IN,
                                                  query=self.analyzer.query)

            result.append((plot, wc, zoom_in_callback))

        return result

    def component_sizes(self):
        assigned_comps = self.analyzer.df[self.analyzer.df['comp'] >= 0]
        d = dict(assigned_comps.groupby('comp')['id'].count())
        return [int(d[k]) for k in range(len(d))]

    def component_ratio(self):
        comps, ratios = PlotPreprocessor.component_ratio_data(self.analyzer.df)
        colors = [self.comp_palette[int(c) - 1] for c in comps]
        source = ColumnDataSource(data=dict(comps=comps, ratios=ratios, colors=colors))

        p = figure(plot_width=PLOT_WIDTH, plot_height=SHORT_PLOT_HEIGHT,
                   toolbar_location="above", tools=TOOLS, x_range=comps)
        p.vbar(x='comps', top='ratios', width=0.8, fill_alpha=0.5, color='colors', source=source)
        p.hover.tooltips = [("Topic", '@comps'), ("Amount", '@ratios %')]

        p.xaxis.axis_label = 'Topic'
        p.yaxis.axis_label = 'Percentage of papers'
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.axis.minor_tick_line_color = None
        p.outline_line_color = None
        p.js_on_event('tap', self.topic_callback(source))

        return p

    def top_cited_papers(self):
        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        ds = ColumnDataSource(PlotPreprocessor.article_view_data_source(
            self.analyzer.top_cited_df, min_year, max_year, False, width=PAPERS_PLOT_WIDTH
        ))
        # Add type coloring
        ds.add([self.pub_types_colors_map[t] for t in self.analyzer.top_cited_df['type']], 'color')

        plot = self.__serve_scatter_article_layout(ds=ds,
                                                   year_range=[min_year, max_year],
                                                   title=f'{len(self.analyzer.top_cited_df)} top cited papers',
                                                   width=PLOT_WIDTH)

        plot.circle(x='year', y='y', fill_alpha=0.5, source=ds, size='size',
                    line_color='color', fill_color='color', legend_field='type')
        plot.legend.location = "top_left"
        return plot

    def max_gain_papers(self):
        logger.debug('Different colors encode different papers')
        cols = ['year', 'id', 'title', 'authors', 'paper_year', 'count']
        max_gain_df = self.analyzer.max_gain_df[cols].replace(np.nan, "Undefined")
        max_gain_df['authors'] = max_gain_df['authors'].apply(lambda authors: cut_authors_list(authors))
        ds_max = ColumnDataSource(max_gain_df)

        factors = self.analyzer.max_gain_df['id'].unique()
        colors = self.factor_colors(factors)

        year_range = [self.analyzer.min_year - 1, self.analyzer.max_year + 1]
        p = figure(tools=TOOLS, toolbar_location="above",
                   plot_width=PLOT_WIDTH, plot_height=SHORT_PLOT_HEIGHT, x_range=year_range,
                   y_axis_type="log" if max(self.analyzer.max_gain_df['count']) > MAX_LINEAR_AXIS else "linear",
                   title='Max gain of citations per year')
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Number of citations'
        p.yaxis.formatter = NumeralTickFormatter(format='0,0')
        p.hover.tooltips = self._html_tooltips([
            ("Author(s)", '@authors'),
            ("Year", '@paper_year'),
            ("Cited by", '@count papers in @year')
        ])
        p.js_on_event('tap', self.paper_callback(ds_max, self.analyzer.source))
        # Use explicit bottom for log scale as workaround
        # https://github.com/bokeh/bokeh/issues/6536
        bottom = min(self.analyzer.max_gain_df['count']) - 0.01
        p.vbar(x='year', width=0.8, top='count', bottom=bottom,
               fill_alpha=0.5, source=ds_max, fill_color=colors, line_color=colors)
        return p

    def max_relative_gain_papers(self):
        logger.debug('Top papers in relative gain for each year')
        logger.debug('Relative gain (year) = Citation Gain (year) / Citations before year')
        logger.debug('Different colors encode different papers')
        cols = ['year', 'id', 'title', 'authors', 'paper_year', 'rel_gain']
        max_rel_gain_df = self.analyzer.max_rel_gain_df[cols].replace(np.nan, "Undefined")
        max_rel_gain_df['authors'] = max_rel_gain_df['authors'].apply(lambda authors: cut_authors_list(authors))
        ds_max = ColumnDataSource(max_rel_gain_df)

        factors = self.analyzer.max_rel_gain_df['id'].astype(str).unique()
        colors = self.factor_colors(factors)

        year_range = [self.analyzer.min_year - 1, self.analyzer.max_year + 1]
        p = figure(tools=TOOLS, toolbar_location="above",
                   plot_width=PLOT_WIDTH, plot_height=SHORT_PLOT_HEIGHT, x_range=year_range,
                   y_axis_type="log" if max(self.analyzer.max_rel_gain_df['rel_gain']) > MAX_LINEAR_AXIS else "linear",
                   title='Max relative gain of citations per year')
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Relative Gain of Citations'
        p.yaxis.formatter = NumeralTickFormatter(format='0,0')
        p.hover.tooltips = self._html_tooltips([
            ("Author(s)", '@authors'),
            ("Year", '@paper_year'),
            ("Relative Gain", '@rel_gain in @year')])
        p.js_on_event('tap', self.paper_callback(ds_max, self.analyzer.source))
        # Use explicit bottom for log scale as workaround
        # https://github.com/bokeh/bokeh/issues/6536
        bottom = min(self.analyzer.max_rel_gain_df['rel_gain']) - 0.01
        p.vbar(x='year', width=0.8, top='rel_gain', bottom=bottom, source=ds_max,
               fill_alpha=0.5, fill_color=colors, line_color=colors)
        return p

    @staticmethod
    def article_citation_dynamics(df, pid):
        d = ColumnDataSource(PlotPreprocessor.article_citation_dynamics_data(df, pid))

        p = figure(tools=TOOLS, toolbar_location="above", plot_width=PLOT_WIDTH,
                   plot_height=SHORT_PLOT_HEIGHT, title="Number of Citations per Year")
        p.vbar(x='x', width=0.8, top='y', source=d, color='#A6CEE3', line_width=3)
        p.xaxis.axis_label = "Year"
        p.yaxis.axis_label = "Number of citations"
        p.hover.tooltips = [
            ("Year", "@x"),
            ("Cited by", "@y paper(s) in @x"),
        ]

        return p

    def papers_statistics_and_word_cloud_and_callback(self):
        ds_stats = ColumnDataSource(PlotPreprocessor.papers_statistics_data(self.analyzer.df))

        year_range = [self.analyzer.min_year - 1, self.analyzer.max_year + 1]
        p = figure(tools=TOOLS, toolbar_location="above",
                   plot_width=PAPERS_PLOT_WIDTH, plot_height=PAPERS_PLOT_HEIGHT,
                   x_range=year_range, title='Papers per year')
        p.y_range.start = 0
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Amount of papers'
        p.hover.tooltips = [("Amount", '@counts'), ("Year", '@year')]

        if self.analyzer.min_year != self.analyzer.max_year:
            # NOTE: VBar is invisible (alpha=0) to provide tooltips, as in self.component_size_summary()
            p.vbar(x='year', width=0.8, top='counts', fill_alpha=0, line_alpha=0, source=ds_stats)
            # VArea is actually displayed
            ds_stats.data['bottom'] = [0] * len(ds_stats.data['year'])
            p.varea(x='year', y1='bottom', y2='counts', fill_alpha=0.5, source=ds_stats)
        else:
            # NOTE: VBar is invisible (alpha=0) to provide tooltips, as in self.component_size_summary()
            p.vbar(x='year', width=0.8, top='counts', source=ds_stats)

        # Build word cloud, size is proportional to token frequency
        kwds = get_frequent_tokens(self.analyzer.top_cited_df, query=None)
        wc = WordCloud(background_color="white", width=WORD_CLOUD_WIDTH, height=WORD_CLOUD_HEIGHT,
                       color_func=lambda *args, **kwargs: 'black',
                       max_words=MAX_WORDS, min_font_size=10, max_font_size=30)
        wc.generate_from_frequencies(kwds)

        # Create Zoom Out callback
        id_list = list(self.analyzer.df['id'])
        zoom_out_callback = self.zoom_callback(id_list, self.analyzer.source,
                                               zoom=ZOOM_OUT, query=self.analyzer.query)

        return p, wc, zoom_out_callback

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

    def __serve_scatter_article_layout(self, ds, year_range, title, width=PLOT_WIDTH):
        min_year, max_year = year_range
        p = figure(tools=TOOLS, toolbar_location="above",
                   plot_width=width, plot_height=PAPERS_PLOT_HEIGHT,
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
        p.js_on_event('tap', self.paper_callback(ds, self.analyzer.source))

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

    def topics_hierarchy(self):
        """ Plot topics hierarchy ignoring OTHER topic"""
        dendrogram = self.analyzer.topics_dendrogram
        if len(dendrogram) == 0:
            return None
        w = len(set(dendrogram[0].keys())) * 10 + 10
        dy = 3
        hm = figure(x_range=[-10, w + 10],
                    y_range=[-3, dy * (len(dendrogram) + 1)],
                    width=PLOT_WIDTH, height=100 * (len(dendrogram) + 1), tools=[])

        paths = []
        for i, level in enumerate(dendrogram):
            if i == 0:
                for k in level.keys():
                    paths.append([k])
            # Edges
            for k, v in level.items():
                for p in paths:
                    if p[i] == k:
                        p.append(v)
        # Add root as last item
        for p in paths:
            p.append(0)

        # Radix sort or paths to ensure no overlaps
        for i in range(0, len(dendrogram) + 1):
            paths.sort(key=lambda p: p[i])

        # Draw edges
        for i in range(len(dendrogram) + 2):
            if i == 0:  # Leaves
                order = dict([(v, j) for j, v in enumerate(unique_everseen([p[i] for p in paths]))])
                dx = int(w / (len(order) + 1))
            else:  # Edges
                new_order = dict([(v, j) for j, v in enumerate(unique_everseen([p[i] for p in paths]))])
                new_dx = int(w / (len(new_order) + 1))
                for p in paths:
                    hm.line([(order[p[i - 1]] + 1) * dx, (new_order[p[i]] + 1) * new_dx],
                            [(i - 1) * dy, i * dy], line_color='black')
                order = new_order
                dx = new_dx

        # Draw leaves
        topics_colors = Plotter.topics_palette_rgb(self.analyzer.df)
        order = dict([(v, j) for j, v in enumerate(unique_everseen([p[0] for p in paths]))])
        dx = int(w / (len(order) + 1))
        for v, j in order.items():
            hm.circle(x=(j + 1) * dx, y=0, size=15, line_color="black", fill_color=topics_colors[v])
        hm.text(x=[(j + 1) * dx for _, j in order.items()],
                y=[-1] * len(order),
                text=[str(v + 1) for v, _ in order.items()],
                text_baseline='middle', text_font_size='11pt')

        hm.axis.major_tick_line_color = None
        hm.axis.minor_tick_line_color = None
        hm.axis.major_label_text_color = None
        hm.axis.major_label_text_font_size = '0pt'
        hm.axis.axis_line_color = None
        hm.grid.grid_line_color = None
        hm.outline_line_color = None

        return hm

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
        palette = [Plotter.color_to_rgb(cmap(i)) for i in range(len(factors))]
        colors = factor_cmap('id', palette=palette, factors=factors)
        return colors

    @staticmethod
    def topics_palette_rgb(df):
        n_comps = len(set(df['comp']))
        cmap = Plotter.factors_colormap(n_comps)
        return dict([(i, Plotter.color_to_rgb(cmap(i))) for i in range(n_comps)])

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
        return dict([(k, v.to_hex()) for k, v in Plotter.topics_palette_rgb(df).items()])
