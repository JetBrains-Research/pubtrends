import json
import logging
from string import Template

import holoviews as hv
import numpy as np
from bokeh.colors import RGB
from bokeh.core.properties import value
from bokeh.embed import components
from bokeh.models import ColumnDataSource, CustomJS
# Tools used: hover,pan,tap,wheel_zoom,box_zoom,reset,save
from bokeh.models import LinearColorMapper, PrintfTickFormatter, ColorBar
from bokeh.models import NumeralTickFormatter
from bokeh.palettes import Category20
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from holoviews import dim
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from models.keypaper.plot_preprocessor import PlotPreprocessor
from models.keypaper.utils import LOCAL_BASE_URL, get_topic_word_cloud_data, \
    get_frequent_tokens, cut_authors_list, ZOOM_OUT, ZOOM_IN, zoom_name, trim, hex2rgb, rgb2hex

TOOLS = "hover,pan,tap,wheel_zoom,box_zoom,reset,save"
hv.extension('bokeh')

log = logging.getLogger(__name__)

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
        return {
            'topics_analyzed': True,
            'n_papers': analyzer.n_papers,
            'n_citations': int(analyzer.df['total'].sum()),
            'n_subtopics': len(analyzer.components),
            'comp_other': analyzer.comp_other,
            'components_similarity': [components(plotter.heatmap_clusters())],
            'component_size_summary': [components(plotter.component_size_summary())],
            'component_years_summary_boxplots': [components(plotter.component_years_summary_boxplots())],
            'subtopics_info_and_word_cloud_and_callback':
                [(components(p), Plotter.word_cloud_prepare(wc), zoom_in_callback) for
                 (p, wc, zoom_in_callback) in plotter.subtopics_info_and_word_cloud_and_callback()],
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
        }
    else:
        return {
            'topics_analyzed': False,
            'n_papers': analyzer.n_papers,
            'n_citations': int(analyzer.df['total'].sum()),
            'n_subtopics': 0,
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
                n_comps = len(self.analyzer.components)
                if n_comps > 20:
                    raise ValueError(f'Too big number of components {n_comps}')
                self.comp_palette = [RGB(*hex2rgb(c)) for c in Category20[20][:n_comps]]
                self.comp_colors = dict(enumerate(self.comp_palette))

            n_pub_types = len(self.analyzer.pub_types)
            pub_types_cmap = plt.cm.get_cmap('jet', n_pub_types)
            self.pub_types_colors_map = dict(
                zip(self.analyzer.pub_types,
                    [RGB(*[round(c * 255) for c in pub_types_cmap(i)[:3]]) for i in range(n_pub_types)]))

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

    def heatmap_clusters(self):
        log.info('Visualizing components with heatmap')

        cluster_edges, clusters = PlotPreprocessor.heatmap_clusters_data(
            self.analyzer.similarity_graph, self.analyzer.df, self.analyzer.comp_sizes
        )

        step = 30
        cmap = plt.cm.get_cmap('PuBu', step)
        colors = [RGB(*[round(c * 255) for c in cmap(i)[:3]]) for i in range(step)]
        mapper = LinearColorMapper(palette=colors,
                                   low=cluster_edges.density.min(),
                                   high=cluster_edges.density.max())

        p = figure(title="Similarity between groups",
                   x_range=clusters, y_range=clusters,
                   x_axis_location="below", plot_width=PLOT_WIDTH, plot_height=PAPERS_PLOT_HEIGHT,
                   tools=TOOLS, toolbar_location='above',
                   tooltips=[('Subtopic 1', '#@comp_x'),
                             ('Subtopic 2', '#@comp_y'),
                             ('Density', '@density, @value')])

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

    def component_size_summary(self):
        log.info('Summary component detailed info visualization')

        min_year, max_year = self.analyzer.min_year, self.analyzer.max_year
        components, data = PlotPreprocessor.component_size_summary_data(
            self.analyzer.df, self.analyzer.components, min_year, max_year
        )

        p = figure(x_range=[min_year - 1, max_year + 1], plot_width=PLOT_WIDTH, plot_height=SHORT_PLOT_HEIGHT,
                   title="Subtopics by Year", toolbar_location="right", tools=TOOLS,
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
        log.info('Summary component year detailed info visualization on boxplot')

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
        boxwhisker = hv.BoxWhisker((labels, values), 'Subtopic', 'Publications year')
        boxwhisker.opts(width=PLOT_WIDTH, height=SHORT_PLOT_HEIGHT,
                        box_fill_color=dim('Subtopic').str(), cmap='tab20')
        return hv.render(boxwhisker, backend='bokeh')

    def subtopics_info_and_word_cloud_and_callback(self):
        log.info('Per component detailed info visualization')

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
                        line_color='color', fill_color='color', legend='type')
            plot.legend.location = "top_left"

            # Word cloud description of subtopic by titles and abstracts
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

        p = figure(plot_width=PLOT_WIDTH, plot_height=30 + 50 * len(comps),
                   toolbar_location="above", tools=TOOLS, y_range=comps)
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
                    line_color='color', fill_color='color', legend='type')
        plot.legend.location = "top_left"
        return plot

    def max_gain_papers(self):
        log.info('Different colors encode different papers')
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
        log.info('Top papers in relative gain for each year')
        log.info('Relative gain (year) = Citation Gain (year) / Citations before year')
        log.info('Different colors encode different papers')
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
        author = self.analyzer.author_stats['author']
        sum = self.analyzer.author_stats['sum']
        if self.analyzer.similarity_graph.nodes():
            subtopics = self.analyzer.author_stats.apply(
                lambda row: self._to_colored_circle(row['comp'], row['counts'], row['sum']), axis=1)
        else:
            subtopics = [' '] * len(self.analyzer.author_stats)  # Ignore subtopics
        return list(zip(map(lambda a: trim(a, MAX_AUTHOR_LENGTH), author), sum, subtopics))

    def journal_statistics(self):
        journal = self.analyzer.journal_stats['journal']
        sum = self.analyzer.journal_stats['sum']
        if self.analyzer.similarity_graph.nodes():
            subtopics = self.analyzer.journal_stats.apply(
                lambda row: self._to_colored_circle(row['comp'], row['counts'], row['sum']), axis=1)
        else:
            subtopics = [' '] * len(self.analyzer.journal_stats)  # Ignore subtopics
        return list(zip(map(lambda j: trim(j, MAX_JOURNAL_LENGTH), journal), sum, subtopics))

    def _to_colored_circle(self, components, counts, sum, top=3):
        # html code to generate circles corresponding to the most popular subtopics
        return ' '.join(
            map(lambda topic: f'''<a class="fas fa-circle" style="color:{self.comp_colors[topic[0]]}"
                                     href="#subtopic-{topic[0] + 1}"></a>
                                  <span class="bk" style="color:black">{int(topic[1] / sum * 100)}%</span> ''',
                zip(components[:top], counts[:top])))

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

    @staticmethod
    def word_cloud_prepare(wc):
        return json.dumps([(word, int(position[0]), int(position[1]),
                            int(font_size), orientation is not None,
                            rgb2hex(color))
                           for (word, count), font_size, position, orientation, color in wc.layout_])

    @staticmethod
    def subtopics_palette(df):
        return dict(enumerate(Category20[20][:len(set(df['comp']))]))