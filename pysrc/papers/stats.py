import datetime
import re

import numpy as np
import pandas as pd
from bokeh.embed import components
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from wordcloud import WordCloud

from pysrc.papers.plotter import Plotter

TOOLS = "hover,pan,tap,wheel_zoom,box_zoom,reset,save"
PLOT_WIDTH = 900
PLOT_HEIGHT = 300
WC_HEIGHT = 600


def prepare_stats_data(logfile):
    visits = []
    terms_searches = []
    paper_searches = []
    terms = []
    for line in open(logfile).readlines():
        if 'INFO' not in line:
            continue
        search = re.search('[\\d-]+ [\\d:]+,\\d+', line)
        if search is None:
            continue
        date = datetime.datetime.strptime(search.group(0), '%Y-%m-%d %H:%M:%S,%f')
        if '/ addr:' in line:
            visits.append(date)
        if '/process regular search addr:' in line:
            terms_searches.append(date)
        if '/search_paper addr:' in line:
            paper_searches.append(date)
        if '/result success addr:' in line:
            terms.append(re.sub('(.*"query": ")|(", "source.*)', '', line.strip()))

    result = {}
    total_visits = len(visits)
    result['total_visits'] = total_visits
    if total_visits:
        p = prepare_timeseries(visits, 'Terms searches')
        result['visits_plot'] = [components(p)]

    total_terms_searches = len(terms_searches)
    result['total_terms_searches'] = total_terms_searches
    if total_terms_searches:
        p = prepare_timeseries(terms_searches, 'Terms searches')
        result['terms_searches_plot'] = [components(p)]

    total_paper_searches = len(paper_searches)
    result['total_paper_searches'] = total_paper_searches
    if total_paper_searches:
        p = prepare_timeseries(paper_searches, 'Terms searches')
        result['paper_searches_plot'] = [components(p)]

    # Generate a word cloud image
    text = ' '.join(terms).replace(',', ' ').replace('"', '')
    wc = WordCloud(width=PLOT_WIDTH, height=WC_HEIGHT, background_color='white', max_font_size=100).generate(text)
    result['word_cloud'] = Plotter.word_cloud_prepare(wc)
    return result


def prepare_timeseries(dates, title):
    df_terms_searches = pd.DataFrame({'count': np.ones(len(dates))}, index=dates)
    df_by_month = df_terms_searches.resample('M').sum()
    df_by_month['date'] = [d.strftime("%m/%Y") for d in df_by_month.index]
    df_by_month.reset_index(drop=True, inplace=True)
    p = figure(plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, x_range=df_by_month['date'], tools=TOOLS,
               title=title)
    p.vbar(x='date', top='count', bottom=0, source=ColumnDataSource(df_by_month), line_width=3, width=0.8)
    p.hover.tooltips = [("Date", "@date"), ("Count", "@count")]
    return p
