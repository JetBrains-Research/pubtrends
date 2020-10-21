import datetime
import json
import re
from queue import Queue

import pandas as pd
from bokeh.embed import components
from bokeh.models import HoverTool
from bokeh.plotting import figure
from wordcloud import WordCloud

from pysrc.papers.plot.plotter import Plotter

TOOLS = "hover,pan,tap,wheel_zoom,box_zoom,reset,save"
PLOT_WIDTH = 900
PLOT_HEIGHT = 300
WC_HEIGHT = 600
RECENT_SEARCHES = 100


def prepare_stats_data(logfile):
    terms_searches = []
    paper_searches = []
    recent_searches = Queue(maxsize=RECENT_SEARCHES)
    terms = []

    for line in open(logfile).readlines():
        if 'INFO' not in line:
            continue
        search_date = re.search('[\\d-]+ [\\d:]+,\\d+', line)
        if search_date is None:
            continue
        date = datetime.datetime.strptime(search_date.group(0), '%Y-%m-%d %H:%M:%S,%f')
        if '/process regular search addr:' in line:
            terms_searches.append(date)
            if recent_searches.full():
                recent_searches.get()  # Free some space
            terms.append(re.sub('(.*"query": ")|(", "source.*)', '', line.strip()))
            recent_searches.put((date, re.sub(".*args:", "", line.strip())))
        if '/search_paper addr:' in line:
            paper_searches.append(date)

    result = {}
    total_terms_searches = len(terms_searches)
    result['total_terms_searches'] = total_terms_searches
    if total_terms_searches:
        p = prepare_timeseries(terms_searches, 'Terms searches')
        result['terms_searches_plot'] = [components(p)]

    total_paper_searches = len(paper_searches)
    result['total_paper_searches'] = total_paper_searches
    if total_paper_searches:
        p = prepare_timeseries(paper_searches, 'Paper searches')
        result['paper_searches_plot'] = [components(p)]

    recent_searches_query_links = []
    while not recent_searches.empty():
        date, rs = recent_searches.get()
        args_map = json.loads(rs)
        query = args_map['query']
        link = f'/result?{"&".join([f"{a}={v}" for a, v in args_map.items()])}'
        recent_searches_query_links.append((date.strftime('%Y-%m-%d'), query, link))
    result['recent_searches'] = recent_searches_query_links[::-1]

    # Generate a word cloud image
    text = ' '.join(terms).replace(',', ' ').replace('[^a-zA-Z0-9]+', ' ')
    if text:  # Check that string is not empty
        wc = WordCloud(collocations=False, width=PLOT_WIDTH - 40, height=WC_HEIGHT - 40,
                       background_color='white', max_font_size=100).generate(text)
        result['word_cloud'] = Plotter.word_cloud_prepare(wc)

    return result


def prepare_timeseries(dates, title):
    df_terms_searches = pd.DataFrame({'date': dates, 'count': 1})
    df_terms_searches_grouped = df_terms_searches.groupby(pd.Grouper(key='date', freq='D')).sum()
    p = figure(plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, tools=TOOLS,
               x_axis_type='datetime', title=title)
    p.line('date', 'count', source=df_terms_searches_grouped)
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [('Date', '@date{%F}'), ('Count', '@count')]
    hover.formatters = {'@date': 'datetime'}
    return p
