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
RECENT = 50


def prepare_stats_data(logfile):
    terms_searches_dates = []
    recent_searches = Queue(maxsize=RECENT)
    searches_infos = dict()
    terms = []

    paper_searches_dates = []
    recent_papers = Queue(maxsize=RECENT)
    papers_infos = dict()

    for line in open(logfile).readlines():
        try:
            if 'INFO' not in line:
                continue
            search_date = re.search('[\\d-]+ [\\d:]+,\\d+', line)
            if search_date is None:
                continue
            date = datetime.datetime.strptime(search_date.group(0), '%Y-%m-%d %H:%M:%S,%f')

            if '/process regular search addr:' in line:
                terms_searches_dates.append(date)
                args = json.loads(re.sub(".*args:", "", line.strip()))
                terms.append(args['query'].replace(',', ' ').replace('[^a-zA-Z0-9]+', ' '))
                jobid = args['jobid']
                if not jobid:
                    continue
                if jobid not in searches_infos:  # Don't re-add already existing task
                    searches_infos[jobid] = dict(jobid=jobid, start=date, args=args)
                    if recent_searches.full():
                        recent_searches.get()  # Free some space
                    recent_searches.put(jobid)

            if '/search_terms with fixed jobid addr:' in line:
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid or jobid:
                    continue
                searches_infos[jobid] = dict(jobid=jobid, start=date, args=args)
                if recent_searches.full():
                    recent_searches.get()  # Free some space
                recent_searches.put(jobid)
                terms_searches_dates.append(date)
                terms.append(args['query'].replace(',', ' ').replace('[^a-zA-Z0-9]+', ' '))

            if '/result success addr:' in line:
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid:
                    continue
                if jobid in searches_infos:
                    info = searches_infos[jobid]
                    if 'end' not in info:
                        info['end'] = date

            if '/papers success addr:' in line:
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid:
                    continue
                if jobid in searches_infos:
                    info = searches_infos[jobid]
                    if 'papers' not in info:
                        info['papers'] = True

            if '/graph success citations addr:' in line:
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid:
                    continue
                if jobid in searches_infos:
                    info = searches_infos[jobid]
                    if 'graph_citations' not in info:
                        info['graph_citations'] = True

            if '/graph success structure addr:' in line:
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid:
                    continue
                if jobid in searches_infos:
                    info = searches_infos[jobid]
                    if 'graph_structure' not in info:
                        info['graph_structure'] = True

            if '/process paper analysis addr:' in line:
                paper_searches_dates.append(date)
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid:
                    continue
                papers_infos[jobid] = dict(jobid=jobid, start=date, args=args)
                if recent_papers.full():
                    recent_papers.get()  # Free some space
                recent_papers.put(jobid)

            if '/paper success addr:' in line:
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid:
                    continue
                if jobid in papers_infos:
                    info = papers_infos[jobid]
                    if 'end' not in info:
                        info['end'] = date
        except:
            pass  # Ignore single line error

    # Build timeseries graphs for terms and papers searches
    result = {}
    total_terms_searches = len(terms_searches_dates)
    result['total_terms_searches'] = total_terms_searches
    if total_terms_searches:
        p = prepare_timeseries(terms_searches_dates, 'Terms searches')
        result['terms_searches_plot'] = [components(p)]

    total_paper_searches = len(paper_searches_dates)
    result['total_paper_searches'] = total_paper_searches
    if total_paper_searches:
        p = prepare_timeseries(paper_searches_dates, 'Paper searches')
        result['paper_searches_plot'] = [components(p)]

    # Process recent searches
    recent_searches_results = []
    while not recent_searches.empty():
        jobid = recent_searches.get()
        info = searches_infos[jobid]
        date, args = info['start'], info['args']
        if 'end' in info:
            duration = duration_string(info['end'] - date)
        else:
            duration = '-'

        link = f'/result?{"&".join([f"{a}={v}" for a, v in args.items()])}'
        recent_searches_results.append((
            date.strftime('%Y-%m-%d %H:%M:%S'),
            args['query'],
            link,
            duration,
            '+' if 'papers' in info else '-',
            '+' if 'graph_citations' in info else '-',
            '+' if 'graph_structure' in info else '-',
        ))
    result['recent_searches'] = recent_searches_results[::-1]

    # Process recent papers
    recent_papers_results = []
    while not recent_papers.empty():
        jobid = recent_papers.get()
        info = papers_infos[jobid]
        date, args = info['start'], info['args']
        if 'end' in info:
            duration = duration_string(info['end'] - date)
        else:
            duration = '-'

        link = f'/paper?{"&".join([f"{a}={v}" for a, v in args.items()])}'
        recent_papers_results.append((
            date.strftime('%Y-%m-%d %H:%M:%S'),
            args['query'],
            link,
            duration,
        ))
    result['recent_papers'] = recent_papers_results[::-1]

    # Generate a word cloud
    text = ' '.join(terms).replace(',', ' ').replace('[^a-zA-Z0-9]+', ' ')
    if text:  # Check that string is not empty
        wc = WordCloud(collocations=False, width=PLOT_WIDTH - 40, height=WC_HEIGHT - 40,
                       background_color='white', max_font_size=100).generate(text)
        result['word_cloud'] = Plotter.word_cloud_prepare(wc)

    # Terms search statistics
    successful_searches = sum('end' in info for info in searches_infos.values())
    result['successful_searches'] = successful_searches
    duration = 0
    for info in searches_infos.values():
        if 'end' in info:
            duration += (info['end'] - info['start']).seconds
    if successful_searches > 0:
        result['searches_avg_duration'] = duration_string(
            datetime.timedelta(seconds=int(duration / successful_searches))
        )
    else:
        result['searches_avg_duration'] = 'N/A'
    result['searches_papers_shown'] = sum('papers' in info for info in searches_infos.values())
    result['searches_graph_citations_shown'] = sum('graph_citations' in info for info in searches_infos.values())
    result['searches_graph_structure_shown'] = sum('graph_structure' in info for info in searches_infos.values())

    # Papers search statistics
    successful_paper_searches = sum('end' in info for info in papers_infos.values())
    result['paper_successful_searches'] = successful_paper_searches
    duration = 0
    for info in papers_infos.values():
        if 'end' in info:
            duration += (info['end'] - info['start']).seconds
    if successful_paper_searches > 0:
        result['paper_searches_avg_duration'] = duration_string(
            datetime.timedelta(seconds=int(duration / successful_paper_searches))
        )
    else:
        result['paper_searches_avg_duration'] = 'N/A'

    result['recent'] = RECENT

    return result


def duration_string(dt):
    dt = dt.seconds
    h = divmod(dt, 86400 * 3600)  # hours
    m = divmod(h[1], 60)  # minutes
    s = m[1]  # seconds
    return f'{h[0]:02d}:{m[0]:02d}:{s:02d}'


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
