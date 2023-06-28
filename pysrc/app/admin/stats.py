import datetime
import json
import re
from queue import Queue

import pandas as pd
from bokeh.embed import components
from bokeh.models import HoverTool
from bokeh.plotting import figure
from wordcloud import WordCloud

from pysrc.papers.plot.plot_preprocessor import PlotPreprocessor
from pysrc.papers.utils import trim, MAX_QUERY_LENGTH

TOOLS = "hover,pan,tap,wheel_zoom,box_zoom,reset,save"
PLOT_WIDTH = 900
PLOT_HEIGHT = 300
WC_HEIGHT = 600
RECENT = 50


def prepare_stats_data(logfile):
    return parse_stats_content(open(logfile).readlines())


def parse_stats_content(lines):
    terms_search_infos = dict()
    terms_search_dates = []
    terms_searches_recent = Queue(maxsize=RECENT)
    terms = []

    paper_search_infos = dict()
    paper_searches_recent = Queue(maxsize=RECENT)
    paper_search_dates = []

    for line in lines:
        try:
            if 'INFO' not in line:
                continue
            search_date = re.search('[\\d-]+ [\\d:]+,\\d+', line)
            if search_date is None:
                continue
            date = datetime.datetime.strptime(search_date.group(0), '%Y-%m-%d %H:%M:%S,%f')

            if '/process regular search addr:' in line:
                terms_search_dates.append(date)
                args = json.loads(re.sub(".*args:", "", line.strip()))
                terms.append(args['query'].replace(',', ' ').replace('[^a-zA-Z0-9]+', ' '))
                jobid = args['jobid']
                if not jobid:
                    continue
                if jobid not in terms_search_infos:  # Don't re-add already existing task
                    terms_search_infos[jobid] = dict(jobid=jobid, start=date, args=args)
                    if terms_searches_recent.full():
                        terms_searches_recent.get()  # Free some space
                    terms_searches_recent.put(jobid)

            if '/search_terms with fixed jobid addr:' in line:
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid or jobid:
                    continue
                terms_search_infos[jobid] = dict(jobid=jobid, start=date, args=args)
                if terms_searches_recent.full():
                    terms_searches_recent.get()  # Free some space
                terms_searches_recent.put(jobid)
                terms_search_dates.append(date)
                terms.append(args['query'].replace(',', ' ').replace('[^a-zA-Z0-9]+', ' '))

            if '/status failure' in line:
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid:
                    continue
                if jobid in terms_search_infos:
                    info = terms_search_infos[jobid]
                    info['status'] = 'Error'
                    if 'Search error: True' in line:
                        info['status'] = 'Not found'
                    if 'end' not in info:
                        info['end'] = date

            if '/result success addr:' in line:
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid:
                    continue
                if jobid in terms_search_infos:
                    info = terms_search_infos[jobid]
                    info['status'] = 'Ok'
                    if 'end' not in info:
                        info['end'] = date

            if '/papers success addr:' in line:
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid:
                    continue
                if jobid in terms_search_infos:
                    info = terms_search_infos[jobid]
                    if 'papers_list' not in info:
                        info['papers_list'] = True

            if '/graph success similarity addr:' in line:
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid:
                    continue
                if jobid in terms_search_infos:
                    info = terms_search_infos[jobid]
                    if 'graph' not in info:
                        info['graph'] = True

            if '/process review addr:' in line:
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid:
                    continue
                if jobid in terms_search_infos:
                    info = terms_search_infos[jobid]
                    if 'review' not in info:
                        info['review'] = True

            if '/process paper analysis addr:' in line:
                paper_search_dates.append(date)
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid:
                    continue
                paper_search_infos[jobid] = dict(jobid=jobid, start=date, args=args)
                if paper_searches_recent.full():
                    paper_searches_recent.get()  # Free some space
                paper_searches_recent.put(jobid)

            if '/paper success addr:' in line:
                args = json.loads(re.sub(".*args:", "", line.strip()))
                jobid = args['jobid']
                if not jobid:
                    continue
                if jobid in terms_search_infos:
                    info = terms_search_infos[jobid]
                    if 'papers_clicks' in info:
                        info['papers_clicks'] += 1
                    else:
                        info['papers_clicks'] = 1
                if jobid in paper_search_infos:
                    info = paper_search_infos[jobid]
                    info['status'] = 'Ok'
                    if 'end' not in info:
                        info['end'] = date
        except:
            pass  # Ignore single line error

    # Build timeseries graphs for terms and papers searches
    result = {}
    terms_searches_total = len(terms_search_dates)
    result['terms_searches_total'] = terms_searches_total
    if terms_searches_total:
        p = prepare_timeseries(terms_search_dates, 'Terms searches')
        result['terms_searches_plot'] = [components(p)]

    paper_searches_total = len(paper_search_dates)
    result['paper_searches_total'] = paper_searches_total
    if paper_searches_total:
        p = prepare_timeseries(paper_search_dates, 'Paper searches')
        result['paper_searches_plot'] = [components(p)]

    # Process recent searches
    terms_searches_recent_results = []
    while not terms_searches_recent.empty():
        jobid = terms_searches_recent.get()
        info = terms_search_infos[jobid]
        date, args = info['start'], info['args']
        if 'end' in info:
            duration = duration_string(info['end'] - date)
        else:
            duration = '-'

        link = f'/result?{"&".join([f"{a}={v}" for a, v in args.items()])}'
        terms_searches_recent_results.append((
            date.strftime('%Y-%m-%d %H:%M:%S'),
            args['source'] if 'source' in args else '',
            trim(args['query'], MAX_QUERY_LENGTH),
            link,
            duration,
            info['status'] if 'status' in info else 'N/A',
            info['papers_clicks'] if 'papers_clicks' in info else 0,
            '+' if 'papers_list' in info else '-',
            '+' if 'graph' in info else '-',
            '+' if 'review' in info else '-',
        ))
    result['terms_searches_recent'] = terms_searches_recent_results[::-1]

    # Process recent papers
    recent_paper_searchers_results = []
    while not paper_searches_recent.empty():
        jobid = paper_searches_recent.get()
        info = paper_search_infos[jobid]
        date, args = info['start'], info['args']
        if 'query' in args:
            title = args['query']
        elif 'id' in args:
            title = f'Id: {args["id"]}'
        else:
            title = 'N/A'
        if 'end' in info:
            duration = duration_string(info['end'] - date)
        else:
            duration = '-'

        link = f'/paper?{"&".join([f"{a}={v}" for a, v in args.items()])}'
        recent_paper_searchers_results.append((
            date.strftime('%Y-%m-%d %H:%M:%S'),
            args['source'] if 'source' in args else '',
            trim(title, MAX_QUERY_LENGTH),
            link,
            duration,
            info['status'] if 'status' in info else 'N/A',
        ))
    result['paper_searches_recent'] = recent_paper_searchers_results[::-1]

    # Generate a word cloud
    text = ' '.join(terms).replace(',', ' ').replace('[^a-zA-Z0-9]+', ' ')
    if text:  # Check that string is not empty
        wc = WordCloud(collocations=False, max_words=100,
                       width=PLOT_WIDTH, height=WC_HEIGHT, background_color='white', max_font_size=100).generate(text)
        result['word_cloud'] = PlotPreprocessor.word_cloud_prepare(wc)

    # Terms search statistics
    terms_searches_successful = sum('end' in info for info in terms_search_infos.values())
    result['terms_searches_successful'] = terms_searches_successful
    duration = 0
    for info in terms_search_infos.values():
        if 'end' in info:
            duration += (info['end'] - info['start']).seconds
    if terms_searches_successful > 0:
        result['terms_searches_avg_duration'] = duration_string(
            datetime.timedelta(seconds=int(duration / terms_searches_successful))
        )
    else:
        result['searches_avg_duration'] = 'N/A'
    result['searches_papers_clicks'] = sum('papers_clicks' in info for info in terms_search_infos.values())
    result['searches_papers_list_shown'] = sum('papers_list' in info for info in terms_search_infos.values())
    result['searches_graph_shown'] = sum('graph' in info for info in terms_search_infos.values())
    result['searches_review_shown'] = sum('review' in info for info in terms_search_infos.values())

    # Papers search statistics
    paper_searches_successful = sum('end' in info for info in paper_search_infos.values())
    result['paper_searches_successful'] = paper_searches_successful
    duration = 0
    for info in paper_search_infos.values():
        if 'end' in info:
            duration += (info['end'] - info['start']).seconds
    if paper_searches_successful > 0:
        result['paper_searches_avg_duration'] = duration_string(
            datetime.timedelta(seconds=int(duration / paper_searches_successful))
        )
    else:
        result['paper_searches_avg_duration'] = 'N/A'

    result['recent'] = RECENT

    return result


def duration_string(dt):
    return duration_seconds(dt.seconds)


def duration_seconds(seconds):
    s = seconds % 60
    m = int(seconds / 60) % 60
    h = int(seconds / 3600)
    return f'{h:02d}:{m:02d}:{s:02d}'


def prepare_timeseries(dates, title):
    df_terms_searches = pd.DataFrame({'date': dates, 'count': 1}, dtype=object)
    df_terms_searches_grouped = df_terms_searches.groupby(pd.Grouper(key='date', freq='D')).sum()
    p = figure(width=PLOT_WIDTH, height=PLOT_HEIGHT, tools=TOOLS,
               x_axis_type='datetime', title=title)
    p.line('date', 'count', source=df_terms_searches_grouped)
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [('Date', '@date{%F}'), ('Count', '@count')]
    hover.formatters = {'@date': 'datetime'}
    p.sizing_mode = 'stretch_width'
    return p
