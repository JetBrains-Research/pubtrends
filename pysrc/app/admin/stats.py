import datetime
import json
import re
from typing import Dict, List

import pandas as pd
from bokeh.embed import components
from bokeh.models import HoverTool
from bokeh.plotting import figure
from wordcloud import WordCloud

from pysrc.app.messages import *
from pysrc.papers.plot.plot_preprocessor import PlotPreprocessor
from pysrc.papers.utils import trim_query

TOOLS = "hover,pan,tap,wheel_zoom,box_zoom,reset,save"
PLOT_WIDTH = 900
PLOT_HEIGHT = 300
WC_HEIGHT = 600
RECENT = 50

FEATURES = [LOG_PAPERS, LOG_GRAPH, LOG_QUESTION]


def feature_name(feature) -> str:
    return feature.replace('/', '').capitalize()


FEATURE_NAMES = [feature_name(feature) for feature in FEATURES]


def prepare_stats_data(logfile):
    return parse_stats_content(open(logfile).readlines())


def load_params(line):
    params = {}
    for t in ['args', 'form', 'json']:
        text = re.search(t + ':[ ]*\\{[^\\}]*\\}', line)
        if text:
            try:
                params.update(json.loads(text.group(0).replace(t + ':', '')))
            except Exception:
                pass
    return params


class SearchInfo:
    def __init__(self, jobid, start):
        self.jobid = jobid
        self.start = start
        self.args = None
        self.type = None
        self.end = None
        self.status = None
        self.papers_clicks = 0
        self.features = {}

    def duration(self):
        return self.end - self.start if self.end is not None else None


def parse_stats_content(lines, test=False):
    search_infos: Dict[str, SearchInfo] = dict()
    terms: List[str] = []

    for line in lines:
        try:
            if 'INFO' not in line and 'ERROR' not in line:
                continue
            search_date = re.search('[\\d-]+ [\\d:]+,\\d+', line)
            if search_date is None:
                continue
            date = datetime.datetime.strptime(search_date.group(0), '%Y-%m-%d %H:%M:%S,%f')

            # Both previous and updated versions of the app log search start/end events
            if (('/process regular search' in line
                 or '/process paper analysis' in line
                 or LOG_PROCESS in line)
                    and not (SUCCESS in line or ERROR in line or EXCEPTION in line)):
                params = load_params(line)
                if 'query' in params:
                    query = params['query']
                    if 'doi=' not in query and 'id=' not in query:
                        terms.append(query.replace(',', ' ').replace('[^a-zA-Z0-9]+', ' '))
                jobid = params['jobid']
                if not jobid:
                    continue
                if jobid not in search_infos:  # Don't re-add already existing task
                    search_infos[jobid] = SearchInfo(jobid, date)

            elif f'{LOG_STATUS} {ERROR}' in line or '/status failure' in line:
                params = load_params(line)
                jobid = params['jobid']
                if not jobid:
                    continue
                if jobid not in search_infos:
                    search_infos[jobid] = SearchInfo(jobid, date)
                info = search_infos[jobid]
                if 'search error' in line.lower():
                    info.status = 'Not found'
                else:
                    info.status = 'Error'
                if info.end is None:
                    info.end = date

            elif f'{LOG_RESULT} {SUCCESS}' in line:
                params = load_params(line)
                jobid = params['jobid']
                if not jobid:
                    continue
                if jobid not in search_infos:
                    search_infos[jobid] = SearchInfo(jobid, date)
                info = search_infos[jobid]
                info.type = 'terms'
                info.status = 'Ok'
                info.args = params
                if info.end is None:
                    info.end = date

            elif f'{LOG_PAPER} {SUCCESS}' in line:
                params = load_params(line)
                jobid = params['jobid']
                if not jobid or jobid not in search_infos:
                    continue
                info = search_infos[jobid]
                if info.type == 'terms':
                    info.papers_clicks += 1
                else:
                    info.type = 'paper'
                    info.status = 'Ok'
                    info.args = params
                    if info.end is None:
                        info.end = date
            else:
                for feature in FEATURES:
                    if f'{feature} {SUCCESS}' in line:
                        params = load_params(line)
                        jobid = params['jobid']
                        if not jobid or jobid not in search_infos:
                            continue
                        info = search_infos[jobid]
                        info.features[feature] = True
                        break

        except Exception as e:
            print(e)  # Ignore single line error

    result = {}
    search_terms = [info for info in search_infos.values() if info.type == 'terms']
    search_papers = [info for info in search_infos.values() if info.type == 'paper']
    result['terms_searches_total'] = len(search_terms)
    result['paper_searches_total'] = len(search_papers)

    # Build timeseries graphs for terms and papers searches
    if not test and search_terms:
        p = prepare_timeseries([info.start for info in search_terms], 'Terms searches')
        result['terms_searches_plot'] = [components(p)]

    if not test and search_papers:
        p = prepare_timeseries([info.start for info in search_papers], 'Paper searches')
        result['paper_searches_plot'] = [components(p)]

    # Terms search statistics
    terms_searches_successful = sum(info.status == 'Ok' for info in search_terms)
    result['terms_searches_successful'] = terms_searches_successful
    result['searches_papers_clicks'] = sum(info.papers_clicks for info in search_terms)
    result['searches_papers_list_shown'] = sum(LOG_PAPERS in info.features for info in search_terms)
    durations = []
    for info in search_terms:
        if info.end is not None and info.end != info.start:
            durations.append(info.duration().seconds)
    if len(durations) > 0:
        result['terms_searches_avg_duration'] = duration_string(
            datetime.timedelta(seconds=int(sum(durations) / len(durations)))
        )
    else:
        result['searches_avg_duration'] = 'N/A'

    result['features'] = FEATURE_NAMES
    result['feature_counts'] = {}
    for feature in FEATURES:
        result['feature_counts'][feature_name(feature)] = sum(feature in info.features for info in search_terms)

    # Papers search statistics
    paper_searches_successful = sum(info.status == 'Ok' for info in search_papers)
    result['paper_searches_successful'] = paper_searches_successful
    durations = []
    for info in search_papers:
        if info.end is not None and info.end != info.start:
            durations.append(info.duration().seconds)
    if len(durations) > 0:
        result['paper_searches_avg_duration'] = duration_string(
            datetime.timedelta(seconds=int(sum(durations) / len(durations)))
        )
    else:
        result['paper_searches_avg_duration'] = 'N/A'

    # Process recent searches
    result['recent'] = RECENT
    terms_searches_recent_results = []
    terms_searches_features_results = {fn: [] for fn in FEATURE_NAMES}
    for info in reversed(search_terms[-RECENT:]):
        if info.end is not None and info.end != info.start:
            duration = duration_string(info.duration())
        else:
            duration = '-'
        params = info.args
        link = f'/result?{"&".join([f"{a}={v}" for a, v in params.items()])}'
        terms_searches_recent_results.append(
            (
                info.start.strftime('%Y-%m-%d %H:%M:%S'),
                params['source'] if 'source' in params else '',
                trim_query(params['query']) if 'query' in params else '',
                link,
                duration,
                info.status or 'N/A',
                info.papers_clicks
            )
        )
        for f in FEATURES:
            terms_searches_features_results[feature_name(f)].append('+' if f in info.features else '-')
    result['terms_searches_recent'] = terms_searches_recent_results
    result['terms_searches_features_results'] = terms_searches_features_results

    # Process recent papers
    recent_paper_searchers_results = []
    for info in reversed(search_papers[-RECENT:]):
        if info.end is not None and info.end != info.start:
            duration = duration_string(info.duration())
        else:
            duration = '-'
        params = info.args
        if 'query' in params:
            title = params['query']
        elif 'id' in params:
            title = f'Id: {params["id"]}'
        else:
            title = 'N/A'

        link = f'/paper?{"&".join([f"{a}={v}" for a, v in params.items()])}'
        recent_paper_searchers_results.append(
            (
                info.start.strftime('%Y-%m-%d %H:%M:%S'),
                params['source'] if 'source' in params else '',
                trim_query(title),
                link,
                duration,
                info.status or 'N/A',
            )
        )
    result['paper_searches_recent'] = recent_paper_searchers_results

    # Generate a word cloud
    text = ' '.join(terms).replace(',', ' ').replace('[^a-zA-Z0-9]+', ' ')
    if text:  # Check that string is not empty
        if not test:
            wc = WordCloud(collocations=False, max_words=100,
                           width=PLOT_WIDTH, height=WC_HEIGHT, background_color='white', max_font_size=100).generate(text)
            result['word_cloud'] = PlotPreprocessor.word_cloud_prepare(wc)
        else:
            result['terms'] = terms

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
