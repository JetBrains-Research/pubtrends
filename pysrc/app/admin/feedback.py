import datetime
import json
import re
from collections import Counter
from queue import Queue

RECENT = 50


def prepare_feedback_data(logfile):
    return parse_feedback_content(open(logfile).readlines())


def parse_feedback_content(lines):
    emotions_counter = Counter()
    recent_emotions = Queue(maxsize=RECENT)
    queries = dict()
    for line in lines:
        if 'INFO' not in line:
            continue
        search_date = re.search('[\\d-]+ [\\d:]+,\\d+', line)
        if search_date is None:
            continue
        date = datetime.datetime.strptime(search_date.group(0), '%Y-%m-%d %H:%M:%S,%f')
        if '/process regular search addr:' in line:
            args = json.loads(re.sub(".*args:", "", line.strip()))
            jobid = args['jobid']
            if not jobid:
                continue
            queries[jobid] = args['query'] if 'query' in args else 'NA'
        if '/process paper analysis addr:' in line:
            args = json.loads(re.sub(".*args:", "", line.strip()))
            jobid = args['jobid']
            if not jobid:
                continue
            queries[jobid] = args['query'] if 'query' in args else 'NA'
        if 'Feedback ' not in line:
            continue
        try:
            fb = json.loads(line.partition('Feedback ')[-1].strip())
            em_key, em_val = fb['key'].replace('feedback-', ''), int(fb['value'])
            if em_key.startswith('cancel:'):
                em_key = em_key.replace('cancel:', '')
                em_val *= -1
                delta = -1
            else:
                delta = 1
            if em_key not in emotions_counter:
                if delta == 1:
                    emotions_counter[em_key] = (em_val, delta)
            else:
                old_sum, old_count = emotions_counter[em_key]
                emotions_counter[em_key] = (old_sum + em_val, old_count + delta)

            if 'jobid' in fb and fb['jobid'] in queries:
                if recent_emotions.full():
                    recent_emotions.get()  # Free some space
                if em_val == 1:
                    em_str = 'Yes'
                elif em_val == -1:
                    em_str = 'No'
                else:
                    em_str = 'Meh'
                recent_emotions.put((date.strftime('%Y-%m-%d %H:%M:%S'),
                                    em_key, em_str, queries[fb['jobid']]))
        except:
            pass

    emotions = []
    while not recent_emotions.empty():
        emotions.append(recent_emotions.get())

    emotions_summary = [(key, int(100 * value[0] / value[1]) / 100, value[1])
                        for key, value in emotions_counter.items() if value[1] != 0]
    return dict(
        recent=RECENT,
        emotions=list(reversed(emotions)),
        emotions_summary=emotions_summary
    )
