import json
import re
from collections import Counter
from queue import Queue

RECENT = 50


def prepare_feedback_data(logfile):
    emotions_counter = Counter()
    recent_messages = Queue(maxsize=RECENT)
    recent_emotions = Queue(maxsize=RECENT)
    queries = dict()

    for line in open(logfile).readlines():
        if 'INFO' not in line:
            continue
        if '/process regular search addr:' in line:
            args = json.loads(re.sub(".*args:", "", line.strip()))
            jobid = args['jobid']
            if not jobid:
                continue
            queries[jobid] = args['query']
        if '/process paper analysis addr:' in line:
            args = json.loads(re.sub(".*args:", "", line.strip()))
            jobid = args['jobid']
            if not jobid:
                continue
            queries[jobid] = args['query']
        if 'Feedback ' not in line:
            continue
        try:
            fb = json.loads(line.partition('Feedback ')[-1].strip())
            if 'type' in fb:  # Messages
                if recent_messages.full():
                    recent_messages.get()  # Free some space
                recent_messages.put((fb['type'], fb['message'], fb['email']))
            else:  # Emotions
                val = fb['value']
                if val == '1':
                    val_str = 'Yes'
                elif val == '-1':
                    val_str = 'No'
                else:
                    val_str = 'Meh'
                emotions_counter[(fb['key'], val_str)] += 1
                if 'jobid' in fb and fb['jobid'] in queries:
                    if recent_emotions.full():
                        recent_emotions.get()  # Free some space
                    recent_emotions.put((fb['key'], val_str, queries[fb['jobid']]))
        except:
            pass

    messages = []
    while not recent_messages.empty():
        messages.append(recent_messages.get())

    emotions = []
    while not recent_emotions.empty():
        emotions.append(recent_emotions.get())

    emotions_summary = [(v[0], v[1], c) for v, c in emotions_counter.most_common()]
    return dict(
        recent=RECENT,
        messages=reversed(messages),
        emotions=reversed(emotions),
        emotions_summary=emotions_summary
    )
