import json
from collections import Counter
from queue import Queue

RECENT = 50


def prepare_feedback_data(logfile):
    emotions_counter = Counter()
    recent_messages = Queue(maxsize=RECENT)

    for line in open(logfile).readlines():
        if 'INFO' not in line or 'Feedback ' not in line:
            continue
        try:
            fb = json.loads(line.partition('Feedback ')[-1].strip())
            # Ignore jobid for now
            fb = {k: v for k, v in fb.items() if k != 'jobid'}
            if 'type' in fb:
                if recent_messages.full():
                    recent_messages.get()  # Free some space
                recent_messages.put(fb)
            else:
                emotions_counter[(fb['key'], fb['value'])] += 1
        except:
            pass

    messages = []
    while not recent_messages.empty():
        fb = recent_messages.get()
        messages.append((fb['type'], fb['message'], fb['email']))
    emotions = [(v[0], v[1], c) for v, c in emotions_counter.most_common()]
    return dict(recent=RECENT, messages=reversed(messages), emotions=emotions)
