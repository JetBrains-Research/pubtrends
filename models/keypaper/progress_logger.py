import logging
from io import StringIO


class ProgressLogger():

    def __init__(self):
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(self.handler)

    def info(self, message, current, total=10, state='PROGRESS', task=None):
        if message:
            self.logger.info(message)

        if task:
            self.handler.flush()
            task.update_state(state=state, meta={'current': current, 'total': total, 'log': self.stream.getvalue()})

    def update_state(self, current, total=10, state='PROGRESS', task=None):
        if task:
            task.update_state(state=state, meta={'current': current, 'total': total, 'log': self.stream.getvalue()})

    def remove_handler(self):
        self.logger.removeHandler(self.handler)
