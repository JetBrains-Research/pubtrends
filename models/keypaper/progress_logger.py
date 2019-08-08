import logging
from io import StringIO


class ProgressLogger:
    TOTAL = 17

    def __init__(self):
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(self.handler)

    def info(self, message, current, total=TOTAL, state='PROGRESS', task=None):
        if message:
            self.logger.info(message)

        self.update_state(current, total=total, state=state, task=task)

    def debug(self, message, current, total=TOTAL, state='PROGRESS', task=None):
        if message:
            self.logger.debug(message)

        self.update_state(current, total=total, state=state, task=task)

    def update_state(self, current, total=TOTAL, state='PROGRESS', task=None):
        if task:
            self.handler.flush()
            task.update_state(state=state, meta={'current': current, 'total': total, 'log': self.stream.getvalue()})

    def remove_handler(self):
        self.logger.removeHandler(self.handler)
