import logging
from io import StringIO


class ProgressLogger:
    def __init__(self, total):
        self.stream = StringIO()
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(message)s', "%H:%M:%S")
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(formatter)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(self.handler)
        self.total = total

    def info(self, message, current, state='PROGRESS', task=None):
        if message:
            self.logger.info(message)

        self.update_state(current, state=state, task=task)

    def debug(self, message, current, state='PROGRESS', task=None):
        if message:
            self.logger.debug(message)

        self.update_state(current, state=state, task=task)

    def update_state(self, current, state='PROGRESS', task=None):
        if task:
            self.handler.flush()
            task.update_state(state=state,
                              meta={'current': current, 'total': self.total, 'log': self.stream.getvalue()})

    def remove_handler(self):
        self.logger.removeHandler(self.handler)
