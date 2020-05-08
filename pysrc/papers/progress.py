import logging
from io import StringIO


class Progress:

    def __init__(self, total):
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', "%Y-%m-%d %H:%M:%S"))
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(self.handler)
        self.total = total

    def info(self, message, current, task=None):
        if message:
            self.logger.info(message)

        if task:
            self.handler.flush()
            task.update_state(state='PROGRESS',
                              meta={'current': current, 'total': self.total, 'log': self.stream.getvalue()})

    def done(self, message='Done', task=None):
        self.info(message, self.total, task)

    def remove_handler(self):
        self.logger.removeHandler(self.handler)

    def log(self):
        return self.stream.getvalue()
