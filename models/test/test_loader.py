from models.keypaper.config import PubtrendsConfig
from models.keypaper.loader import Loader


class TestLoader(Loader):
    def __init__(self):
        config = PubtrendsConfig(test=True)
        super(TestLoader, self).__init__(config, connect=False)
