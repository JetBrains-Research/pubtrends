import pandas as pd

from models.keypaper.config import PubtrendsConfig
from models.keypaper.loader import Loader

PUBLICATION_DATA = [['1', 1963, 'Article 1', None,            ''],
                    ['2', 1965, 'Article 2', 'abstract',      ''],
                    ['3', 1967, 'Article 3', 'otherabstract', ''],
                    ['4', 1968, 'Article 4', None,            ''],
                    ['5', 1975, 'Article 5', None,            '']]

CITATION_STATS_DATA = [['1', 0, 2, 15, 17],
                       ['2', 0, 0, 1, 1],
                       ['3', 0, 5, 13, 18],
                       ['4', 0, 1, 8, 9],
                       ['5', 0, 0, 5, 5]]
CITATION_YEARS = [1970, 1972, 1974]

CITATION_DATA = [['4', '1'], ['3', '1'], ['4', '2'], ['3', '2'],
                 ['5', '1'], ['5', '2'], ['5', '3'], ['5', '4']]

COCITATION_DATA = [['3', '1', '2', 1967],
                   ['4', '1', '2', 1968],
                   ['5', '1', '2', 1975],
                   ['5', '3', '4', 1975]]

COCITATION_GROUPED_DATA = [['1', '2', 1, 1, 1, 3],
                           ['3', '4', 0, 0, 1, 1]]
COCITATION_YEARS = [1967, 1968, 1969]

COCITATION_GRAPH_NODES = ['1', '2', '3', '4']
COCITATION_GRAPH_EDGES = [('1', '2', 3), ('3', '4', 1)]

EXPECTED_MAX_GAIN = {1972: '3', 1974: '1'}
EXPECTED_MAX_RELATIVE_GAIN = {1972: '3', 1974: '4'}


class MockLoader(Loader):

    def __init__(self):
        config = PubtrendsConfig(test=True)
        super(MockLoader, self).__init__(config, connect=False)

    def search(self, current=None, task=None):
        self.ids = [1, 2, 3, 4, 5]

    def load_publications(self, current=None, task=None):
        self.pub_df = pd.DataFrame(PUBLICATION_DATA, columns=['id', 'year', 'title', 'abstract', 'authors'])

    def load_citation_stats(self, current=None, task=None):
        self.cit_df = pd.DataFrame(CITATION_STATS_DATA, columns=['id', *CITATION_YEARS, 'total'])

    def load_cocitations(self, current=None, task=None):
        self.cocit_df = pd.DataFrame(COCITATION_DATA, columns=['citing', 'cited_1', 'cited_2', 'year'])
        self.cocit_grouped_df = pd.DataFrame(COCITATION_GROUPED_DATA,
                                             columns=['cited_1', 'cited_2', *COCITATION_YEARS, 'total'])
