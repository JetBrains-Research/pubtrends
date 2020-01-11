import pandas as pd

from models.keypaper.config import PubtrendsConfig
from models.keypaper.loader import Loader

PUBLICATION_DATA = [['1', 1963, 'Article 1', None, 'Article', 'Geller R, Geller M, Bing Ch', 'Nature'],
                    ['2', 1965, 'Article 2', 'abstract', 'Article', 'Buffay Ph, Geller M, Doe J', 'Science'],
                    ['3', 1967, 'Article 3', 'otherabstract', 'Article', 'Doe J, Buffay Ph', 'Nature'],
                    ['4', 1968, 'Article 4', None, 'Article', 'Doe J, Geller R', 'Science'],
                    ['5', 1975, 'Article 5', None, 'Review', 'Green R, Geller R, Doe J', 'Nature']]

CITATION_STATS_DATA = [['1', 1972, 2], ['1', 1974, 15],
                       ['2', 1974, 1],
                       ['3', 1972, 5], ['3', 1974, 13],
                       ['4', 1972, 1], ['4', 1974, 8]]
CITATION_YEARS = [1972, 1974]

CITATION_DATA = [['4', '1'], ['3', '1'], ['4', '2'], ['3', '2'],
                 ['5', '1'], ['5', '2'], ['5', '3'], ['5', '4']]

CITATION_GRAPH_NODES = ['1', '2', '3', '4', '5']
CITATION_GRAPH_EDGES = [(e[0], e[1]) for e in CITATION_DATA]

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

    def find(self, key, value, current=0, task=None):
        raise Exception('Not implemented')

    def expand(self, ids, current=0, task=None):
        raise Exception('Not implemented')

    def search(self, terms, limit=None, sort=None, current=None, task=None):
        return [1, 2, 3, 4, 5], True

    def load_publications(self, ids=None, temp_table_created=True, current=None, task=None):
        return pd.DataFrame(PUBLICATION_DATA, columns=['id', 'year', 'title', 'abstract', 'type', 'authors', 'journal'])

    def load_citation_stats(self, ids=None, current=None, task=None):
        return pd.DataFrame(CITATION_STATS_DATA, columns=['id', 'year', 'count'])

    def load_citations(self, ids=None, current=None, task=None):
        return pd.DataFrame(CITATION_DATA, columns=['id_out', 'id_in'])

    def load_cocitations(self, ids=None, current=None, task=None):
        return pd.DataFrame(COCITATION_DATA, columns=['citing', 'cited_1', 'cited_2', 'year'])


class MockLoaderSingle(Loader):

    def __init__(self):
        config = PubtrendsConfig(test=True)
        super(MockLoaderSingle, self).__init__(config, connect=False)

    def find(self, key, value, current=0, task=None):
        raise Exception('Not implemented')

    def expand(self, ids, current=0, task=None):
        raise Exception('Not implemented')

    def search(self, terms, limit=None, sort=None, current=None, task=None):
        return [1, 2], True

    def load_publications(self, ids=None, temp_table_created=True, current=None, task=None):
        return pd.DataFrame(PUBLICATION_DATA[0:1],
                            columns=['id', 'year', 'title', 'abstract', 'type', 'authors', 'journal'])

    def load_citation_stats(self, ids=None, current=None, task=None):
        return pd.DataFrame([['1', 1972, 2], ['1', 1974, 15]],
                            columns=['id', 'year', 'count'])

    def load_citations(self, ids=None, current=None, task=None):
        return pd.DataFrame([], columns=['id_out', 'id_in'])

    def load_cocitations(self, ids=None, current=None, task=None):
        return pd.DataFrame([], columns=['citing', 'cited_1', 'cited_2', 'year'])


class MockLoaderEmpty(Loader):

    def __init__(self):
        config = PubtrendsConfig(test=True)
        super(MockLoaderEmpty, self).__init__(config, connect=False)

    def search(self, terms, limit=None, sort=None, current=None, task=None):
        return [], True

    def find(self, key, value, current=0, task=None):
        raise Exception('Not implemented')

    def load_publications(self, ids, current=0, task=None):
        raise Exception('Not implemented')

    def load_citation_stats(self, ids, current=0, task=None):
        raise Exception('Not implemented')

    def load_citations(self, ids, current=0, task=None):
        raise Exception('Not implemented')

    def load_cocitations(self, ids, current=0, task=None):
        raise Exception('Not implemented')

    def expand(self, ids, current=0, task=None):
        raise Exception('Not implemented')
