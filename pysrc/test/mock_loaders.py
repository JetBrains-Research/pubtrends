import pandas as pd

from pysrc.papers.db.loader import Loader

PUBLICATION_DATA = [
    ['1', 1963, 'Article 1', 'just a paper', 'Article', 'Geller R, Geller M, Bing Ch', 'Nature',
     'term1,term2,term3', 'kw1,kw2'],
    ['2', 1965, 'Article 2', 'abstract', 'Article', 'Buffay Ph, Geller M, Doe J', 'Science',
     'term2,term3,term4', 'kw2,kw3'],
    ['3', 1967, 'Article 3', 'other abstract', 'Article', 'Doe J, Buffay Ph', 'Nature',
     'term3,term4,term5', 'kw3,kw4'],
    ['4', 1968, 'Article 4', 'interesting paper', 'Article', 'Doe J, Geller R', 'Science',
     'term4,term5,term1', 'kw4,kw5'],
    ['5', 1975, 'Article 5', 'just a breakthrough', 'Review', 'Green R, Geller R, Doe J', 'Nature',
     'term5,term1,term2', 'kw5,kw1']
]

CITATION_STATS_DATA = [['1', 1972, 2], ['1', 1974, 15],
                       ['2', 1974, 1],
                       ['3', 1972, 5], ['3', 1974, 13],
                       ['4', 1972, 1], ['4', 1974, 8]]
CITATION_DATA = [['4', '1'], ['3', '1'], ['4', '2'], ['3', '2'],
                 ['5', '1'], ['5', '2'], ['5', '3'], ['5', '4']]

COCITATION_DATA = [['3', '1', '2', 1967],
                   ['4', '1', '2', 1968],
                   ['5', '1', '2', 1975],
                   ['5', '3', '4', 1975]]

COCITATION_GROUPED_DATA = [['1', '2', 1, 1, 1, 3],
                           ['3', '4', 0, 0, 1, 1]]
COCITATION_YEARS = [1967, 1968, 1969]

COCITATION_DF = pd.DataFrame([['1', '2', 3], ['3', '4', 1]], columns=['cited_1', 'cited_2', 'total'])
COCITATION_GRAPH_EDGES = [('1', '2', 3), ('3', '4', 1)]

BIBLIOGRAPHIC_COUPLING_DATA = [['3', '4', 2],
                               ['3', '5', 2],
                               ['4', '5', 2]]
BIBCOUPLING_DF = pd.DataFrame(BIBLIOGRAPHIC_COUPLING_DATA, columns=['citing_1', 'citing_2', 'total'])

PAPERS_GRAPH_EDGES = [('1', '2', {'cocitation': 3}), ('1', '4', {'citation': 1}), ('1', '3', {'citation': 1}),
                      ('1', '5', {'citation': 1}), ('2', '4', {'citation': 1}), ('2', '3', {'citation': 1}),
                      ('2', '5', {'citation': 1}), ('3', '4', {'cocitation': 1, 'bibcoupling': 2}),
                      ('3', '5', {'bibcoupling': 2, 'citation': 1}), ('4', '5', {'bibcoupling': 2, 'citation': 1})]
EXPECTED_MAX_GAIN = {1972: '3', 1974: '1'}
EXPECTED_MAX_RELATIVE_GAIN = {1972: '3', 1974: '4'}


class MockLoader(Loader):

    def last_update(self):
        return None

    def find(self, key, value):
        raise Exception('Not implemented')

    def expand(self, ids, limit):
        raise Exception('Not implemented')

    def search(self, terms, limit=None, sort=None, noreviews=True):
        return ['1', '2', '3', '4', '5']

    def load_publications(self, ids=None):
        return pd.DataFrame(
            PUBLICATION_DATA,
            columns=['id', 'year', 'title', 'abstract', 'type', 'authors', 'journal', 'mesh', 'keywords']
        )

    def load_citations_by_year(self, ids=None):
        return pd.DataFrame(CITATION_STATS_DATA, columns=['id', 'year', 'count'])

    def load_references(self, pid, limit):
        raise Exception('Not implemented')

    def load_citations_counts(self, ids):
        raise Exception('Not implemented')

    def load_citations(self, ids=None):
        return pd.DataFrame(CITATION_DATA, columns=['id_out', 'id_in'])

    def load_cocitations(self, ids=None):
        return pd.DataFrame(COCITATION_DATA, columns=['citing', 'cited_1', 'cited_2', 'year'])

    def load_bibliographic_coupling(self, ids=None):
        return pd.DataFrame(BIBLIOGRAPHIC_COUPLING_DATA, columns=['citing_1', 'citing_2', 'total'])


class MockLoaderSingle(Loader):

    def last_update(self):
        return None

    def find(self, key, value):
        raise Exception('Not implemented')

    def expand(self, ids, limit):
        raise Exception('Not implemented')

    def search(self, terms, limit=None, sort=None, noreviews=True):
        return ['1']

    def load_publications(self, ids=None):
        return pd.DataFrame(
            PUBLICATION_DATA[0:1],
            columns=['id', 'year', 'title', 'abstract', 'type', 'authors', 'journal', 'mesh', 'keywords']
        )

    def load_citations_by_year(self, ids=None):
        return pd.DataFrame([['1', 1972, 2], ['1', 1974, 15]],
                            columns=['id', 'year', 'count'])

    def load_references(self, pid, limit):
        raise Exception('Not implemented')

    def load_citations_counts(self, ids):
        raise Exception('Not implemented')

    def load_citations(self, ids=None):
        return pd.DataFrame(columns=['id_out', 'id_in'], dtype=object)

    def load_cocitations(self, ids=None):
        df = pd.DataFrame(columns=['citing', 'cited_1', 'cited_2', 'year'], dtype=object)
        df['year'] = df['year'].astype(int)
        return df

    def load_bibliographic_coupling(self, ids=None):
        df = pd.DataFrame(columns=['citing_1', 'citing_2', 'total'], dtype=object)
        df['total'] = df['total'].astype(int)
        return df


class MockLoaderEmpty(Loader):

    def last_update(self):
        return None

    def search(self, terms, limit=None, sort=None, noreviews=True):
        return []

    def find(self, key, value):
        raise Exception('Not implemented')

    def load_publications(self, ids):
        raise Exception('Not implemented')

    def load_citations_by_year(self, ids):
        raise Exception('Not implemented')

    def load_references(self, pid, limit):
        raise Exception('Not implemented')

    def load_citations_counts(self, ids):
        raise Exception('Not implemented')

    def load_citations(self, ids):
        raise Exception('Not implemented')

    def load_cocitations(self, ids):
        raise Exception('Not implemented')

    def load_bibliographic_coupling(self, ids=None):
        raise Exception('Not implemented')

    def expand(self, ids, limit):
        raise Exception('Not implemented')
