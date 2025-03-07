import numpy as np
import pandas as pd

from pysrc.papers.db.article import AuxInfo, Author, Journal
from pysrc.papers.db.loader import Loader
from pysrc.papers.db.pm_article import PubmedArticle

REQUIRED_ARTICLES = [
    PubmedArticle(pmid=1, title='Article Title 1', year=1963, doi='10.000/0000',
                  aux=AuxInfo(
                      authors=[Author(name='Geller R'), Author(name='Geller M'), Author(name='Bing Ch')],
                      journal=Journal(name='Nature'))),
    PubmedArticle(2, 'Article Title 2', abstract='Abstract', year=1965,
                  aux=AuxInfo(
                      authors=[Author(name='Buffay Ph'), Author(name='Geller M'), Author(name='Doe J')],
                      journal=Journal(name='Science'))),
    PubmedArticle(3, 'Article Title 3', abstract='Other Abstract', year=1967,
                  aux=AuxInfo(
                      authors=[Author(name='Doe J'), Author(name='Buffay Ph')],
                      journal=Journal(name='Nature'))),
    PubmedArticle(4, 'Article Title 4', year=1968,
                  aux=AuxInfo(
                      authors=[Author(name='Doe J'), Author(name='Geller R')],
                      journal=Journal(name='Science'))),
    PubmedArticle(5, 'Article Title 5', year=1975,
                  aux=AuxInfo(
                      authors=[Author(name='Green R'), Author(name='Geller R'), Author(name='Doe J')],
                      journal=Journal(name='Nature'))),
    PubmedArticle(6, 'Article Title 6', type='Review')
]

EXTRA_ARTICLES = [
    PubmedArticle(7, 'Article Title 7', year=1968),
    PubmedArticle(8, 'Article Title 8', year=1969),
    PubmedArticle(9, 'Article Title 9', year=1970),
    PubmedArticle(10, 'Article Title 10', year=1970)
]

ARTICLES = REQUIRED_ARTICLES + EXTRA_ARTICLES
EXTRA_ARTICLE = PubmedArticle(
    pmid=100,
    title='activemodule dna',
    abstract='''
... physical activity module ... (inactive) ... activating                modules ... antipositive ... active ...
... methylation ...
''',
    year=2021,
    doi='10.1249/01.mss.0000229457.73333.9a,doi:10.1101/gad.918201 ',
    aux=AuxInfo(authors=[], journal=Journal(name=''))
)

PART_OF_ARTICLES = [REQUIRED_ARTICLES[2], REQUIRED_ARTICLES[3]]

EXPANDED_IDS_DF = pd.DataFrame([
    [10, 1], [2, 1], [7, 1], [8, 1], [9, 1], [5, 3]],
    columns=['id', 'total']
)
EXPANDED_IDS_DF['id'] = EXPANDED_IDS_DF['id'].astype(str)

EXPANDED_TOP_CITED_5_DF = pd.DataFrame([[10, 1], [2, 1], [8, 1], [9, 1], [5, 3]], columns=['id', 'total'])
EXPANDED_TOP_CITED_5_DF['id'] = EXPANDED_TOP_CITED_5_DF['id'].astype(str)

OUTER_CITATIONS = [
    ('7', '1'), ('7', '2'), ('7', '3'), ('8', '1'), ('8', '3'),
    ('8', '4'), ('9', '4'), ('9', '5'), ('10', '4'), ('10', '5')
]

INNER_CITATIONS = [
    ('2', '1'), ('3', '2'), ('4', '3'), ('5', '4'), ('6', '5')
]

CITATIONS = INNER_CITATIONS + OUTER_CITATIONS

CITATION_STATS = [
    ['1', 1965, 1], ['1', 1968, 1], ['1', 1969, 1],
    ['2', 1967, 1], ['2', 1968, 1],
    ['3', 1968, 2], ['3', 1969, 1],
    ['4', 1969, 1], ['4', 1970, 2], ['4', 1975, 1],
    ['5', 1970, 3]  # TODO: YEAR confusion!
]

COCITATIONS = [
    ['7', '1', '2', 1968], ['7', '1', '3', 1968], ['7', '2', '3', 1968],
    ['8', '1', '3', 1969], ['8', '1', '4', 1969], ['8', '3', '4', 1969],
    ['9', '4', '5', 1970], ['10', '4', '5', 1970]
]

random_order = np.arange(0, len(REQUIRED_ARTICLES))
np.random.shuffle(random_order)
EXPECTED_PUB_DF = Loader.process_publications_dataframe(
    [article.pmid for article in REQUIRED_ARTICLES],
    pd.DataFrame([REQUIRED_ARTICLES[i].to_list() for i in random_order],
                 columns=['id', 'title', 'doi', 'aux', 'abstract', 'year', 'type', 'authors', 'journal',
                          'keywords', 'mesh']))

EXPECTED_CIT_STATS_DF = pd.DataFrame(CITATION_STATS, columns=['id', 'year', 'count']).sort_values(
    by=['id', 'year']
).reset_index(drop=True)

EXPECTED_CIT_DF = pd.DataFrame(INNER_CITATIONS, columns=['id_out', 'id_in'])

EXPECTED_COCIT_DF = pd.DataFrame(COCITATIONS, columns=['citing', 'cited_1', 'cited_2', 'year'])

EXPECTED_PUB_DF_GIVEN_IDS = Loader.process_publications_dataframe(
    [article.pmid for article in PART_OF_ARTICLES],
    pd.DataFrame([article.to_list() for article in PART_OF_ARTICLES],
                 columns=['id', 'title', 'doi', 'aux', 'abstract', 'year', 'type', 'authors', 'journal',
                          'keywords', 'mesh']))
