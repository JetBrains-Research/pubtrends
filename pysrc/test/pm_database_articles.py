import json
from dataclasses import dataclass, field
from datetime import date
from typing import List

import pandas as pd

from pysrc.papers.loader import Loader


@dataclass
class Author:
    name: str = ''
    affiliation: List[str] = field(default_factory=lambda: [])

    def to_dict(self):
        return {'name': self.name, 'affiliation': self.affiliation}


@dataclass
class Journal:
    name: str = ''

    def to_dict(self):
        return {'name': self.name}


@dataclass
class AuxInfo:
    authors: List[Author] = field(default_factory=lambda: [])
    journal: Journal = field(default=Journal())

    def to_dict(self):
        return {'authors': [author.to_dict() for author in self.authors],
                'journal': self.journal.to_dict()}


@dataclass
class PubmedArticle:
    pmid: int
    title: str
    aux: AuxInfo = AuxInfo()
    abstract: str = None
    type: str = 'Article'
    doi: str = ''
    date: date = date(1970, 1, 1)

    def authors(self) -> str:
        return ', '.join([author.name for author in self.aux.authors])

    def journal(self) -> str:
        return self.aux.journal.name

    def __str__(self):
        return ', '.join(self.to_list())

    def to_dict(self):
        return {
            'pmid': self.pmid,
            'title': self.title,
            'date': self.date,
            'abstract': self.abstract,
            'type': self.type,
            'doi': self.doi,
            'aux': json.dumps(self.aux.to_dict())
        }

    def to_list(self):
        return [self.pmid, self.title, self.doi, json.dumps(self.aux.to_dict()),
                self.abstract if self.abstract else '',
                str(self.date), self.type, self.authors(), self.journal()]

    def to_list_year(self):
        return [self.pmid, self.title, self.doi, json.dumps(self.aux.to_dict()),
                self.abstract if self.abstract else '',
                int(self.date.year), self.type, self.authors(), self.journal()]


REQUIRED_ARTICLES = [
    PubmedArticle(1, 'Article Title 1', date=date(1963, 2, 1), doi='10.000/0000',
                  aux=AuxInfo(
                      authors=[Author(name='Geller R'), Author(name='Geller M'), Author(name='Bing Ch')],
                      journal=Journal(name='Nature'))),
    PubmedArticle(2, 'Article Title 2', abstract='Abstract', date=date(1965, 4, 10),
                  aux=AuxInfo(
                      authors=[Author(name='Buffay Ph'), Author(name='Geller M'), Author(name='Doe J')],
                      journal=Journal(name='Science'))),
    PubmedArticle(3, 'Article Title 3', abstract='Other Abstract', date=date(1967, 6, 21),
                  aux=AuxInfo(
                      authors=[Author(name='Doe J'), Author(name='Buffay Ph')],
                      journal=Journal(name='Nature'))),
    PubmedArticle(4, 'Article Title 4', date=date(1968, 1, 1),
                  aux=AuxInfo(
                      authors=[Author(name='Doe J'), Author(name='Geller R')],
                      journal=Journal(name='Science'))),
    PubmedArticle(5, 'Article Title 5', date=date(1975, 1, 1),
                  aux=AuxInfo(
                      authors=[Author(name='Green R'), Author(name='Geller R'), Author(name='Doe J')],
                      journal=Journal(name='Nature'))),
    PubmedArticle(6, 'Article Title 6', type='Review')
]

EXTRA_ARTICLES = [
    PubmedArticle(7, 'Article Title 7', date=date(1968, 1, 1)),
    PubmedArticle(8, 'Article Title 8', date=date(1969, 1, 1)),
    PubmedArticle(9, 'Article Title 9', date=date(1970, 1, 1)),
    PubmedArticle(10, 'Article Title 10', date=date(1970, 1, 1))
]

ARTICLES = REQUIRED_ARTICLES + EXTRA_ARTICLES

PART_OF_ARTICLES = [REQUIRED_ARTICLES[2], REQUIRED_ARTICLES[3]]

EXPANDED_IDS = ['2', '3', '4', '5', '7', '8', '9', '10']

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
    # ['5', 1970, 3] YEAR confusion!
]

COCITATIONS = [
    ['7', '1', '2', 1968], ['7', '1', '3', 1968], ['7', '2', '3', 1968],
    ['8', '1', '3', 1969], ['8', '1', '4', 1969], ['8', '3', '4', 1969],
    ['9', '4', '5', 1970], ['10', '4', '5', 1970]
]

EXPECTED_PUB_DF = Loader.process_publications_dataframe(
    pd.DataFrame([article.to_list_year() for article in REQUIRED_ARTICLES],
                 columns=['id', 'title', 'doi', 'aux', 'abstract', 'year', 'type', 'authors', 'journal']))

EXPECTED_CIT_STATS_DF = pd.DataFrame(CITATION_STATS, columns=['id', 'year', 'count']).sort_values(
    by=['id', 'year']
).reset_index(drop=True)

EXPECTED_CIT_DF = pd.DataFrame(INNER_CITATIONS, columns=['id_out', 'id_in'])

EXPECTED_COCIT_DF = pd.DataFrame(COCITATIONS, columns=['citing', 'cited_1', 'cited_2', 'year'])

EXPECTED_PUB_DF_GIVEN_IDS = Loader.process_publications_dataframe(
    pd.DataFrame([article.to_list_year() for article in PART_OF_ARTICLES],
                 columns=['id', 'title', 'doi', 'aux', 'abstract', 'year', 'type', 'authors', 'journal']))
