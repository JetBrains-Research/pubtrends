import json
from dataclasses import dataclass, field
from datetime import date
from typing import List


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
