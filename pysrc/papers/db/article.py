from dataclasses import dataclass, field
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
    language: str = field(default_factory=lambda: '')
    databanks: List[str] = field(default_factory=lambda: [])

    def to_dict(self):
        return {'authors': [author.to_dict() for author in self.authors],
                'journal': self.journal.to_dict(),
                'language': self.language,
                'databanks': self.databanks}


@dataclass
class Article:
    pmid: int or None
    title: str
    abstract: str or None = None
    doi: str = None
    keywords: List[str] = field(default_factory=lambda: [])
    aux: AuxInfo = AuxInfo()

    def authors(self) -> str:
        return ', '.join([author.name for author in self.aux.authors])

    def journal(self) -> str:
        return self.aux.journal.name
