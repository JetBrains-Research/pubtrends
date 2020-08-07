import json
from dataclasses import dataclass


@dataclass
class SemanticScholarArticle:
    ssid: str
    crc32id: int
    title: str
    year: int = 1970
    type: str = 'Article'
    doi: str = ''

    def to_dict(self):
        assert self.crc32id is not None
        return {
            'id': self.ssid,
            'crc32id': self.crc32id,
            'pmid': None,
            'title': self.title,
            'year': self.year,
            'abstract': '',
            'type': self.type,
            'doi': self.doi,
            'aux': json.dumps({"journal": {"name": ""}, "authors": []})
        }
