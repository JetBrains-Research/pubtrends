import json
from dataclasses import dataclass, field

from pysrc.papers.db.article import Article


@dataclass
class SemanticScholarArticle(Article):
    ssid: str = field(default='')  # Artificial for Article override
    crc32id: int = field(default=0)  # Artificial for Article override
    year: int = 1970

    def to_dict(self):
        assert self.crc32id is not None
        return {
            'ssid': self.ssid,
            'crc32id': self.crc32id,
            'pmid': self.pmid,
            'title': self.title,
            'year': self.year,
            'abstract': self.abstract,
            'type': self.type,
            'keywords': self.keywords,
            'doi': self.doi,
            'aux': json.dumps({"journal": {"name": ""}, "authors": []})
        }
