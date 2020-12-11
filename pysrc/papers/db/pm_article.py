import json
from dataclasses import dataclass, field
from typing import List

from pysrc.papers.db.article import Article


@dataclass
class PubmedArticle(Article):
    year: int = 1970
    mesh: List[str] = field(default_factory=lambda: [])
    type: str = 'Article'

    def to_dict(self):
        return {
            'pmid': self.pmid,
            'title': self.title,
            'year': self.year,
            'abstract': self.abstract or '',
            'type': self.type,
            'doi': self.doi or '',
            'keywords': ','.join(self.keywords),
            'mesh': ','.join(self.mesh),
            'aux': json.dumps(self.aux.to_dict())
        }

    def to_list(self):
        return [self.pmid, self.title, self.doi or '',
                json.dumps(self.aux.to_dict()),
                self.abstract or '',
                self.year, self.type, self.authors(), self.journal(),
                ','.join(self.keywords), ','.join(self.mesh)]
