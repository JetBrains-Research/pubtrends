from dataclasses import dataclass

import pandas as pd


@dataclass
class Article:
    id: int
    comp: int
    authors: str
    journal: str

    def to_dict(self):
        return {
            'id': self.id,
            'comp': self.comp,
            'authors': self.authors,
            'journal': self.journal
        }


article1 = Article(id=1, comp=0, authors='Geller R, Geller M, Bing Ch', journal='Nature')

article2 = Article(id=2, comp=1, authors='Buffay Ph, Geller M, Doe J', journal='Science')

article3 = Article(id=3, comp=1, authors='Doe J, Buffay Ph', journal='Nature')

article4 = Article(id=4, comp=1, authors='Doe J, Geller R', journal='Science')

article5 = Article(id=5, comp=0, authors='Green R, Geller R, Doe J', journal='Nature')

articles = [article1, article2, article3, article4, article5]

df_authors_and_journals = pd.DataFrame.from_records([article.to_dict() for article in articles])

author_stats = [['Doe J', [1, 0], [3, 1], 4],
                ['Geller R', [0, 1], [2, 1], 3],
                ['Buffay Ph', [1], [2], 2],
                ['Geller M', [0, 1], [1, 1], 2],
                ['Bing Ch', [0], [1], 1],
                ['Green R', [0], [1], 1]]

author_df = pd.DataFrame(author_stats, columns=['author', 'comp', 'counts', 'sum'])

journal_stats = [['Nature', [0, 1], [2, 1], 3],
                 ['Science', [1], [2], 2]]

journal_df = pd.DataFrame(journal_stats, columns=['journal', 'comp', 'counts', 'sum'])
