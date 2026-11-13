import os
import shutil
import logging
from typing import List, Optional, Sequence

import whoosh.index as index
from whoosh import fields, qparser, scoring
from whoosh.analysis import StandardAnalyzer
from whoosh.query import Or, Term, NumericRange

from pysrc.papers.utils import SearchError, SORT_MOST_RECENT

logger = logging.getLogger(__name__)

# Field boost for title
TITLE_BOOST = 3.0


def create_schema() -> fields.Schema:
    analyzer = StandardAnalyzer(stoplist=None, minsize=1)
    return fields.Schema(
        id=fields.ID(stored=True, unique=True),
        title=fields.TEXT(stored=True, analyzer=analyzer, field_boost=TITLE_BOOST),
        abstract=fields.TEXT(stored=False, analyzer=analyzer),
        keywords=fields.TEXT(stored=False, analyzer=analyzer),
        mesh=fields.TEXT(stored=False, analyzer=analyzer),
        year=fields.NUMERIC(stored=True, sortable=True),
        type=fields.ID(stored=True),
        doi=fields.ID(stored=True),
        crc32id=fields.NUMERIC(stored=True),
    )


class LuceneIndexManager:
    """
    Manages creation, indexing, deletion, and searching of Whoosh (Lucene python analogue) index.
    """

    def __init__(self, index_path: str, bm25_k1: float = 1.2, bm25_b: float = 0.75):
        self.index_path = os.path.expanduser(index_path)
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.schema = create_schema()
        self._index = None

    def get_index(self, recreate: bool = False) -> index.Index:
        if recreate:
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path)
            os.makedirs(self.index_path, exist_ok=True)
            self._index = index.create_in(self.index_path, self.schema)
            return self._index

        if self._index is not None:
            return self._index

        if index.exists_in(self.index_path):
            self._index = index.open_dir(self.index_path)
        else:
            os.makedirs(self.index_path, exist_ok=True)
            self._index = index.create_in(self.index_path, self.schema)
        return self._index

    def reset(self):
        self.get_index(recreate=True)

    def insert_pubmed_articles(self, articles: Sequence):
        if not articles:
            return
        ix = self.get_index()
        writer = ix.writer()
        for a in articles:
            keywords_str = ' '.join(a.keywords) if a.keywords else ''
            mesh_str = ' '.join(a.mesh) if a.mesh else ''
            writer.update_document(
                id=str(a.pmid),
                title=str(a.title or ''),
                abstract=str(a.abstract or ''),
                keywords=keywords_str,
                mesh=mesh_str,
                year=int(a.year) if a.year is not None else None,
                type=str(a.type or ''),
                doi=str(a.doi or ''),
            )
        writer.commit()

    def insert_semantic_scholar_articles(self, articles: Sequence):
        if not articles:
            return
        ix = self.get_index()
        writer = ix.writer()
        for a in articles:
            writer.update_document(
                id=str(a.ssid),
                crc32id=int(a.crc32id) if a.crc32id is not None else None,
                title=str(a.title or ''),
                abstract=str(a.abstract or ''),
                keywords='',
                mesh='',
                year=int(a.year) if a.year is not None else None,
                type=str(getattr(a, 'type', '') or ''),
                doi=str(a.doi or ''),
            )
        writer.commit()

    def delete(self, ids: Sequence):
        if not ids:
            return
        ix = self.get_index()
        writer = ix.writer()
        for doc_id in ids:
            writer.delete_by_term('id', str(doc_id))
        writer.commit()

    def parse_query(self, query_str: str):
        if not query_str or not query_str.strip():
            raise SearchError("Illegal query: empty query")
        if query_str.count('"') % 2 != 0:
            raise SearchError(f"Illegal query: unclosed quotes in '{query_str}'")

        parser = qparser.MultifieldParser(
            ['title', 'abstract', 'keywords', 'mesh'],
            schema=self.schema,
            group=qparser.AndGroup
        )
        parser.add_plugin(qparser.PlusMinusPlugin())

        # Support comma-separated OR clauses (e.g., "COVID-19, Coronavirus")
        parts = [p.strip() for p in query_str.split(',') if p.strip()]
        if not parts:
            raise SearchError(f"Illegal query: '{query_str}'")

        try:
            if len(parts) == 1:
                return parser.parse(parts[0])
            else:
                return Or([parser.parse(p) for p in parts])
        except Exception as e:
            raise SearchError(f"Failed to parse query '{query_str}': {e}") from e

    def search(
        self,
        query_str: str,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
        noreviews: bool = True,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        query_expander=None,
    ) -> List[str]:
        q = self.parse_query(query_str)
        if query_expander is not None:
            q = query_expander.expand(query_str, q)
        ix = self.get_index()

        filter_query = None
        if min_year is not None or max_year is not None:
            filter_query = NumericRange('year', min_year, max_year)

        mask_query = None
        if noreviews:
            mask_query = Term('type', 'Review')

        weighting = scoring.BM25F(B=self.bm25_b, K1=self.bm25_k1)
        with ix.searcher(weighting=weighting) as searcher:
            kwargs = {
                'filter': filter_query,
                'mask': mask_query,
                'limit': limit,
            }
            if sort == SORT_MOST_RECENT:
                kwargs['sortedby'] = 'year'
                kwargs['reverse'] = True

            results = searcher.search(q, **kwargs)
            return [r['id'] for r in results]

    def search_title(self, title: str, limit: int = 100) -> List[str]:
        if not title or not title.strip():
            return []
        ix = self.get_index()
        parser = qparser.QueryParser('title', schema=self.schema)
        # Parse title as exact phrase first, or fallback to terms
        clean_title = title.replace('"', '').strip()
        try:
            q = parser.parse(f'"{clean_title}"')
        except Exception:
            q = parser.parse(clean_title)

        with ix.searcher() as searcher:
            results = searcher.search(q, limit=limit)
            return [r['id'] for r in results]
