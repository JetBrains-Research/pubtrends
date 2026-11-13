import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from whoosh.analysis import StandardAnalyzer
from whoosh.query import And, Or, Phrase, Term

logger = logging.getLogger(__name__)

MESH_MAX_TERM_LENGTH = 96
MESH_MAX_TERM_WORDS = 6
MESH_MAX_PHRASE_WORDS = 6
MESH_MAX_CONCEPTS_EXPANDED = 1
MESH_MAX_DESCENDANTS_PER_CONCEPT = 20
MESH_MAX_TERMS_PER_CONCEPT = 16
MAX_QUERY_WORDS = 32

SEARCH_FIELDS = ("title", "abstract", "keywords", "mesh")
SIMPLE_QUERY = re.compile(r"[^\W_]+(?:\s+[^\W_]+)*\Z", re.UNICODE)
ANALYZER = StandardAnalyzer(stoplist=None, minsize=1)


@dataclass(frozen=True)
class MeshRecord:
    ui: str
    heading: str
    entry_terms: list[str]
    tree_numbers: list[str]


def _clean_term(value: str) -> Optional[str]:
    # Entry terms include pipe-delimited semantic type and provenance metadata.
    term = value.split("|", 1)[0].strip().strip(",").strip()
    if not term or len(term) > MESH_MAX_TERM_LENGTH or len(term.split()) > MESH_MAX_TERM_WORDS:
        return None
    return term


def _expand_terms(values: Iterable[str]) -> list[str]:
    terms = []
    seen = set()
    for value in values:
        term = _clean_term(value)
        if not term:
            continue
        variants = [term]
        comma = term.find(",")
        if comma > 0 and term.find(",", comma + 1) < 0:
            inverted = _clean_term(term[comma + 1:] + " " + term[:comma])
            if inverted:
                variants.append(inverted)
        for variant in variants:
            key = variant.casefold()
            if key not in seen:
                seen.add(key)
                terms.append(variant)
    return terms


def parse_mesh_ascii(lines: Iterable[str]):
    """Yield descriptor and supplementary records from an NLM ASCII MeSH release."""
    ui = None
    heading = None
    entry_terms = []
    tree_numbers = []

    def emit():
        clean_heading = _clean_term(heading) if heading else None
        if not ui or not clean_heading:
            return None
        terms = _expand_terms([heading] + entry_terms)
        return MeshRecord(
            ui=ui,
            heading=clean_heading,
            entry_terms=[term for term in terms if term.casefold() != clean_heading.casefold()],
            tree_numbers=list(tree_numbers),
        )

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if line.startswith("*NEWRECORD"):
            record = emit()
            if record:
                yield record
            ui = None
            heading = None
            entry_terms = []
            tree_numbers = []
            continue

        separator = line.find(" = ")
        if separator < 0:
            continue
        tag, value = line[:separator], line[separator + 3:]
        if tag == "UI":
            ui = value.strip()
        elif tag in ("MH", "NM"):
            heading = value
        elif tag in ("ENTRY", "PRINT ENTRY", "SY"):
            entry_terms.append(value)
        elif tag == "MN":
            tree_numbers.append(value.strip())

    record = emit()
    if record:
        yield record


class MeshTermExpander:
    """Adds bounded MeSH alternatives to simple positive Whoosh queries."""

    def __init__(self, connection: Callable, enabled: bool = True):
        self.connection = connection
        self.enabled = enabled

    def expand(self, query_str: str, original_query):
        text = (query_str or "").strip()
        if not self.enabled or not SIMPLE_QUERY.fullmatch(text):
            return original_query

        words = [word.lower() for word in text.split()]
        if not words or len(words) > MAX_QUERY_WORDS or any(word in {"and", "or", "not"} for word in words):
            return original_query

        try:
            return self._expand_words(words, original_query)
        except Exception as error:
            logger.warning("MeSH query expansion failed; using literal search: %s", error)
            return original_query

    def _expand_words(self, words, original_query):
        candidate_indices = {}
        spans = {}
        candidates = []
        for start in range(len(words)):
            for length in range(1, min(MESH_MAX_PHRASE_WORDS, len(words) - start) + 1):
                phrase = " ".join(words[start:start + length])
                if phrase not in candidate_indices:
                    candidate_indices[phrase] = len(candidates)
                    candidates.append(phrase)
                spans[start, length] = candidate_indices[phrase]

        connection = self.connection() if callable(self.connection) else self.connection
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.ord AS ord, t.ui AS ui
                FROM unnest(%s::text[]) WITH ORDINALITY AS c(phrase, ord)
                JOIN pmmeshterms t
                  ON t.term_key = phraseto_tsquery('english', c.phrase)::text
                 AND t.term_key <> ''
                ORDER BY c.ord, t.ui
                """,
                (candidates,),
            )
            uis_by_candidate = {}
            for ordinal, ui in cursor.fetchall():
                uis_by_candidate.setdefault(ordinal - 1, set()).add(ui)

        concepts = []
        position = 0
        while position < len(words):
            matched_length = next(
                (length for length in range(min(MESH_MAX_PHRASE_WORDS, len(words) - position), 0, -1)
                 if spans[position, length] in uis_by_candidate),
                1,
            )
            uis = uis_by_candidate.get(spans[position, matched_length], set())
            concepts.append((position, matched_length, uis))
            position += matched_length

        to_expand = {
            (start, length)
            for start, length, _ in sorted(
                (concept for concept in concepts if concept[2]),
                key=lambda concept: concept[1],
                reverse=True,
            )[:MESH_MAX_CONCEPTS_EXPANDED]
        }
        components = []
        for start, length, uis in concepts:
            literal = " ".join(words[start:start + length])
            alternatives = []
            if (start, length) in to_expand:
                alternatives.extend(self._text_query(term) for term in self._lookup_terms(uis))
            literal_query = self._text_query(literal)
            if literal_query is not None:
                alternatives.append(literal_query)
            if length > 1:
                individual = [self._text_query(word) for word in words[start:start + length]]
                if all(query is not None for query in individual):
                    alternatives.append(And(individual))
            alternatives = [query for query in alternatives if query is not None]
            if alternatives:
                components.append(Or(alternatives))

        if not components or not any(uis for _, _, uis in concepts):
            return original_query
        expanded_query = And(components) if len(components) > 1 else components[0]
        return Or([original_query, expanded_query])

    @staticmethod
    def _text_query(text):
        tokens = [token.text for token in ANALYZER(text)]
        if not tokens:
            return None
        field_queries = []
        for field in SEARCH_FIELDS:
            field_queries.append(Term(field, tokens[0]) if len(tokens) == 1 else Phrase(field, tokens))
        return Or(field_queries)

    def _lookup_terms(self, uis):
        if not uis:
            return []
        with (self.connection() if callable(self.connection) else self.connection).cursor() as cursor:
            cursor.execute(
                """
                WITH roots AS (
                    SELECT DISTINCT unnest(%s::text[]) AS ui
                ), descendants AS (
                    SELECT DISTINCT child.ui
                    FROM roots
                    JOIN pmmeshtree parent ON parent.ui = roots.ui
                    JOIN pmmeshtree child ON child.tree_number LIKE parent.tree_number || '.%%'
                    LIMIT %s
                )
                SELECT t.term
                FROM pmmeshterms t
                WHERE t.term_key <> ''
                  AND (t.ui = ANY(%s::text[])
                       OR (t.is_heading AND t.ui IN (SELECT ui FROM descendants)))
                GROUP BY t.term
                ORDER BY bool_or(t.is_heading) DESC, length(t.term), t.term
                LIMIT %s
                """,
                (list(uis), MESH_MAX_DESCENDANTS_PER_CONCEPT, list(uis), MESH_MAX_TERMS_PER_CONCEPT),
            )
            return [row[0] for row in cursor.fetchall()]
