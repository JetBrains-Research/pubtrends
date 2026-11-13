import shutil
import tempfile
import unittest
from types import SimpleNamespace

from pysrc.papers.db.lucene_searcher import LuceneIndexManager
from pysrc.papers.db.mesh_terms import MeshRecord, MeshTermExpander, parse_mesh_ascii


class TestMeshAsciiParser(unittest.TestCase):

    def test_parses_descriptors_supplementary_concepts_and_inverted_terms(self):
        text = """*NEWRECORD
RECTYPE = D
MH = Antibodies, Monoclonal
ENTRY = Monoclonal Antibody|T047|...
MN = C20.683.515
UI = D000911
*NEWRECORD
RECTYPE = C
NM = bevonium
SY = Acabel|T109|...
UI = C004939
"""

        records = list(parse_mesh_ascii(text.splitlines()))

        self.assertEqual(2, len(records))
        self.assertEqual("D000911", records[0].ui)
        self.assertEqual("Antibodies, Monoclonal", records[0].heading)
        self.assertEqual(["Monoclonal Antibodies", "Monoclonal Antibody"], records[0].entry_terms)
        self.assertEqual(["C20.683.515"], records[0].tree_numbers)
        self.assertEqual(("C004939", "bevonium", ["Acabel"], []),
                         (records[1].ui, records[1].heading, records[1].entry_terms,
                          records[1].tree_numbers))

    def test_skips_invalid_records_and_unwieldy_terms(self):
        text = """*NEWRECORD
MH = Missing UI
*NEWRECORD
MH = Valid Heading
UI = D000001
ENTRY = alpha beta gamma delta epsilon zeta eta|T109
"""

        records = list(parse_mesh_ascii(text.splitlines()))

        self.assertEqual(1, len(records))
        self.assertEqual("Valid Heading", records[0].heading)
        self.assertEqual([], records[0].entry_terms)


class FakeMeshCursor:

    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, query, params):
        self.connection.calls += 1
        if self.connection.fail:
            raise RuntimeError("MeSH tables are unavailable")
        if "WITH ORDINALITY" in query:
            self.rows = []
            for ordinal, candidate in enumerate(params[0], start=1):
                for ui in self.connection.matches.get(" ".join(candidate.lower().split()), []):
                    self.rows.append((ordinal, ui))
        elif "WITH roots AS" in query:
            roots = set(params[0])
            expanded = roots.union(*(self.connection.descendants.get(ui, set()) for ui in roots))
            self.rows = [(term,) for ui in expanded for term in self.connection.terms.get(ui, [])]
        else:
            raise AssertionError(f"Unexpected MeSH query: {query}")

    def fetchall(self):
        return self.rows


class FakeMeshConnection:

    def __init__(self, matches=None, terms=None, descendants=None, fail=False):
        self.matches = matches or {}
        self.terms = terms or {}
        self.descendants = descendants or {}
        self.fail = fail
        self.calls = 0

    def cursor(self):
        return FakeMeshCursor(self)


class TestMeshQueryExpansion(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = LuceneIndexManager(self.temp_dir)
        self.manager.reset()
        articles = [
            self.article("1", "Myocardial infarction outcomes", year=2020),
            self.article("2", "Heart attack outcomes", year=2021),
            self.article("3", "Myocardial stunning outcomes", year=2022),
            self.article("4", "Mammary tissue outcomes", year=2021),
            self.article("5", "Myocardial stunning review", year=2021, kind="Review"),
            self.article("6", "Breast neoplasms overview", year=2021),
        ]
        self.manager.insert_pubmed_articles(articles)
        self.connection = FakeMeshConnection(
            matches={
                "heart attack": ["D1"],
                "breast": ["D3"],
                "breast cancer": ["D2"],
            },
            terms={
                "D1": ["Myocardial Infarction", "Heart Attack"],
                "D2": ["Breast Cancer", "Breast Neoplasms"],
                "D3": ["Breast", "Mammary Tissue"],
                "D4": ["Myocardial Stunning"],
            },
            descendants={"D1": {"D4"}},
        )
        self.expander = MeshTermExpander(lambda: self.connection)

    @staticmethod
    def article(pmid, title, year, kind="Article"):
        return SimpleNamespace(pmid=pmid, title=title, abstract="", keywords=[], mesh=[],
                               year=year, type=kind, doi="")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_expands_synonyms_and_descendants_without_losing_literal_results(self):
        results = self.manager.search("heart attack", noreviews=True, query_expander=self.expander)

        self.assertCountEqual(["1", "2", "3"], results)

    def test_longest_matching_phrase_wins(self):
        results = self.manager.search("breast cancer", noreviews=False, query_expander=self.expander)

        self.assertEqual(["6"], results)

    def test_syntax_is_not_expanded_and_filters_remain_active(self):
        calls_before = self.connection.calls
        quoted = self.manager.search('"heart attack"', noreviews=False, query_expander=self.expander)
        self.assertEqual(calls_before, self.connection.calls)

        boolean_query = self.manager.parse_query("heart OR attack")
        self.expander.expand("heart OR attack", boolean_query)
        self.assertEqual(calls_before, self.connection.calls)

        filtered = self.manager.search("heart attack", min_year=2021, max_year=2022,
                                       noreviews=True, query_expander=self.expander)

        self.assertEqual(["2"], quoted)
        self.assertCountEqual(["2", "3"], filtered)

    def test_unavailable_vocabulary_falls_back_to_literal_search(self):
        self.connection.fail = True

        results = self.manager.search("heart attack", noreviews=False, query_expander=self.expander)

        self.assertEqual(["2"], results)


if __name__ == "__main__":
    unittest.main()
