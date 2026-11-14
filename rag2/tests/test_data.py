"""Tests for rag2/data.py: dev split, language filter, in-memory index."""
import json
import os
import sqlite3
import tempfile
import unittest
from array import array

from copilot import rag as copilot_rag
from rag2 import data


def fake_embed(texts, query=False, timeout=60):
    """Deterministic stand-in for the embedding server (same for both indexes)."""
    from rag2 import mock
    return mock.mock_embed(texts)


def make_db(path, problems, vectors):
    db = sqlite3.connect(path)
    db.execute(copilot_rag.SCHEMA)
    db.executemany("INSERT OR REPLACE INTO problems VALUES (?, ?, ?)", [
        (f"{p['id']}|{p['language']}", json.dumps(p, ensure_ascii=False), array("f", v).tobytes())
        for p, v in zip(problems, vectors)])
    db.commit()
    db.close()


class DevSplitTest(unittest.TestCase):
    def test_is_dev_matches_spec_exactly(self):
        import hashlib
        for pid in ("lvl-0_2015_13", "sample-1", "x", "", "lvl-4_2024_20"):
            expected = int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16) % 5 == 0
            self.assertEqual(data.is_dev(pid), expected)

    def test_split_partitions_everything(self):
        problems = [{"id": f"p{i}", "language": "eng", "text": "t"} for i in range(50)]
        fit, dev = data.split_train(problems)
        self.assertEqual(len(fit) + len(dev), 50)
        ids_fit = {p["id"] for p in fit}
        ids_dev = {p["id"] for p in dev}
        self.assertFalse(ids_fit & ids_dev)
        self.assertEqual(ids_fit | ids_dev, {p["id"] for p in problems})
        self.assertTrue(all(data.is_dev(pid) for pid in ids_dev))
        self.assertFalse(any(data.is_dev(pid) for pid in ids_fit))

    def test_is_dev_deterministic(self):
        self.assertEqual(data.is_dev("lvl-0_2015_13"), data.is_dev("lvl-0_2015_13"))


class LoadTest(unittest.TestCase):
    def test_language_filter_keeps_file_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "problems.jsonl")
            rows = [
                {"id": "a", "language": "eng", "text": "one"},
                {"id": "b", "language": "cat", "text": "dos"},
                {"id": "c", "language": "eng", "text": "three"},
            ]
            with open(path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            loaded = data.load_problems(path, languages=("eng",))
            self.assertEqual([p["id"] for p in loaded], ["a", "c"])
            loaded_all = data.load_problems(path, languages=None)
            self.assertEqual(len(loaded_all), 3)


class MemIndexTest(unittest.TestCase):
    def setUp(self):
        self._old_mock = data._MOCK_EMBED
        data._MOCK_EMBED = fake_embed
        self._old_rag_embed = copilot_rag.embed
        copilot_rag.embed = fake_embed  # copilot.rag.Index must use the same fake

    def tearDown(self):
        data._MOCK_EMBED = self._old_mock
        copilot_rag.embed = self._old_rag_embed

    def test_memindex_matches_copilot_index(self):
        problems = [
            {"id": "t1", "language": "eng", "text": "How many apples has Anna got left?",
             "options": {"A": "8", "B": "9", "C": "10"}, "label": "C"},
            {"id": "t2", "language": "eng", "text": "What is the value of 2 + 3 * 4?",
             "options": {"A": "20", "B": "14"}, "label": "B"},
            {"id": "t3", "language": "eng", "text": "A square has perimeter 36 cm, area?",
             "options": {"A": "81 cm2", "B": "144 cm2"}, "label": "A"},
        ]
        query = {"id": "q", "language": "eng", "text": "How many apples has Anna got?",
                 "options": {"A": "8", "B": "9"}, "label": "B"}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.db")
            vectors = fake_embed([data.format_problem(p) for p in problems])
            make_db(path, problems, vectors)
            theirs = copilot_rag.Index(path).search(query, k=3, exclude_id="q")
            ours = data.MemIndex.from_db(path).search(query, k=3, exclude_id="q")
        self.assertEqual([p["id"] for _s, p in theirs], [p["id"] for _s, p in ours])
        for (s1, _p1), (s2, _p2) in zip(theirs, ours):
            self.assertAlmostEqual(s1, s2, places=6)

    def test_exclude_id_skips_every_row_of_that_id(self):
        vec = [1.0, 0.0, 0.0]
        rows = [("a|eng", {"id": "same", "text": "x"}, vec),
                ("a|cat", {"id": "same", "text": "y"}, vec),
                ("b|eng", {"id": "other", "text": "z"}, vec)]
        index = data.MemIndex(rows)
        data._MOCK_EMBED = lambda texts: [[1.0, 0.0, 0.0]]  # cosine 1.0 with both
        hits = index.search({"id": "q", "text": "q"}, k=3, exclude_id="same")
        self.assertEqual([p["id"] for _s, p in hits], ["other"])

    def test_from_db_filters_dev_ids(self):
        candidates = [f"d{i}" for i in range(100)]
        dev_id = next(pid for pid in candidates if data.is_dev(pid))
        fit_id = next(pid for pid in (f"f{i}" for i in range(100)) if not data.is_dev(pid))
        rows = [(f"{dev_id}|eng", {"id": dev_id, "language": "eng", "text": "dev"}, [1.0, 0.0]),
                (f"{fit_id}|eng", {"id": fit_id, "language": "eng", "text": "fit"}, [0.0, 1.0])]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.db")
            make_db(path, [r[1] for r in rows], [r[2] for r in rows])
            index = data.MemIndex.from_db(path, exclude_ids={dev_id})
        self.assertEqual([p["id"] for _k, p, _v in index.rows], [fit_id])

    def test_has_options(self):
        self.assertTrue(data.has_options({"options": {"A": "1"}}))
        self.assertFalse(data.has_options({"options": {}}))
        self.assertFalse(data.has_options({}))


if __name__ == "__main__":
    unittest.main()
