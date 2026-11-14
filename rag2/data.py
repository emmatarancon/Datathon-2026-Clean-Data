"""Shared data loading, the deterministic dev split, and the in-memory index.

The in-memory index reuses the float32 vectors already stored in data/index.db
(never recomputes embeddings) and mirrors copilot.rag.Index.search exactly:
same query embedding (with the bge query prefix), same dot product over unit
vectors (= cosine), same exclude_id rule, same order.
"""
import hashlib
import json
import os
import sqlite3
from array import array

from copilot.llm import format_problem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH = os.path.join(DATA_DIR, "val.jsonl")
INDEX_PATH = os.path.join(DATA_DIR, "index.db")
RAG2_DATA_DIR = os.path.join(DATA_DIR, "rag2")

# Test/mock hook: when set, query_embed uses it instead of the embedding server.
_MOCK_EMBED = None


def query_embed(texts):
    """Unit vectors for query texts (bge query prefix), like Index.search does."""
    if _MOCK_EMBED is not None:
        return _MOCK_EMBED(texts)
    from copilot.embed import embed
    return embed(texts, query=True)


def load_problems(path=TRAIN_PATH, languages=("eng",)):
    """Problems from a .jsonl, in file order, filtered by language (eng here)."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                problem = json.loads(line)
                if languages is None or problem.get("language") in languages:
                    rows.append(problem)
    return rows


def load_train(languages=("eng",)):
    return load_problems(TRAIN_PATH, languages)


def load_val(languages=("eng",)):
    return load_problems(VAL_PATH, languages)


def is_dev(problem_id):
    """The deterministic dev rule: int(sha1(id)[:8], 16) % 5 == 0."""
    return int(hashlib.sha1(problem_id.encode()).hexdigest()[:8], 16) % 5 == 0


def dev_ids(ids):
    return {pid for pid in ids if is_dev(pid)}


def split_train(problems):
    """(train_fit, dev) by id: dev problems never enter the tuning index."""
    fit = [p for p in problems if not is_dev(p["id"])]
    dev = [p for p in problems if is_dev(p["id"])]
    return fit, dev


def has_options(problem):
    """True when OCR read text options (options is {} when it didn't)."""
    return bool(problem.get("options"))


class MemIndex:
    """Cosine index in memory over (key, problem, vector) rows.

    search() is line-by-line equivalent to copilot.rag.Index.search.
    """

    def __init__(self, rows):
        self.rows = [(key, problem, vec) for key, problem, vec in rows]

    @classmethod
    def from_db(cls, db_path=INDEX_PATH, exclude_ids=None):
        """Load stored vectors from a build_index.py database, skipping ids.

        exclude_ids is a set of problem ids (the dev ids for tuning), so ALL
        rows of those ids are left out of the tuning index.
        """
        exclude = set(exclude_ids or ())
        rows = []
        db = sqlite3.connect(db_path)
        for key, problem, blob in db.execute("SELECT key, problem, vec FROM problems"):
            stored = json.loads(problem)
            if stored["id"] in exclude:
                continue
            vec = array("f")
            vec.frombytes(blob)
            rows.append((key, stored, vec))
        db.close()
        return cls(rows)

    @classmethod
    def from_problems(cls, problems, vectors):
        """Build from problems plus one vector each (mock mode and tests)."""
        return cls([(f"{p['id']}|{p.get('language', '')}", p, v)
                    for p, v in zip(problems, vectors)])

    def __len__(self):
        return len(self.rows)

    def search(self, problem, k=1, exclude_id=None):
        """Up to k (score, problem) pairs, most similar first. Same as Index.search."""
        query = query_embed([format_problem(problem)])[0]
        scored = []
        for _key, stored, vec in self.rows:
            if exclude_id and stored["id"] == exclude_id:
                continue
            scored.append((sum(a * b for a, b in zip(query, vec)), stored))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:k]
