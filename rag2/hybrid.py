"""Technique 2: hybrid retrieval, BM25 + cosine, pure standard library.

Embeddings catch the topic but confuse problems sharing vocabulary with
different numbers; BM25 is sensitive to exact words and numbers. Both rankings
are min-max normalized over the union of their top-20 candidates and combined
as w*cosine + (1-w)*bm25; RRF (k=60) is measured as an alternative. There is
no topic field in M3Kang (SKILL.md), so retrieval quality uses level+year
equality and the number overlap of the statements as the topic proxy.
"""
import hashlib
import math
import os
import pickle
import re
import time
from collections import Counter

from copilot.llm import format_problem
from rag2 import data, values
from rag2.gate import compact_example

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
# deliberately short: numbers and content words must survive
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "by", "for", "with", "and", "or",
    "to", "is", "are", "was", "were", "be", "been", "has", "have", "had", "do",
    "does", "did", "that", "this", "these", "those", "it", "its", "as", "from",
    "not", "no",
}
K1 = 1.5
B = 0.75
TOPN = 20
DEFAULT_W = 0.7
RRF_K = 60
CACHE_THRESHOLD_S = 3.0


def tokenize(text):
    """Lowercase words and numbers ('3.5' stays one token); stopwords out."""
    return [t for t in TOKEN_RE.findall(str(text or "").lower()) if t not in STOPWORDS]


def doc_text(problem):
    """BM25 document: statement plus option values."""
    options = " ".join(str(v) for v in (problem.get("options") or {}).values())
    return f"{problem.get('text', '')} {options}"


class BM25:
    """Inverted-index BM25 over the training documents, built once at startup."""

    def __init__(self, docs, k1=K1, b=B):
        self.k1, self.b = k1, b
        self.ids = [doc_id for doc_id, _text in docs]
        self.doc_lens = []
        self.postings = {}  # term -> [[row index, term frequency], ...]
        for idx, (_doc_id, text) in enumerate(docs):
            tokens = tokenize(text)
            self.doc_lens.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, []).append([idx, tf])
        self.n = len(docs)
        self.avgdl = (sum(self.doc_lens) / self.n) if self.n else 0.0

    @classmethod
    def from_payload(cls, payload):
        bm25 = cls.__new__(cls)
        bm25.k1 = payload["k1"]
        bm25.b = payload["b"]
        bm25.ids = payload["ids"]
        bm25.doc_lens = payload["doc_lens"]
        bm25.postings = {term: [list(p) for p in posting]
                         for term, posting in payload["postings"].items()}
        bm25.n = payload["n"]
        bm25.avgdl = payload["avgdl"]
        return bm25

    def payload(self, fingerprint):
        return {"fingerprint": fingerprint, "k1": self.k1, "b": self.b,
                "ids": self.ids, "doc_lens": self.doc_lens,
                "postings": self.postings, "n": self.n, "avgdl": self.avgdl}

    def idf(self, term):
        df = len(self.postings.get(term, ()))
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query_text):
        """{row index: bm25 score} for one query text (deterministic)."""
        scores = {}
        for term in tokenize(query_text):  # repeated query terms count twice
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            for idx, tf in posting:
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_lens[idx] / self.avgdl)
                scores[idx] = scores.get(idx, 0.0) + idf * tf * (self.k1 + 1.0) / denom
        return scores


def _fingerprint(bm25):
    raw = f"{bm25.ids}|{bm25.doc_lens}|{bm25.n}"
    return hashlib.sha1(raw.encode()).hexdigest()


def build_bm25(index, cache_path=None):
    """BM25 over the index rows; cached with pickle when building is slow.

    The cache stores postings and statistics, never raw problem text, and is
    only reused when the fingerprint (ids and document lengths) matches.
    """
    docs = [(problem["id"], doc_text(problem)) for _key, problem, _vec in index.rows]
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if payload.get("fingerprint") == _fingerprint(BM25(docs)):
                return BM25.from_payload(payload), 0.0, True
        except Exception:
            pass  # a broken cache is rebuilt, never trusted
    start = time.perf_counter()
    bm25 = BM25(docs)
    elapsed = time.perf_counter() - start
    cached = False
    if cache_path and elapsed > CACHE_THRESHOLD_S:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(bm25.payload(_fingerprint(bm25)), f)
            cached = True
        except Exception:
            pass
    return bm25, elapsed, cached


def cosine_pairs(index, problem, exclude_id=None, k=TOPN):
    """Top-k (row index, cosine) pairs, mirroring MemIndex.search."""
    query = data.query_embed([format_problem(problem)])[0]
    pairs = []
    for idx, (_key, stored, vec) in enumerate(index.rows):
        if exclude_id and stored["id"] == exclude_id:
            continue
        pairs.append((idx, sum(a * b for a, b in zip(query, vec))))
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    return pairs[:k]


def bm25_pairs(bm25, problem, k=TOPN):
    scores = bm25.score(doc_text(problem))
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:k]


def minmax(pairs):
    """Each score normalized to [0,1] inside the candidate list."""
    if not pairs:
        return {}
    scores = [s for _idx, s in pairs]
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return {idx: 0.0 for idx, _s in pairs}
    return {idx: (s - lo) / (hi - lo) for idx, s in pairs}


def fuse(cos, bm, w=DEFAULT_W, topn=TOPN):
    """w*normalized cosine + (1-w)*normalized BM25 over the top candidates."""
    cos_norm = minmax(cos[:topn])
    bm_norm = minmax(bm[:topn])
    combined = {}
    for idx, s in cos_norm.items():
        combined[idx] = combined.get(idx, 0.0) + w * s
    for idx, s in bm_norm.items():
        combined[idx] = combined.get(idx, 0.0) + (1.0 - w) * s
    return sorted(combined.items(), key=lambda pair: pair[1], reverse=True)


def rrf(cos, bm, k=RRF_K, topn=TOPN):
    """Reciprocal Rank Fusion with constant k=60."""
    combined = {}
    for rank, (idx, _s) in enumerate(cos[:topn]):
        combined[idx] = combined.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _s) in enumerate(bm[:topn]):
        combined[idx] = combined.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(combined.items(), key=lambda pair: pair[1], reverse=True)


def select_examples(problem, index, bm25, exclude_id=None, w=DEFAULT_W,
                    method="weighted", max_examples=2, budget_s=35.0):
    """(examples, info): fused examples with the same budget rule as the gate.

    The fusion replaces the threshold gate: no tau here, but at most
    max_examples and the same latency budget (least similar drops first).
    """
    from copilot.llm import build_messages
    from rag2 import protocol

    cos = cosine_pairs(index, problem, exclude_id)
    bm = bm25_pairs(bm25, problem)
    fused = rrf(cos, bm) if method == "rrf" else fuse(cos, bm, w)
    examples = []
    for idx, _score in fused[:max_examples]:
        view = compact_example(index.rows[idx][1])
        if view is not None:
            examples.append(view)

    def estimate(current):
        tokens = protocol.messages_tokens(build_messages(problem, current))
        return protocol.estimate_latency_seconds(tokens), tokens

    seconds, tokens = estimate(examples)
    dropped = 0
    while examples and seconds > budget_s:
        examples.pop()  # least similar of the fused list (kept order)
        dropped += 1
        seconds, tokens = estimate(examples)
    examples.reverse()  # most similar last, nearest to the question
    info = {
        "retrieved": ";".join(
            f"{index.rows[idx][1]['id']}:{score:.2f}" for idx, score in fused[:max_examples]),
        "kept": [e["id"] for e in examples],
        "coverage": bool(examples),
        "dropped_by_budget": dropped,
        "estimated_tokens": tokens,
        "estimated_seconds": round(seconds, 2),
        "cos_top1": index.rows[cos[0][0]][1]["id"] if cos else "",
        "bm25_top1": index.rows[bm[0][0]][1]["id"] if bm else "",
        "fused_top1": index.rows[fused[0][0]][1]["id"] if fused else "",
    }
    return examples, info


def number_jaccard(a_text, b_text):
    """Set overlap of the statement numbers, the topic proxy without labels."""
    a = set(values.statement_numbers(a_text))
    b = set(values.statement_numbers(b_text))
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def top1_related(problem, neighbor):
    """Retrieval-quality check for one top-1 neighbour.

    related = same level AND year, or statement number Jaccard >= 0.5.
    """
    same_level_year = (neighbor.get("level") == problem.get("level")
                       and neighbor.get("year") == problem.get("year"))
    jaccard = number_jaccard(problem.get("text", ""), neighbor.get("text", ""))
    return {"same_level_year": same_level_year, "number_jaccard": round(jaccard, 3),
            "related": bool(same_level_year or jaccard >= 0.5)}
