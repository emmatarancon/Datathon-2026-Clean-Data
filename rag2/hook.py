"""The glue between copilot.pipeline.solve() and the rag2 techniques.

Called only when RAG_MODE in config.env is not 'baseline'. decide() returns
either a dedup answer (the pipeline returns it without calling the model) or
the examples for the gated/hybrid prompt. It never raises: the caller wraps it
and falls back to the original path on any exception.

The paths CSV (results/rag2-paths-<hora>.csv, override with RAG2_PATHS_FILE)
records which path answered each problem — model or dedup — with the id only,
never any problem text.
"""
import csv
import os
import sys
import time

from copilot.config import CONFIG
from rag2 import data, dedup, gate, hybrid, protocol

MODES = ("gate", "hybrid", "gate+dedup", "hybrid+dedup")

_STATE = {"stamp": None, "warned": False, "bm25_rows": None, "bm25": None}


def _stamp():
    if _STATE["stamp"] is None:
        _STATE["stamp"] = time.strftime("%Y%m%d-%H%M%S")
    return _STATE["stamp"]


def paths_file():
    override = os.environ.get("RAG2_PATHS_FILE")
    if override:
        return override
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "results", f"rag2-paths-{_stamp()}.csv")


def log_path(problem_id, path, mode, n_examples, answer, seconds, status):
    """Append one row (id + path, no problem text) to the paths CSV.

    Logging must never break the official run: errors are swallowed.
    """
    try:
        target = paths_file()
        os.makedirs(os.path.dirname(target), exist_ok=True)
        new = not os.path.exists(target)
        with open(target, "a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["id", "path", "mode", "n_examples", "answer",
                                 "seconds", "status"])
            writer.writerow([problem_id, path, mode, n_examples, answer,
                             f"{seconds:.2f}" if seconds is not None else "", status])
    except Exception:
        pass


def get_bm25(index):
    """The BM25 index over the current retrieval index, built once per process."""
    rows = len(index.rows)
    if _STATE["bm25_rows"] != rows:
        cache = os.path.join(data.RAG2_DATA_DIR, "bm25.pkl")
        _STATE["bm25"], _elapsed, _cached = hybrid.build_bm25(index, cache_path=cache)
        _STATE["bm25_rows"] = rows
        if _elapsed >= hybrid.CACHE_THRESHOLD_S:
            print(f"NOTE rag2: BM25 built in {_elapsed:.1f} s "
                  f"({'cached' if _cached else 'not cached'})", file=sys.stderr)
    return _STATE["bm25"]


def decide(problem, index=None, exclude_id=None, mode="gate", start=None):
    """The hook entry point (see the module docstring for the contract)."""
    if mode not in MODES:
        return None
    if index is None:
        if not _STATE["warned"]:
            _STATE["warned"] = True
            print(f"NOTE rag2: RAG_MODE is {mode} but no index was loaded; "
                  "run predict.py/eval.py with --rag 1 to use it", file=sys.stderr)
        return None
    seconds = time.perf_counter() - start if start is not None else 0.0
    problem_id = problem.get("id", "")
    if mode.endswith("+dedup"):
        letter, info = dedup.try_transfer(
            problem, index, exclude_id=exclude_id,
            tau_dup=protocol.config_get("RAG2_TAU_DUP", 0.97))
        if letter is not None:
            log_path(problem_id, "dedup", mode, 0, letter, seconds, "ok")
            return {"answer": letter, "status": "ok", "prompt_tokens": 0,
                    "retrieved": f"{info.get('neighbor_id', '')}:{info.get('cosine', '')}",
                    "examples": [], "rag2_path": "dedup"}
    if mode.startswith("hybrid"):
        examples, info = hybrid.select_examples(
            problem, index, get_bm25(index), exclude_id=exclude_id,
            w=protocol.config_get("RAG2_W", 0.7),
            budget_s=protocol.config_get("RAG2_BUDGET_S", 35.0))
        path = "hybrid"
    else:
        examples, info = gate.select_examples(
            problem, index, exclude_id=exclude_id,
            tau=protocol.config_get("RAG2_TAU", 0.85),
            budget_s=protocol.config_get("RAG2_BUDGET_S", 35.0))
        path = "gate"
    log_path(problem_id, "model", mode, len(examples), "", seconds, "ok")
    return {"path": path, "examples": examples, "retrieved": info.get("retrieved", ""),
            "rag2_path": path}
