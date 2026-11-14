"""Technique 1: similarity gate + latency budget.

Examples only help if they are truly similar, and every prompt token costs
about 0.09 s of reading on the board (11 tokens/s). The gate keeps only the
top-K cosine neighbours at or above tau (max 2, options read), renders them in
a compact format (statement trimmed to 120 tokens) and drops the least similar
example first until the estimated latency fits the budget. If no neighbour
passes the gate the prompt is byte-identical to the baseline (no examples).
"""
from copilot.llm import build_messages, format_problem
from rag2 import data, protocol

K = 5                # neighbours retrieved before the gate
MAX_EXAMPLES = 2
EXAMPLE_TOKENS = 120  # compact example statement budget
DEFAULT_TAU = 0.85
DEFAULT_BUDGET_S = 35.0


def compact_example(example, max_tokens=EXAMPLE_TOKENS):
    """The compact view of a solved example, or None without read options.

    Same rendering as build_messages uses, but the statement is trimmed to
    max_tokens (chars/3.5 estimate), so the example costs about a third.
    """
    if not data.has_options(example):
        return None
    return {
        "id": example.get("id", ""),
        "text": protocol.truncate_tokens(str(example.get("text", "")).strip(), max_tokens),
        "options": dict(example.get("options") or {}),
        "label": example.get("label", ""),
    }


def _estimate_seconds(examples, problem):
    """Estimated latency of the full prompt with these compact examples."""
    messages = build_messages(problem, examples)
    tokens = protocol.messages_tokens(messages)
    return protocol.estimate_latency_seconds(tokens), tokens


def select_examples(problem, index, exclude_id=None, tau=DEFAULT_TAU,
                    k=K, max_examples=MAX_EXAMPLES, budget_s=DEFAULT_BUDGET_S):
    """Return (examples, info) for the gated prompt.

    examples is ordered least-similar first, most-similar last (closest to the
    question). info carries the retrieval string, coverage flag and estimates.
    When nobody passes the gate, examples is [] and the prompt equals baseline.
    """
    hits = index.search(problem, k=k, exclude_id=exclude_id)
    kept = [(score, stored) for score, stored in hits
            if score >= tau][:max_examples]
    examples = []
    for _score, stored in kept:  # similarity order, best first, for the budget
        view = compact_example(stored)
        if view is not None:
            examples.append(view)
    # budget: drop the least similar (last kept) until the estimate fits
    seconds, tokens = _estimate_seconds(examples, problem)
    dropped = 0
    while examples and seconds > budget_s:
        examples.pop()  # the last one is the least similar of those kept
        dropped += 1
        seconds, tokens = _estimate_seconds(examples, problem)
    examples.reverse()  # most similar last, nearest to the question
    retrieved = ";".join(f"{p['id']}:{s:.2f}" for s, p in hits)
    info = {
        "retrieved": retrieved,
        "hits": [(s, p) for s, p in hits],
        "kept": [p["id"] for p in examples],
        "coverage": bool(examples),
        "dropped_by_budget": dropped,
        "estimated_tokens": tokens,
        "estimated_seconds": round(seconds, 2),
    }
    return examples, info


def baseline_identical(examples):
    """True when the gated prompt is exactly the baseline prompt."""
    return not examples


def prompt_bytes(problem, examples):
    """The exact messages sent to the model, for the equivalence test."""
    return build_messages(problem, examples)


def format_problem_reference(problem):
    """Re-export for tests comparing with the existing renderer."""
    return format_problem(problem)
