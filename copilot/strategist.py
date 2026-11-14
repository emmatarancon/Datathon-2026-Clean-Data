"""Smart solving strategies that replace the simple baseline.

Strategies available (selected by the pipeline):

  1. **fewshot** – retrieve 5 examples, keep the best 3 by similarity,
     build a structured prompt with examples + step-by-step reasoning,
     answer once.
  2. **vote** – same prompt as fewshot but run it 3 times (different seeds);
     the majority letter wins. If there is a tie, fall back to the example
     with the highest similarity.
  3. **vote_safe** – like vote, but after each attempt the model reports
     letter probabilities; if the top letter is "sure" (above threshold)
     we accept it immediately (no need for 3 attempts).
  4. **calibrated** – single attempt with logprobs + letter-bias correction
     (reason/calibration.py).
  5. **fallback** – always return the most common answer for the same
     level/year group (computed from the retrieval index).

The pipeline tries them in order until one succeeds within the time limit.
"""
import json
import time
from collections import Counter

from copilot.confidence import summarize as summarize_probs, sure_threshold
from copilot.config import CONFIG
from copilot.llm import URL, answer_letter, answer_letter_probs, build_structured_prompt
from copilot.rag import Index

# ---------------------------------------------------------------------- helpers

def _majority_vote(answers):
    """Return the letter that appears most often. Tie-break: A < B < C < D < E."""
    LETTERS = "ABCDE"
    counts = Counter(answers)
    max_count = max(counts.values())
    candidates = [l for l, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        return candidates[0], max_count
    # tie-break: pick the one that comes first in A-E order
    return min(candidates, key=lambda l: LETTERS.index(l)), max_count


def _get_retrieved_examples(problem, index, k_retrieve=5, exclude_id=None):
    """Retrieve and filter examples. Prefer same-level problems."""
    if index is None:
        return [], "no_index"
    hits = index.search(problem, k=k_retrieve, exclude_id=exclude_id)
    if not hits:
        return [], "no_hits"

    # Prefer examples at the same level
    level = problem.get("level")
    same_level = [(s, p) for s, p in hits if p.get("level") == level]
    if not same_level:
        same_level = [(s, p) for s, p in hits if p.get("level") is not None]
    
    examples = []
    for _s, p in (same_level or hits)[:3]:
        if p.get("label"):  # only include examples with a known answer
            examples.append(p)
    
    if not examples:
        return [], "no_examples"
    return examples, "ok"


def _get_fallback_by_similarity(problem, index, k_retrieve=20):
    """Return the most common answer among similar problems in the index."""
    if index is None:
        return None
    hits = index.search(problem, k=k_retrieve)
    if not hits:
        return None
    
    # Among high-similarity neighbors, count the labels
    high_sim = [(s, p) for s, p in hits if s > 0.7]
    if not high_sim:
        high_sim = [(s, p) for s, p in hits if s > 0.5]
    
    if not high_sim:
        return None
    
    labels = [p["label"] for _s, p in high_sim if p.get("label")]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


# ---------------------------------------------------------------------- strategies

def try_fewshot(problem, index=None, k_retrieve=5, timeout=50):
    """Strategy: few-shot with structured reasoning, one attempt.

    Returns (letter, note) or (None, "reason") if retrieval fails.
    """
    examples, reason = _get_retrieved_examples(problem, index, k_retrieve)
    if not examples:
        return None, reason

    try:
        letter, _timings = answer_letter(
            problem, examples, timeout=timeout,
            prompt_func=build_structured_prompt)
        return letter, f"fewshot/{len(examples)}ex"
    except Exception as e:
        return None, f"error: {e}"


def try_vote(problem, index=None, k_retrieve=5, n_attempts=3, timeout=15):
    """Strategy: few-shot + majority vote across n_attempts.

    Each attempt uses a different seed for variation. If a clear majority
    emerges, return that letter. Otherwise, return the highest-similarity
    neighbor's label.
    """
    examples, reason = _get_retrieved_examples(problem, index, k_retrieve)
    if not examples:
        return None, reason

    answers = []
    max_seed = int(CONFIG.get("SEED", 42))
    for attempt in range(n_attempts):
        try:
            seed = max_seed + (attempt * 7 + 13) % 1000
            letter, _timings = answer_letter(
                problem, examples, timeout=timeout, seed=seed,
                prompt_func=build_structured_prompt)
            answers.append(letter)
        except Exception:
            continue
        if len(answers) >= n_attempts:
            break

    if not answers:
        return None, "no_answers"

    letter, majority_count = _majority_vote(answers)
    note = f"vote/{len(answers)}attempts"
    if majority_count >= 2:
        note += f"/{majority_count}-vote"
    return letter, note


def try_vote_safe(problem, index=None, k_retrieve=5, n_attempts=3, timeout=20):
    """Strategy: vote with confidence check.

    Each attempt gets logprobs. If the model is "sure" (top prob >= threshold),
    accept immediately. Otherwise collect all attempts and do majority vote.
    """
    examples, reason = _get_retrieved_examples(problem, index, k_retrieve)
    if not examples:
        return None, reason

    threshold = sure_threshold()
    answers = []
    confident_letter = None
    max_seed = int(CONFIG.get("SEED", 42))

    for attempt in range(n_attempts):
        try:
            seed = max_seed + (attempt * 7 + 13) % 1000
            letter, _timings, probs = answer_letter_probs(
                problem, examples, timeout=timeout, seed=seed,
                prompt_func=build_structured_prompt)
            answers.append(letter)
            if probs:
                info = summarize_probs(probs, threshold)
                if info["sure"] and confident_letter is None:
                    confident_letter = letter
                    break  # accept confident answer
        except Exception:
            continue

    if not answers:
        return None, "no_answers"

    if confident_letter is not None:
        note = f"vote_safe/confident"
        return confident_letter, note

    letter, majority_count = _majority_vote(answers)
    note = f"vote_safe/{len(answers)}attempts/{majority_count}-vote/no_confident"
    return letter, note


def try_calibrated(problem, index=None, k_retrieve=3, timeout=50):
    """Strategy: single attempt with logprobs + bias correction.

    Uses reason/calibration.py for letter-bias correction.
    """
    try:
        letter, _timings, probs = answer_letter_probs(
            problem, [], timeout=timeout, prompt_func=build_structured_prompt)
    except Exception:
        return None, "no_logprobs"

    if not probs:
        return None, "no_logprobs"

    # Apply bias correction if available
    from copilot.pipeline import calibration_file, _corrected_answer
    cal_path = calibration_file()
    if cal_path:
        from reason.calibration import load_bias, corrected_letter
        try:
            bias = load_bias(cal_path)
            if bias:
                options = {str(key).strip().upper() for key in (problem.get("options") or {})}
                letter = corrected_letter(probs, options or None, bias)
                if letter:
                    return letter, f"calibrated"
        except Exception:
            pass

    return letter, "calibrated/no_bias"


def try_fallback(problem, index=None, k_retrieve=20):
    """Strategy: return the most common answer among similar problems.

    This is a weak strategy but useful as a last resort.
    """
    return _get_fallback_by_similarity(problem, index, k_retrieve), "fallback"


# ---------------------------------------------------------------------- dispatch

STRATEGIES = {
    "fewshot": try_fewshot,
    "vote": try_vote,
    "vote_safe": try_vote_safe,
    "calibrated": try_calibrated,
    "fallback": try_fallback,
}


def solve_smart(problem, index=None, mode="auto", timeout=50):
    """Run the best strategy based on mode or auto-detection.

    In "auto" mode: try vote_safe first (best accuracy), fall back to
    fewshot, then calibrated, then fallback.

    mode can be: "auto", "fewshot", "vote", "vote_safe", "calibrated", "fallback"
    """
    if mode == "auto":
        # Try in order of quality vs time tradeoff
        results = [
            try_vote_safe(problem, index, timeout=max(timeout * 0.4, 15)),
            try_fewshot(problem, index, timeout=max(timeout * 0.5, 20)),
            try_calibrated(problem, index, timeout=max(timeout * 0.5, 20)),
            try_fallback(problem, index),
        ]
        for letter, note in results:
            if letter:
                return letter, note
        return CONFIG.get("FALLBACK_ANSWER", "D"), "all_failed"

    strategy_fn = STRATEGIES.get(mode)
    if strategy_fn is None:
        return None, f"unknown mode: {mode}"
    letter, note = strategy_fn(problem, index, timeout=timeout)
    return letter, note