"""Technique 3: transfer the answer across near-duplicate problems.

Kangaroo repeats problems across levels and years with different ids. When the
current problem is a near-identical twin of a training problem (cosine, text
ratio, EXACT number multiset and option-value multiset all agree), the VALUE
of the neighbour's correct option is looked up among the current options and
its letter is answered with no model call (0 s). The letters may be in another
order, so the letter itself is never copied. The current problem's label is
never read here — only the neighbour's training label is.
"""
import difflib
import re

from rag2 import data, values

DEFAULT_TAU_DUP = 0.97
TEXT_RATIO = 0.90


def normalized_text(text):
    """Lowercase, without the leading problem number, whitespace collapsed.

    The leading '13.' is the position in that year's exam, not part of the
    problem: near-duplicates from different years carry different numbers.
    The lookahead keeps decimals like '13.5 kg' intact.
    """
    t = str(text or "").strip().lower()
    t = re.sub(r"^\s*\d+\s*[.)]\s*(?!\d)", "", t)
    return re.sub(r"\s+", " ", t)


def try_transfer(problem, index, exclude_id=None, tau_dup=DEFAULT_TAU_DUP, k=1):
    """Return (letter or None, info) for the top-1 neighbour, or a reason.

    Every failed condition records its reason in info; None means "follow the
    normal model path".
    """
    info = {"neighbor_id": "", "cosine": None, "reason": ""}
    if not data.has_options(problem):
        info["reason"] = "current problem has no read options"
        return None, info
    hits = index.search(problem, k=k, exclude_id=exclude_id)
    if not hits:
        info["reason"] = "no candidates"
        return None, info
    score, cand = hits[0]
    info["neighbor_id"] = cand.get("id", "")
    info["cosine"] = round(score, 4)
    if cand.get("id") == problem.get("id"):
        info["reason"] = "same id"
        return None, info
    if score < tau_dup:
        info["reason"] = "cosine below tau_dup"
        return None, info
    if not data.has_options(cand):
        info["reason"] = "neighbor without read options"
        return None, info
    ratio = difflib.SequenceMatcher(
        None, normalized_text(problem.get("text")), normalized_text(cand.get("text"))).ratio()
    info["text_ratio"] = round(ratio, 4)
    if ratio < TEXT_RATIO:
        info["reason"] = "text ratio below 0.90"
        return None, info
    if values.statement_numbers(normalized_text(problem.get("text"))) != \
            values.statement_numbers(normalized_text(cand.get("text"))):
        info["reason"] = "statement numbers differ"
        return None, info
    if values.values_multiset(problem.get("options")) != values.values_multiset(cand.get("options")):
        info["reason"] = "option values differ"
        return None, info
    label = str(cand.get("label", "")).strip().upper()[:1]
    neighbor_options = cand.get("options") or {}
    if label not in neighbor_options:
        info["reason"] = "neighbor label missing or invalid"
        return None, info
    want = values.canonical_value(neighbor_options[label])
    matches = sorted(letter for letter, v in (problem.get("options") or {}).items()
                     if values.canonical_value(v) == want)
    info["want"] = want
    info["matches"] = matches
    if len(matches) == 1:
        info["reason"] = "transferred"
        return matches[0], info
    info["reason"] = f"{len(matches)} options match the value"
    return None, info
