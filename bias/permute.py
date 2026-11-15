"""Correction C: answer by permuted options, scored per CONTENT.

Cyclic shifts of the options (k=1,2,3) remove position information: for each
shift the model's letter logprobs are mapped back to the CONTENT they picked,
and the content with the best average score wins. The answer is the letter of
that content in the ORIGINAL order. A time budget limits the requests: when it
runs out, the shifts already measured are used (and none, the hook falls back).
"""
import math
import time

from copilot.confidence import LETTERS
from copilot.llm import build_messages
from bias.calibration import letter_logprobs, renormalize

SHIFTS = (1, 2, 3)
REQUEST_BUDGET_S = 12.0  # one shifted request costs ~9-12 s on the board


def shifted_options(options, shift):
    """The options dict with contents moved `shift` letters forward."""
    letters = [l for l in LETTERS if l in (options or {})]
    return {letter: options[letters[(index - shift) % len(letters)]]
            for index, letter in enumerate(letters)}


def content_letter(answered_letter, shift, n):
    """The original letter whose content the model picked under `shift`."""
    index = LETTERS.index(answered_letter)
    return LETTERS[(index - shift) % n]


def distinct_contents(problem):
    """True when the problem has 5 options with five different values."""
    options = problem.get("options") or {}
    values = list(options.values())
    return len(options) == 5 and len({str(v).strip().lower() for v in values}) == 5


def content_scores(problem, examples=(), image_b64=None, url=None, timeout=12.0,
                   shifts=SHIFTS, bias=None, deadline=None):
    """({letter: mean score}, info): scores per ORIGINAL letter, higher wins.

    score = mean over the used shifts of (log p - bias) of the content. When
    the deadline runs out the already-measured shifts are used; with none, the
    scores dict comes back empty and the hook falls back.
    """
    original = problem.get("options") or {}
    letters = [l for l in LETTERS if l in original]
    n = len(letters)
    totals, counts, prompt_n, used = {}, {}, [], []
    for shift in shifts:
        if deadline is not None and time.monotonic() >= deadline:
            break  # budget exhausted: keep what is already measured
        shifted = dict(problem, options=shifted_options(original, shift))
        grouped, timings = letter_logprobs(
            build_messages(shifted, examples, image_b64), timeout=timeout, url=url)
        if not grouped:
            continue
        used.append(shift)
        prompt_n.append(timings.get("prompt_n", 0))
        for letter, prob in renormalize(grouped).items():
            target = content_letter(letter, shift, n)
            score = math.log(prob) if prob > 0.0 else -6.0
            if bias and target in bias:
                score -= bias[target]
            totals[target] = totals.get(target, 0.0) + score
            counts[target] = counts.get(target, 0) + 1
    scores = {letter: totals[letter] / counts[letter] for letter in totals}
    return scores, {"used_shifts": used, "prompt_n": prompt_n}


def permuted_letter(problem, examples=(), image_b64=None, url=None, timeout=12.0,
                    shifts=SHIFTS, bias=None, deadline=None):
    """(letter in original order or None, info): the best-content letter."""
    options = problem.get("options") or {}
    if not distinct_contents(problem):
        return None, {"reason": "options missing, not 5, or with repeated content"}
    scores, info = content_scores(problem, examples, image_b64, url, timeout,
                                  shifts, bias, deadline)
    if not scores:
        info["reason"] = info.get("reason", "no permutation finished in the budget")
        return None, info
    candidates = [l for l in LETTERS if l in options and l in scores]
    if not candidates:
        info["reason"] = "no scored content among the problem's letters"
        return None, info
    info["scores"] = {l: round(scores[l], 3) for l in candidates}
    return max(candidates, key=lambda l: (scores[l], -LETTERS.index(l))), info
