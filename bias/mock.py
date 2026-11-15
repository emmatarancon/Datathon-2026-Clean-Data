"""--mock: simulated model answers and logprobs, so diagnose.py/final.py run
on the laptop without the board or its servers. Mock numbers are NOT
measurements: they only prove the plumbing works. Real measurements happen
only on the board. The mock distribution is almost uniform with a slight D
prior, mirroring the behavior under study.
"""
import hashlib

BASE_PROBS = {"A": 0.19, "B": 0.16, "C": 0.21, "D": 0.28, "E": 0.16}


def _seed(*parts):
    raw = "|".join(str(p) for p in parts)
    return int(hashlib.sha1(raw.encode()).hexdigest()[:8], 16)


def mock_probs(problem, examples=()):
    """Deterministic probabilities over A-E: uniform-ish with a D lean."""
    jitter = {letter: (_seed(problem.get("id", ""), letter) % 40 - 20) / 400.0
              for letter in "ABCDE"}
    raw = {letter: max(0.01, BASE_PROBS[letter] + jitter[letter])
           for letter in "ABCDE"}
    total = sum(raw.values())
    return {letter: value / total for letter, value in raw.items()}


def mock_answer_probs(problem, examples=(), timeout=60, image_b64=None, url=None):
    """(letter, timings, probs) like answer_letter_probs, deterministic."""
    probs = mock_probs(problem, examples)
    letter = max("ABCDE", key=lambda l: (probs[l], -"ABCDE".index(l)))
    prompt_n = 90 + len(str(problem.get("text", ""))) // 4
    return letter, {"prompt_n": prompt_n}, probs


def mock_answer(problem, examples=(), timeout=60, image_b64=None, url=None):
    """(letter, timings) like answer_letter, deterministic."""
    letter, timings, _probs = mock_answer_probs(problem, examples)
    return letter, timings


def mock_letter_logprobs(messages, timeout=60, url=None, with_grammar=True):
    """({letter: logprob}, timings) like bias.calibration.letter_logprobs."""
    import math
    problem = _problem_from_messages(messages)
    probs = mock_probs(problem) if problem else dict(BASE_PROBS)
    grouped = {letter: math.log(probs[letter]) for letter in "ABCDE"}
    prompt_n = 90 + len(str(problem.get("text", ""))) // 4 if problem else 60
    return grouped, {"prompt_n": prompt_n}


def _problem_from_messages(messages):
    """The last user message back into a rough problem dict (for the mock)."""
    for message in reversed(messages or []):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return {"id": content[:40], "text": content, "options": {"A": "1"}}
    return None


def install():
    """Patch the pipeline model calls and the bias logprob path."""
    import copilot.pipeline as pipeline
    pipeline.answer_letter = mock_answer
    pipeline.answer_letter_probs = mock_answer_probs
    from bias import calibration
    calibration.letter_logprobs = mock_letter_logprobs
