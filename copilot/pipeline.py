"""The baseline copilot: one problem in, one letter out.

This is the file to change first. predict.py and eval.py both call solve(),
so whatever you build here is exactly what the official run measures.
"""
import socket
import sys
import time
import urllib.error

from .confidence import enabled as doubt_enabled, summarize
from .config import CONFIG
from .llm import URL, answer_letter, answer_letter_probs

TIME_LIMIT = float(CONFIG["TIME_LIMIT_SECONDS"])
FALLBACK = CONFIG["FALLBACK_ANSWER"]
MARGIN = 3.0  # seconds kept free for Python and for writing the answer


def calibration_file():
    """The CALIBRATION_FILE path from config.env, or None when empty (off).

    Off by default, so with nothing set in config.env every request, answer
    and output stays exactly what it used to be.
    """
    path = str(CONFIG.get("CALIBRATION_FILE", "") or "").strip()
    return path or None


def _corrected_answer(problem, letter, probs):
    """The letter after bias correction (reason/calibration.py), or the raw one.

    Returns (letter, note). The letter only changes when there are usable
    letter probabilities, a valid calibration file, and a bias for every
    candidate letter; otherwise the model's sampled letter stays and the
    note explains why.
    """
    path = calibration_file()
    if not probs:
        print("WARNING calibration: no logprobs from the server, keeping the model's letter",
              file=sys.stderr)
        return letter, "kept: no logprobs"
    try:
        from reason.calibration import corrected_letter, load_bias
        bias = load_bias(path)
    except Exception as error:  # the hook must never break the official run
        print(f"WARNING calibration: hook unavailable ({error}); keeping the raw letter", file=sys.stderr)
        return letter, f"kept: {error}"
    if not bias:
        print(f"WARNING calibration: no usable bias in {path}; keeping the raw letter", file=sys.stderr)
        return letter, "kept: no usable bias file"
    options = {str(key).strip().upper() for key in (problem.get("options") or {})}
    chosen = corrected_letter(probs, options or None, bias)
    if chosen is None:
        print("WARNING calibration: no candidate letters, keeping the model's letter", file=sys.stderr)
        return letter, "kept: no letters"
    if chosen != letter:
        print(f"NOTE calibration: the model said {letter}, the bias correction prefers {chosen}",
              file=sys.stderr)
    return chosen, f"applied: {chosen}"


def solve(problem, index=None, k=0, exclude_id=None, image_b64=None, llm_url=URL,
          time_limit=TIME_LIMIT, doubt=None):
    """Return {"answer", "seconds", "status", ...} for one problem.

    index and k turn on retrieval: the k most similar solved problems are
    added to the prompt as examples. Each example costs reading time.
    image_b64 and llm_url send the problem image to a vision server.
    doubt=None follows DOUBT_DETECTOR in config.env; True or False forces it.
    When it is on, the result also carries "probs" ({letter: probability}) and
    "confidence" (top letter, margin, sure or doubt) — the request, the answer
    and the timings stay exactly the same as without it.
    CALIBRATION_FILE in config.env (empty = off) turns on letter-bias
    correction (reason/calibration.py): the answer becomes the argmax of
    logprob - bias and the result carries a "calibration" note.
    """
    start = time.perf_counter()
    result = {"answer": FALLBACK, "status": "ok", "prompt_tokens": 0, "retrieved": "", "examples": []}
    if doubt is None:
        doubt = doubt_enabled()
    try:
        examples = []
        if index is not None and k > 0:
            hits = index.search(problem, k=k, exclude_id=exclude_id)
            examples = [p for _score, p in hits]
            result["examples"] = hits
            result["retrieved"] = ";".join(f"{p['id']}:{s:.2f}" for s, p in hits)
        remaining = time_limit - (time.perf_counter() - start) - MARGIN
        calibrating = calibration_file() is not None
        if doubt or calibrating:  # the doubt detector and the calibration need
            # the letter probabilities: same request, plus the logprobs fields
            letter, timings, probs = answer_letter_probs(problem, examples, timeout=max(remaining, 1.0),
                                                         image_b64=image_b64, url=llm_url)
        else:
            letter, timings = answer_letter(problem, examples, timeout=max(remaining, 1.0),
                                            image_b64=image_b64, url=llm_url)
            probs = None
        if calibrating:  # answer with argmax(logprob - bias), not the sampled letter
            letter, note = _corrected_answer(problem, letter, probs)
            result["calibration"] = note
        result["answer"] = letter
        result["prompt_tokens"] = timings.get("prompt_n", 0)
        if probs:
            result["probs"] = {l: round(p, 4) for l, p in probs.items()}
            result["confidence"] = summarize(probs)
    except (socket.timeout, TimeoutError):
        result["status"] = "timeout"
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            result["status"] = "timeout"
        else:
            result["status"] = f"error: {error.reason}"
    result["seconds"] = round(time.perf_counter() - start, 2)
    return result
