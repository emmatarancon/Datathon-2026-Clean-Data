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


def _rag2(problem, index, exclude_id, start):
    """rag2 hook (reason branch): None on baseline or on any failure.

    RAG_MODE in config.env picks the mode: baseline (the default) keeps this
    function returning None and the original path untouched; gate, hybrid,
    gate+dedup and hybrid+dedup ask rag2 for the examples and, for the dedup
    variants, possibly for the whole answer without a model call. Any
    exception in rag2 falls back to the original path, never the other way.
    """
    mode = str(CONFIG.get("RAG_MODE", "baseline") or "baseline").strip().lower()
    if mode in ("", "baseline"):
        return None
    try:
        from rag2.hook import decide
        return decide(problem, index=index, exclude_id=exclude_id, mode=mode, start=start)
    except Exception as error:  # the hook must never break the official run
        print(f"WARNING rag2: hook unavailable ({error}); using the original path",
              file=sys.stderr)
        return None


def calibration_file():
    """The CALIBRATION_FILE path from config.env, or None when empty (off).

    Off by default, so with nothing set in config.env every request, answer
    and output stays exactly what it used to be.
    """
    path = str(CONFIG.get("CALIBRATION_FILE", "") or "").strip()
    return path or None


def debias_mode():
    """DEBIAS_MODE from config.env: off (default) or a bias/ correction.

    off keeps solve() exactly what it was; calibrate, permute and
    calibrate+permute ask bias/ for the letter from the model's own logprobs.
    """
    mode = str(CONFIG.get("DEBIAS_MODE", "off") or "off").strip().lower()
    return mode if mode in ("calibrate", "permute", "calibrate+permute") else "off"


def _bias_answer(problem, letter, probs, examples, image_b64, llm_url, remaining):
    """bias hook (reason branch): None keeps the model's letter (off/failure).

    Any exception in bias/ falls back to the original letter, never the other
    way round; missing logprobs are handled inside the hook the same way.
    """
    mode = debias_mode()
    if mode == "off":
        return None
    try:
        from bias.hook import debias_letter
        return debias_letter(problem, letter, probs, mode=mode, examples=examples,
                             image_b64=image_b64, llm_url=llm_url, remaining=remaining)
    except Exception as error:  # the hook must never break the official run
        print(f"WARNING bias: hook unavailable ({error}); keeping the raw letter",
              file=sys.stderr)
        return None


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
    rag2_state = _rag2(problem, index, exclude_id, start)
    if rag2_state and rag2_state.get("answer"):  # dedup: answered without the model
        result.update(rag2_state)
        result["seconds"] = round(time.perf_counter() - start, 2)
        return result
    if doubt is None:
        doubt = doubt_enabled()
    try:
        examples = []
        if rag2_state:  # rag2 (gate or hybrid) chose the examples
            examples = rag2_state.get("examples") or []
            result["retrieved"] = rag2_state.get("retrieved", "")
            result["rag2_path"] = rag2_state.get("path", "")
        elif index is not None and k > 0:
            hits = index.search(problem, k=k, exclude_id=exclude_id)
            examples = [p for _score, p in hits]
            result["examples"] = hits
            result["retrieved"] = ";".join(f"{p['id']}:{s:.2f}" for s, p in hits)
        remaining = time_limit - (time.perf_counter() - start) - MARGIN
        calibrating = calibration_file() is not None
        debias = debias_mode()
        if doubt or calibrating or debias != "off":  # the doubt detector, the
            # calibration and the debias need the letter probabilities: same
            # request, plus the logprobs fields
            letter, timings, probs = answer_letter_probs(problem, examples, timeout=max(remaining, 1.0),
                                                         image_b64=image_b64, url=llm_url)
        else:
            letter, timings = answer_letter(problem, examples, timeout=max(remaining, 1.0),
                                            image_b64=image_b64, url=llm_url)
            probs = None
        if calibrating:  # answer with argmax(logprob - bias), not the sampled letter
            letter, note = _corrected_answer(problem, letter, probs)
            result["calibration"] = note
        if debias != "off":  # bias/ hook: letter from the model's own logprobs
            chosen = _bias_answer(problem, letter, probs, examples, image_b64, llm_url,
                                  time_limit - (time.perf_counter() - start) - MARGIN)
            if chosen is not None:
                letter, note = chosen
                result["debias"] = note
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
