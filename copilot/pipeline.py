"""The improved copilot: one problem in, one letter out.

This file replaces the baseline solve() with a smarter approach:
  1. Smart retrieval: up to 5 examples, filtered by level similarity
  2. Multiple strategies tried in order: vote_safe > fewshot > calibrated > fallback
  3. Each strategy is tried until one succeeds within the time limit

predict.py and eval.py both call solve(), so whatever you build here is exactly
what the official run measures.
"""
import socket
import sys
import time
import urllib.error

from .confidence import enabled as doubt_enabled, summarize as summarize_probs
from .config import CONFIG
from .llm import URL, answer_letter, answer_letter_probs, build_messages
from .strategist import solve_smart

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
          time_limit=TIME_LIMIT, doubt=None, strategy=None):
    """Return {"answer", "seconds", "status", ...} for one problem.
    
    If strategy is None, read from STRATEGY in config.env (default: "auto").
    """
    if strategy is None:
        strategy = str(CONFIG.get("STRATEGY", "auto")).strip().lower() or "auto"
    """Return {"answer", "seconds", "status", ...} for one problem.

    The strategy parameter controls the solving approach:
      - "auto": tries vote_safe > fewshot > calibrated > fallback (default)
      - "vote": few-shot + majority vote (3 attempts)
      - "vote_safe": vote with confidence-based early stopping
      - "fewshot": structured few-shot prompt, one attempt
      - "calibrated": single attempt with bias correction
      - "fallback": return most common answer for similar problems
      - "baseline": original simple prompt (no examples, no voting)

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

    # Handle rag2 path (dedup: answered without the model)
    rag2_state = _rag2(problem, index, exclude_id, start)
    if rag2_state and rag2_state.get("answer"):
        result.update(rag2_state)
        result["seconds"] = round(time.perf_counter() - start, 2)
        return result

    if doubt is None:
        doubt = doubt_enabled()

    try:
        # --- Build examples list from RAG ---
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

        # --- Main solving path ---
        if strategy == "baseline":
            # Original simple approach: one attempt, no examples
            if doubt or doubt_enabled() or calibration_file():
                letter, timings, probs = answer_letter_probs(
                    problem, examples, timeout=max(remaining, 1.0),
                    image_b64=image_b64, url=llm_url)
            else:
                letter, timings = answer_letter(
                    problem, examples, timeout=max(remaining, 1.0),
                    image_b64=image_b64, url=llm_url)
                probs = None
            if calibration_file():
                letter, note = _corrected_answer(problem, letter, probs)
                result["calibration"] = note
            result["answer"] = letter
            result["prompt_tokens"] = timings.get("prompt_n", 0)

        elif strategy == "auto":
            # Try smart strategies in order
            letter, note = solve_smart(problem, index, mode="auto", timeout=remaining)
            if letter:
                result["answer"] = letter
                result["strategy_used"] = note
            else:
                # Final fallback: use the simple baseline
                letter, timings = answer_letter(
                    problem, examples, timeout=max(remaining, 1.0),
                    image_b64=image_b64, url=llm_url)
                result["answer"] = letter
                result["strategy_used"] = "fallback_to_baseline"
                timings_dict = timings if timings else {}
                result["prompt_tokens"] = timings_dict.get("prompt_n", 0)

        elif strategy == "vote_safe":
            # Use the vote_safe strategy with confidence checking
            letter, note = solve_smart(problem, index, mode="vote_safe", timeout=remaining)
            if letter:
                result["answer"] = letter
                result["strategy_used"] = note
            else:
                letter, timings, probs = answer_letter_probs(
                    problem, examples, timeout=max(remaining, 1.0),
                    image_b64=image_b64, url=llm_url)
                result["answer"] = letter
                result["strategy_used"] = "fallback_baseline"
                result["prompt_tokens"] = timings.get("prompt_n", 0)
                if probs:
                    result["probs"] = {l: round(p, 4) for l, p in probs.items()}
                    result["confidence"] = summarize_probs(probs)
                    if doubt or doubt_enabled() or calibration_file():
                        if calibration_file():
                            letter2, note = _corrected_answer(problem, letter, probs)
                            result["answer"] = letter2
                            result["calibration"] = note
                            result["prompt_tokens"] = timings.get("prompt_n", 0)

        elif strategy == "fewshot":
            letter, note = solve_smart(problem, index, mode="fewshot", timeout=remaining)
            if letter:
                result["answer"] = letter
                result["strategy_used"] = note
            else:
                letter, timings = answer_letter(
                    problem, examples, timeout=max(remaining, 1.0),
                    image_b64=image_b64, url=llm_url)
                result["answer"] = letter
                result["strategy_used"] = "fallback_baseline"
                result["prompt_tokens"] = timings.get("prompt_n", 0)

        elif strategy == "vote":
            letter, note = solve_smart(problem, index, mode="vote", timeout=remaining)
            if letter:
                result["answer"] = letter
                result["strategy_used"] = note
            else:
                letter, timings = answer_letter(
                    problem, examples, timeout=max(remaining, 1.0),
                    image_b64=image_b64, url=llm_url)
                result["answer"] = letter
                result["strategy_used"] = "fallback_baseline"
                result["prompt_tokens"] = timings.get("prompt_n", 0)

        elif strategy == "calibrated":
            letter, note = solve_smart(problem, index, mode="calibrated", timeout=remaining)
            if letter:
                result["answer"] = letter
                result["strategy_used"] = note
            else:
                letter, timings = answer_letter(
                    problem, examples, timeout=max(remaining, 1.0),
                    image_b64=image_b64, url=llm_url)
                result["answer"] = letter
                result["strategy_used"] = "fallback_baseline"
                result["prompt_tokens"] = timings.get("prompt_n", 0)

        elif strategy == "fallback":
            letter = solve_smart(problem, index, mode="fallback", timeout=remaining)
            if letter:
                result["answer"] = letter
                result["strategy_used"] = "fallback"
            else:
                letter, timings = answer_letter(
                    problem, examples, timeout=max(remaining, 1.0),
                    image_b64=image_b64, url=llm_url)
                result["answer"] = letter
                result["strategy_used"] = "baseline"
                result["prompt_tokens"] = timings.get("prompt_n", 0)

        else:
            # Unknown strategy, fall back to baseline
            letter, timings = answer_letter(
                problem, examples, timeout=max(remaining, 1.0),
                image_b64=image_b64, url=llm_url)
            result["answer"] = letter
            result["strategy_used"] = f"unknown_strategy_{strategy}_baseline"
            result["prompt_tokens"] = timings.get("prompt_n", 0)

        # Handle calibration and doubt at the end
        if doubt or doubt_enabled():
            probs = result.get("probs")
            if probs:
                result["confidence"] = summarize_probs(probs)
            elif result.get("strategy_used", "").startswith("fallback"):
                # If we fell back to baseline, re-check confidence
                cal_path = calibration_file()
                if cal_path and "probs" in result:
                    letter2, note = _corrected_answer(problem, result["answer"], result.get("probs"))
                    result["answer"] = letter2
                    result["calibration"] = note

    except (socket.timeout, TimeoutError):
        result["status"] = "timeout"
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            result["status"] = "timeout"
        else:
            result["status"] = f"error: {error.reason}"
    result["seconds"] = round(time.perf_counter() - start, 2)
    return result