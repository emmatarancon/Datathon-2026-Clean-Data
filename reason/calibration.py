"""Fix the model's letter bias with its own log probabilities.

The server answers with one letter (grammar `root ::= [A-E]`), but it reports
the log probability of that token's alternatives. One letter the model picks
too often is a bias we can subtract: for every letter present in the answer
we compare its log probability against the bias of that letter, and output
the letter with the best corrected score instead of the sampled one.

The bias comes from one of two calibration methods:

- contextual: ask for the letter probabilities of an empty problem (text and
  all options "N/A") and average them. A letter that is frequent even with no
  content shows a prior, and a per-letter bias is the log of that average.
- empirical prior: average the raw log probabilities of every letter over a
  deterministic subset of `data/train.jsonl` problems (with options), and
  subtract that mean.

Either way, the per-letter bias ends up in a small JSON file (saved in
results/), and `corrected_letter()` turns (log probability - bias) into the
letter to answer. Both pieces are used by the hook in copilot/pipeline.py,
which stays off until `CALIBRATION_FILE` in config.env points at such a file.

Only the standard library, so it runs on the board without pip.
"""
import json
import math
import os
import random
import sys
import time
import urllib.error

# Run both as `python3 -m reason.calibration` and as a plain script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from copilot.client import post                     # noqa: E402
from copilot.config import CONFIG                   # noqa: E402
from copilot.confidence import LETTERS              # noqa: E402
from copilot.llm import URL, build_messages         # noqa: E402

GRAMMAR = "root ::= [A-E]"
NO_LETTERS = "no usable letter logprobs"


def letter_of_token(token):
    """'A', ' A' or 'a' -> 'A'; anything longer or foreign -> None."""
    token = str(token).strip().upper()
    return token if len(token) == 1 and token in LETTERS else None


def group_letter_logprobs(entries):
    """Raw {letter: logprob} from the top_logprobs of one generated token.

    The model may reach the same letter through several tokens ("A" and " A"),
    so the probabilities of the variants are added together (logaddexp). NaN
    values and non-letter tokens are skipped.
    """
    grouped = {}
    for entry in entries or []:
        try:
            value = float(entry["logprob"])
        except (KeyError, TypeError, ValueError):
            continue
        letter = letter_of_token(entry.get("token", ""))
        if letter is None or not math.isfinite(value):
            continue
        if letter in grouped:
            grouped[letter] = _logaddexp(grouped[letter], value)
        else:
            grouped[letter] = value
    return grouped


def _logaddexp(a, b):
    high = max(a, b)
    if high == -math.inf:
        return -math.inf
    return high + math.log1p(math.exp(-abs(a - b)))


def renormalize(grouped):
    """{letter: probability} over the letters reported (softmax)."""
    if not grouped:
        return {}
    high = max(grouped.values())
    weights = {letter: math.exp(value - high) for letter, value in grouped.items()}
    total = sum(weights.values())
    if not total > 0.0:
        return {}
    return {letter: weight / total for letter, weight in weights.items()}


def letter_logprobs(prompt_messages, timeout=60, url=URL, retry_without_grammar=True):
    """Raw {letter: logprob} for an already-built chat prompt, or None.

    Uses the existing HTTP client: the request is the same one the pipeline
    sends (grammar, one token, temperature 0, seed), plus the logprobs fields.
    If the grammar-masked answer carries no usable letters (or the server
    ignores the logprobs fields) the request is repeated once without the
    grammar, so the returned letters are whichever the server reports.
    """
    base = {"messages": [dict(m) for m in prompt_messages],
            "max_tokens": 1, "temperature": 0, "seed": int(CONFIG["SEED"]),
            "cache_prompt": True, "logprobs": True, "top_logprobs": 5}
    for index, grammar in enumerate([GRAMMAR, None] if retry_without_grammar else [GRAMMAR]):
        payload = dict(base)
        if grammar:
            payload["grammar"] = grammar
        try:
            response = post(url, payload, timeout=timeout)
            entries = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError, ValueError):
            entries = None  # answered, but without logprobs
        grouped = group_letter_logprobs(entries)
        if grouped:
            return grouped
        if index == 0 and not retry_without_grammar:
            break
    return None


def empty_problem():
    """A problem with no content, in the exact format the pipeline uses.

    `build_messages` reads "text" and "options" only, so the calibration
    prompt is exactly a pipeline prompt with nothing in it.
    """
    return {"id": "calibration-na", "language": "eng", "level": -1, "text": "N/A",
            "options": {letter: "N/A" for letter in LETTERS}}


def contextual_bias(n=5, timeout=60, url=URL):
    """Bias from empty inputs: average probabilities, subtract log of them.

    Returns the calibration record; 'bias' holds {letter: log(mean prob)},
    which makes the corrected argmax identical to dividing each probability
    by its average, as specified.
    """
    messages = build_messages(empty_problem(), examples=(), image_b64=None)
    per_request = []
    for _ in range(max(1, int(n))):
        grouped = letter_logprobs(messages, timeout=timeout, url=url)
        if not grouped:
            raise RuntimeError(NO_LETTERS)
        per_request.append(grouped)
    mean_logprobs, mean_probs, counts = {}, {}, {}
    for grouped in per_request:  # each letter averages only where it showed up
        for letter, prob in renormalize(grouped).items():
            mean_probs[letter] = mean_probs.get(letter, 0.0) + prob
            counts[letter] = counts.get(letter, 0) + 1
    for grouped in per_request:
        for letter, value in grouped.items():
            mean_logprobs[letter] = mean_logprobs.get(letter, 0.0) + value
    mean_probs = {letter: total / counts[letter] for letter, total in mean_probs.items()}
    mean_logprobs = {letter: total / counts[letter] for letter, total in mean_logprobs.items()}
    bias = {letter: math.log(mean_probs[letter]) for letter in LETTERS if mean_probs.get(letter, 0.0) > 0.0}
    return {"method": "contextual", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(CONFIG["SEED"]), "url": url, "n": len(per_request),
            "mean_probs": mean_probs, "mean_logprobs": mean_logprobs,
            "bias": bias, "requests": per_request}


def prior_bias(input_path="data/train.jsonl", n=100, timeout=60, url=URL, seed=42):
    """Bias from the raw log probabilities over a deterministic train subset.

    The subset is the LAST n problems of the same Random(42) shuffle that
    eval.py uses, so calibration and a `--limit N` dev run stay disjoint as
    long as n + N <= number of train problems. Only problems with options are
    used (the letters ask for them), and the labels are never read.
    """
    with open(input_path) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    problems = [p for p in problems if p.get("options") and p.get("text")]
    random.Random(seed).shuffle(problems)
    problems = problems[-n:] if 0 < n < len(problems) else problems
    if not problems:
        raise RuntimeError(f"no problems with options in {input_path}")
    sums, counts, ids = {}, {}, []
    for problem in problems:
        ids.append(f"{problem['id']}|{problem.get('language', '')}")
        grouped = letter_logprobs(build_messages(problem, examples=(), image_b64=None),
                                  timeout=timeout, url=url)
        if not grouped:
            raise RuntimeError(f"{NO_LETTERS} for {ids[-1]}")
        for letter, value in grouped.items():
            sums[letter] = sums.get(letter, 0.0) + value
            counts[letter] = counts.get(letter, 0) + 1
    bias = {letter: sums[letter] / counts[letter] for letter in LETTERS if counts.get(letter)}
    mean_probs = {letter: counts[letter] / len(problems) for letter in bias}
    return {"method": "prior", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(CONFIG["SEED"]), "shuffle_seed": seed, "url": url,
            "input": input_path, "n": len(problems), "ids": ids,
            "counts": counts, "mean_probs": mean_probs, "bias": bias}


# ---------------------------------------------------------------- application

def corrected_letter(probs, allowed_letters=None, bias=None):
    """The letter the model should answer given its letter probabilities.

    probs is {letter: probability} as produced by copilot.confidence (zeros
    for letters the server did not report). Candidates are the letters with
    probability, restricted to allowed_letters (the problem's own options,
    when it has any). The winner is the highest log(p) - bias; ties keep the
    first letter in A-E order, so the result is deterministic.

    When the bias does not cover every candidate the correction would be
    apples-to-oranges, so we return the uncorrected argmax (with temperature
    0 that is exactly the letter the model sampled). None means "nothing to
    go on": no usable probabilities at all.
    """
    if not bias or not probs:
        return None
    present = [letter for letter in LETTERS
               if probs.get(letter, 0.0) > 0.0
               and (not allowed_letters or letter in allowed_letters)]
    if not present:
        return None
    if any(letter not in bias for letter in present):
        return max(present, key=probs.get)
    return max(present, key=lambda letter: (math.log(probs[letter]) - bias[letter]))


_BIAS_CACHE = {}


def load_bias(path):
    """{letter: bias} from a calibration JSON file, cached; None when broken.

    Fantasy letters and unreadable values are dropped one by one, so a stray
    field in the JSON cannot cost the whole correction.
    """
    path = os.path.abspath(path)
    try:
        stamp = os.stat(path).st_mtime
    except OSError:
        return None
    if _BIAS_CACHE.get(path, (None, None))[0] == stamp:
        return _BIAS_CACHE[path][1]
    bias = None
    try:
        with open(path) as f:
            record = json.load(f)
        entries = dict((record.get("bias") or {}).items())
        bias = {}
        for letter, value in entries.items():
            try:
                letter, value = str(letter).strip().upper(), float(value)
            except (TypeError, ValueError):
                continue
            if letter in LETTERS and math.isfinite(value):
                bias[letter] = value
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        bias = None
    _BIAS_CACHE[path] = (stamp, bias)
    return bias


def wilson_interval(hits, n, z=1.96):
    """Wilson score interval (95% by default) for a proportion."""
    if not n:
        return (0.0, 1.0)
    share = hits / n
    square = z * z / (4 * n * n)
    middle = share + z * z / (2 * n)
    spread = z * math.sqrt(share * (1 - share) / n + square)
    denominator = 1 + z * z / n
    return (max(0.0, (middle - spread) / denominator),
            min(1.0, (middle + spread) / denominator))


# ------------------------------------------------------------------------ CLI

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Measure the letter bias of the model and save it to results/calibration-<method>.json",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default="contextual",
                        help="comma-separated: contextual, prior (default: contextual)")
    parser.add_argument("--input", default="data/train.jsonl",
                        help="train problems for the prior method")
    parser.add_argument("--n", type=int, default=None,
                        help="contextual: empty requests; prior: train problems "
                             "(default: 5 and 100)")
    parser.add_argument("--url", default=URL, help="llama-server chat completions URL")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--out", default="", help="override the results/calibration-<method>.json path")
    args = parser.parse_args(argv)

    for method in [m.strip() for m in args.method.split(",") if m.strip()]:
        start = time.perf_counter()
        try:
            if method == "contextual":
                record = contextual_bias(args.n or 5, timeout=args.timeout, url=args.url)
            elif method == "prior":
                record = prior_bias(args.input, args.n or 100, timeout=args.timeout, url=args.url)
            else:
                sys.exit(f"unknown method: {method}")
        except (RuntimeError, OSError) as error:  # server down, no letters, no data
            sys.exit(f"calibration {method} failed: {error} (is the model server on port "
                     f"{CONFIG['LLM_PORT']} up?)")
        out = args.out or f"results/calibration-{method}.json"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(record, f, indent=2)
        print(f"== calibration {method}: {record['n']} requests in "
              f"{time.perf_counter() - start:.1f} s -> {out}")
        for letter in LETTERS:
            value = record["bias"].get(letter)
            if value is not None:
                extra = math.exp(value) if method == "contextual" else value
                unit = "mean probability" if method == "contextual" else "mean logprob"
                print(f"   {letter}: bias {value:+.3f}  ({unit} {extra:.3f})")
        if any(record["bias"].get(letter) is None for letter in LETTERS):
            print("   WARNING: some letters never showed up; the hook falls back to the raw letter for them")


if __name__ == "__main__":
    main()



