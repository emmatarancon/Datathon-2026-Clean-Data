"""Letter logprobs and the debiasing math for the bias study.

The grouping and the argmax mirror the proven logic of reason/calibration.py
(previous team session, 31 tests); copied here because bias/ must import only
the standard library and copilot (final check d). letter_logprobs() additionally
returns the server timings, which diagnose.py uses to measure prefix reuse.
"""
import math
import time

from copilot.client import post
from copilot.config import CONFIG
from copilot.confidence import LETTERS
from copilot.llm import URL, build_messages

GRAMMAR = "root ::= [A-E]"


def _logaddexp(a, b):
    high = max(a, b)
    if high == -math.inf:
        return -math.inf
    return high + math.log1p(math.exp(-abs(a - b)))


def group_letter_logprobs(entries):
    """{letter: logprob} from one token's top_logprobs; 'A' and ' A' add up."""
    grouped = {}
    for entry in entries or []:
        try:
            value = float(entry["logprob"])
        except (KeyError, TypeError, ValueError):
            continue
        token = str(entry.get("token", "")).strip().upper()
        if len(token) != 1 or token not in LETTERS or not math.isfinite(value):
            continue
        grouped[token] = _logaddexp(grouped[token], value) if token in grouped else value
    return grouped


def renormalize(grouped):
    """{letter: probability} over the reported letters (softmax)."""
    if not grouped:
        return {}
    high = max(grouped.values())
    weights = {letter: math.exp(value - high) for letter, value in grouped.items()}
    total = sum(weights.values())
    if not total > 0.0:
        return {}
    return {letter: weight / total for letter, weight in weights.items()}


def letter_logprobs(messages, timeout=60, url=URL, with_grammar=True):
    """({letter: logprob} or None, timings) for one built prompt.

    Same request as the pipeline (grammar, one token, temperature 0, seed) plus
    the logprobs fields. with_grammar=False probes what the server reports
    without the grammar mask (D5's empirical check).
    """
    import sys
    payload = {"messages": [dict(m) for m in messages],
               "max_tokens": 1, "temperature": 0, "seed": int(CONFIG["SEED"]),
               "cache_prompt": True, "logprobs": True, "top_logprobs": 5}
    if with_grammar:
        payload["grammar"] = GRAMMAR
    try:
        response = post(url, payload, timeout=timeout)
        entries = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        timings = response.get("timings", {})
    except (KeyError, IndexError, TypeError, ValueError) as error:
        print(f"WARNING bias: response without usable logprobs ({error})", file=sys.stderr)
        return None, {}
    return group_letter_logprobs(entries) or None, timings


def debiased_argmax(probs, allowed_letters=None, bias=None):
    """argmax(log p - bias) over the letters with probability; None if nothing.

    Letters absent from the problem's options are excluded; if the bias does
    not cover every candidate, the raw argmax wins (deterministic A-E order).
    """
    if not probs:
        return None
    present = [l for l in LETTERS if probs.get(l, 0.0) > 0.0
               and (not allowed_letters or l in allowed_letters)]
    if not present:
        return None
    if not bias or any(l not in bias for l in present):
        return max(present, key=probs.get)
    return max(present, key=lambda l: (math.log(probs[l]) - bias[l]))


def empty_problem():
    """The no-content problem, in the exact pipeline format (contextual bias)."""
    return {"id": "calibration-na", "language": "eng", "level": -1, "text": "N/A",
            "options": {l: "N/A" for l in LETTERS}}


def contextual_bias(n=5, timeout=60, url=URL):
    """Bias from empty inputs: bias[letter] = log(mean probability)."""
    messages = build_messages(empty_problem(), examples=(), image_b64=None)
    per_request = []
    for _ in range(max(1, int(n))):
        grouped, _timings = letter_logprobs(messages, timeout=timeout, url=url)
        if not grouped:
            raise RuntimeError("no usable letter logprobs")
        per_request.append(grouped)
    sums, counts = {}, {}
    for grouped in per_request:
        for letter, prob in renormalize(grouped).items():
            sums[letter] = sums.get(letter, 0.0) + prob
            counts[letter] = counts.get(letter, 0) + 1
    mean_probs = {l: sums[l] / counts[l] for l in counts}
    bias = {l: math.log(p) for l, p in mean_probs.items() if p > 0.0}
    return _record("contextual", len(per_request), url, bias,
                   extra={"mean_probs": mean_probs, "requests": per_request})


def prior_bias(input_path, n=100, seed=2026, timeout=60, url=URL):
    """Bias from the mean logprob of each letter over train_fit problems."""
    import hashlib
    import json
    with open(input_path) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    fit = [p for p in problems if p.get("options") and p.get("text")
           and int(hashlib.sha1(p["id"].encode()).hexdigest()[:8], 16) % 5 != 0]
    fit = fit[-n:] if 0 < n < len(fit) else fit
    if not fit:
        raise RuntimeError(f"no train_fit problems with options in {input_path}")
    sums, ids = {}, []
    for problem in fit:
        grouped, _timings = letter_logprobs(
            build_messages(problem, examples=(), image_b64=None), timeout=timeout, url=url)
        if not grouped:
            raise RuntimeError(f"no usable letter logprobs for {problem['id']}")
        ids.append(f"{problem['id']}|{problem.get('language', '')}")
        for letter, value in grouped.items():
            sums[letter] = sums.get(letter, 0.0) + value
    bias = {l: sums[l] / len(fit) for l in sums}
    return _record("prior", len(fit), url, bias,
                   extra={"ids": ids, "shuffle_seed": seed, "input": input_path})


def _record(method, n, url, bias, extra=None):
    """The calibration record with N, seed and date, ready to save."""
    record = {"method": method, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
              "seed": int(CONFIG["SEED"]), "url": url, "n": n, "bias": bias}
    record.update(extra or {})
    return record


def save_bias(record, path):
    """The calibration record to results/bias-calibration-<method>.json."""
    import json
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def load_bias(path):
    """{letter: bias} from a calibration JSON, or None when broken/missing."""
    import json
    if not path:
        return None
    try:
        with open(path) as f:
            record = json.load(f)
    except (OSError, ValueError):
        return None
    bias = {}
    for letter, value in (record.get("bias") or {}).items():
        try:
            current = str(letter).strip().upper()
            value = float(value)
        except (TypeError, ValueError):
            continue
        if current in LETTERS and math.isfinite(value):
            bias[current] = value
    return bias or None


def cli(argv=None):
    """Save the chosen calibration methods to results/bias-calibration-*.json."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="contextual",
                        help="comma-separated: contextual, prior (default: contextual)")
    parser.add_argument("--input", default="data/train.jsonl")
    parser.add_argument("--n", type=int, default=None,
                        help="contextual: empty requests (5); prior: problems (100)")
    parser.add_argument("--out", default="", help="override the output path")
    args = parser.parse_args(argv)
    for method in [m.strip() for m in args.method.split(",") if m.strip()]:
        try:
            if method == "contextual":
                record = contextual_bias(args.n or 5)
            elif method == "prior":
                record = prior_bias(args.input, args.n or 100)
            else:
                sys.exit(f"unknown method: {method}")
        except (RuntimeError, OSError) as error:
            sys.exit(f"calibration {method} failed: {error} (is the model server up?)")
        out = args.out or f"results/bias-calibration-{method}.json"
        save_bias(record, out)
        print(f"== calibration {method}: {record['n']} requests -> {out}")
        for letter in LETTERS:
            if letter in record["bias"]:
                print(f"   {letter}: bias {record['bias'][letter]:+.3f}")


if __name__ == "__main__":
    cli()

