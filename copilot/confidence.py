"""The doubt detector: how sure is the model about its letter?

The copilot forces one letter out of the model with the grammar `root ::= [A-E]`
and `max_tokens: 1`. Asking the server for the log probabilities of that token
(`"logprobs": true, "top_logprobs": 5` in the request, as the README suggests)
also returns the probability of every other letter, because the grammar masks
the rest and the reported probabilities are renormalized over A-E. One bar
clearly taller than the rest means the model is sure; five similar bars mean
it is guessing.

Measuring it is a calibration: split the answers by how confident the model
was, and check that the answers it was sure about are really the ones it gets
right. `calibration_text()` prints that table; `calibration.py` runs it on the
CSV that `eval.py --doubt` writes.

Everything here is additive. With `DOUBT_DETECTOR=0` (the default in
config.env) no extra request field is sent, no extra key appears in solve(),
and every output keeps exactly its baseline shape.
"""
import math

from .config import CONFIG

LETTERS = "ABCDE"


def enabled():
    """Is the doubt detector on? DOUBT_DETECTOR in config.env (default off)."""
    return str(CONFIG.get("DOUBT_DETECTOR", "0")).strip().lower() in ("1", "true", "yes", "on")


def sure_threshold():
    """A letter at or above this probability (0-1) counts as 'sure'."""
    try:
        threshold = float(CONFIG.get("DOUBT_SURE_PROB", 0.5))
    except (TypeError, ValueError):
        threshold = 0.5
    return threshold if 0.0 < threshold <= 1.0 else 0.5


def extract_letter_probs(response):
    """Normalized {letter: probability} from a chat completion response, or None.

    The server may report raw or grammar-masked probabilities; renormalizing
    over the letters it reports makes both cases comparable. Fewer than two
    letters with a finite logprob means the logprobs are unusable, so we return
    None instead of pretending the model is sure about something.
    """
    try:
        entries = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return None
    logprobs = {}
    for entry in entries or []:
        try:
            token = str(entry["token"]).strip().upper()
            value = float(entry["logprob"])
        except (KeyError, TypeError, ValueError):
            continue
        # the grammar only allows single letters, but stay tolerant all the same
        if len(token) == 1 and token in LETTERS and token not in logprobs and value > -math.inf:
            logprobs[token] = value
    if len(logprobs) < 2:
        return None
    weights = {token: math.exp(value) for token, value in logprobs.items()}
    total = sum(weights.values())
    if not total > 0.0:
        return None
    probs = {letter: 0.0 for letter in LETTERS}
    for token, weight in weights.items():
        probs[token] = weight / total
    return probs


def summarize(probs, threshold=None):
    """The verdict for the bars: top letter, margin, entropy, sure or doubt."""
    if threshold is None:
        threshold = sure_threshold()
    top_letter = max(probs, key=probs.get)
    ordered = sorted(probs.values(), reverse=True)
    top = probs[top_letter]
    second = ordered[1] if len(ordered) > 1 else 0.0
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0.0) / math.log(len(probs) or 2)
    sure = top >= threshold
    return {"top": top_letter, "prob": round(top, 4), "margin": round(top - second, 4),
            "entropy": round(entropy, 4), "sure": sure, "verdict": "sure" if sure else "doubt",
            "threshold": threshold}


def bars(probs, width=24):
    """The five bars as text, for the terminal. The model's pick is marked."""
    info = summarize(probs)
    lines = []
    for letter in LETTERS:
        p = probs.get(letter, 0.0)
        filled = round(p * width)
        mark = "  <- the model's pick" if letter == info["top"] else ""
        lines.append(f"{letter} {'█' * filled}{'·' * (width - filled)} {100 * p:5.1f}%{mark}")
    state = "sure, one bar dominates" if info["sure"] else "doubting, the bars are close"
    lines.append(f"The model is {state} (top letter {info['top']} at {100 * info['prob']:.1f}%).")
    return "\n".join(lines)


# Buckets for the calibration table. The top probability of five letters is
# never below 0.2: that is the value when they are all equally likely.
BUCKETS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def bucket_label(p):
    """Which confidence bucket a top probability falls in, e.g. '[0.4-0.6)'."""
    p = min(max(p, 0.0), 1.0)
    for lo, hi in zip(BUCKETS, BUCKETS[1:]):
        if lo <= p < hi:
            return f"[{lo:.1f}-{hi:.1f}]" if hi == 1.0 else f"[{lo:.1f}-{hi:.1f})"
    return f"[{BUCKETS[-2]:.1f}-{BUCKETS[-1]:.1f}]"


def bucket_labels():
    """The bucket labels in order, so the table never has gaps."""
    return [bucket_label(lo) for lo in BUCKETS[:-1]]


def calibration(rows, threshold=None):
    """Confidence against accuracy: does 'sure' mean 'right'?

    rows are dicts with at least "probs" ({letter: probability}) and "correct"
    (bool). Rows without usable probs are skipped, so the same function works
    on the in-memory rows of eval.py and on the CSV of calibration.py.
    """
    if threshold is None:
        threshold = sure_threshold()
    buckets = {label: {"n": 0, "correct": 0, "confidence": 0.0} for label in bucket_labels()}
    sure = {"n": 0, "correct": 0}
    doubt = {"n": 0, "correct": 0}
    counted = 0
    squared_error = 0.0
    for row in rows:
        probs = row.get("probs") or {}
        if len(probs) < 2:
            continue
        info = summarize(probs, threshold)
        p = info["prob"]
        correct = 1 if row.get("correct") else 0
        cell = buckets[bucket_label(p)]
        cell["n"] += 1
        cell["correct"] += correct
        cell["confidence"] += p
        group = sure if info["sure"] else doubt
        group["n"] += 1
        group["correct"] += correct
        counted += 1
        squared_error += (p - correct) ** 2
    ece = 0.0
    if counted:
        ece = sum((cell["n"] / counted) * abs(cell["correct"] / cell["n"] - cell["confidence"] / cell["n"])
                  for cell in buckets.values() if cell["n"])
    return {"threshold": threshold, "counted": counted, "buckets": buckets,
            "sure": sure, "doubt": doubt, "ece": ece,
            "brier": squared_error / counted if counted else 0.0}


def calibration_text(rows, threshold=None):
    """The calibration table as text, for eval.py and calibration.py."""
    report = calibration(rows, threshold)
    if not report["counted"]:
        return "Doubt detector: no letter probabilities in these results (run with --doubt)."
    lines = ["Doubt detector: when the model is sure, is it right?",
             f"  (a letter is sure at or above {report['threshold']:.0%} confidence)",
             "",
             f"{'confidence':13}{'n':>5}{'accuracy':>10}{'mean confidence':>17}"]
    for label in bucket_labels():
        cell = report["buckets"][label]
        if cell["n"]:
            lines.append(f"{label:13}{cell['n']:>5}{100 * cell['correct'] / cell['n']:>9.1f}%"
                         f"{100 * cell['confidence'] / cell['n']:>16.1f}%")
    lines.append("")
    for name, group in (("sure", report["sure"]), ("doubting", report["doubt"])):
        if group["n"]:
            lines.append(f"  {name:9} {group['correct']:>3}/{group['n']:<3} "
                         f"({100 * group['correct'] / group['n']:.1f}%) correct")
    lines.append(f"  ECE {report['ece']:.3f}, Brier {report['brier']:.3f} "
                 "(0 is perfect: confidence should match accuracy)")
    return "\n".join(lines)