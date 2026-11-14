"""Summaries for the letter-bias experiments (reason/calibration.py output).

eval.py already reports accuracy, latency and the letter counts; what it does
not do is the Wilson interval, nor comparing two runs' predicted letters side
by side, so this tiny reader does both from its per-problem CSVs:

    python3 reason/eval_summary.py results/eval-*-dev-*.csv
    python3 reason/eval_summary.py results/eval-20260101-000000-baseline.csv results/eval-20260101-120000-calibrated.csv
"""
import csv
import math
import os
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from reason.calibration import LETTERS, wilson_interval     # noqa: E402


def summarize(path):
    """Accuracy + Wilson, latency, letters predicted vs true from one eval CSV."""
    with open(path) as f:
        rows = [row for row in csv.DictReader(f) if row.get("answer")]
    hits = [row["answer"] == row["label"] for row in rows]
    seconds = [float(row["seconds"]) for row in rows]
    predicted = Counter(row["answer"] for row in rows)
    truth = Counter(row["label"] for row in rows)
    low, high = wilson_interval(sum(hits), len(hits))
    print(f"\n{os.path.basename(path)}")
    print(f"  N {len(rows)}, hits {sum(hits)}, accuracy {100 * sum(hits) / len(rows):.1f}% "
          f"(Wilson 95%: {100 * low:.1f}%-{100 * high:.1f}%)")
    print(f"  seconds: mean {sum(seconds) / len(seconds):.1f}, worst {max(seconds):.1f}")
    print("  letters  " + "  ".join(f"{l} {predicted.get(l, 0):>3}/{truth.get(l, 0):<3}" for l in LETTERS)
          + "   (predicted/true)")
    return {"path": path, "n": len(rows), "hits": sum(hits), "predicted": predicted,
            "truth": truth, "mean": sum(seconds) / len(seconds), "worst": max(seconds)}


def compare(first, second):
    """Which letters moved between two runs of the same problems."""
    moved = {}
    for letter in LETTERS:
        delta = second["predicted"].get(letter, 0) - first["predicted"].get(letter, 0)
        if delta:
            moved[letter] = delta
    print(f"\nLetters moved from {os.path.basename(first['path'])} "
          f"to {os.path.basename(second['path'])}: "
          + (", ".join(f"{l} {delta:+d}" for l, delta in sorted(moved.items())) or "none")
          + f" | accuracy {first['hits']}/{first['n']} -> {second['hits']}/{second['n']}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    summaries = [summarize(path) for path in sys.argv[1:]]
    for first, second in zip(summaries, summaries[1:]):
        compare(first, second)


if __name__ == "__main__":
    main()
