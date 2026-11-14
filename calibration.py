#!/usr/bin/env python3
"""Is the copilot's confidence honest? The doubt detector's calibration.

    python3 eval.py --doubt --limit 50 --name doubt1
    python3 calibration.py results/eval-<time>-doubt1.csv

Reads the per-problem CSV that eval.py writes with --doubt (columns pA..pE)
and answers the question the doubt detector exists for: when the model is
sure, how often is it actually right? The confidence of each answer goes into
a bucket, and each bucket gets its accuracy: a calibrated model has them
equal, a guessing model is "sure" and wrong just as often.
"""
import argparse
import csv

from copilot.confidence import LETTERS, calibration_text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="a results/eval-*.csv written by eval.py --doubt")
    p.add_argument("--sure", type=float, default=None, help="override the sure threshold (0-1)")
    args = p.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        probs = {}
        for letter in LETTERS:
            raw = (row.get(f"p{letter}") or "").strip()
            if raw:
                try:
                    probs[letter] = float(raw)
                except ValueError:
                    probs = {}
                    break
        if probs:  # rows without probabilities (timeouts, old server) don't count
            parsed.append({"probs": probs, "correct": row.get("correct", "").strip() in ("1", "True"),
                           "answer": row.get("answer"), "label": row.get("label")})
    if not parsed:
        raise SystemExit(f"{args.csv} has no pA..pE values. Run eval.py --doubt first.")
    print(f"{args.csv}: {len(rows)} problems, {len(parsed)} with letter probabilities\n")
    print(calibration_text(parsed, threshold=args.sure))


if __name__ == "__main__":
    main()