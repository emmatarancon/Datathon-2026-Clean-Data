#!/usr/bin/env python3
"""Official entry point: answer every problem with one letter, A to E.

    python3 predict.py --input problems.jsonl --output predictions.csv
    python3 predict.py --input problems.jsonl --output predictions.csv --rag 1

Input: one JSON object per line with at least "id" and "text", plus
"options" ({"A": ..., "E": ...}) and "language" when available.
Output: a CSV with id, language, answer, seconds, status (plus pA..pE,
top_prob and verdict when the doubt detector is on: --doubt or DOUBT_DETECTOR=1).

Rules: runs offline, at most 60 seconds per problem, deterministic.
"""
import argparse
import csv
import json
import os
import sys

from copilot.confidence import LETTERS, enabled as doubt_enabled
from copilot.pipeline import solve


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rag", type=int, default=0, metavar="K", help="add K similar solved problems")
    p.add_argument("--index", default="data/index.db")
    p.add_argument("--doubt", action="store_true",
                   help="add the model's letter probabilities (pA..pE, top_prob, verdict); "
                        "DOUBT_DETECTOR=1 in config.env turns it on without this flag")
    args = p.parse_args()

    index = None
    if args.rag:
        if not os.path.exists(args.index):
            sys.exit(f"{args.index} not found. Build it first: python3 build_index.py")
        from copilot.rag import Index
        index = Index(args.index)

    with open(args.input) as f:
        problems = [json.loads(line) for line in f if line.strip()]

    doubt = args.doubt or doubt_enabled()
    with open(args.output, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["id", "language", "answer", "seconds", "status"]
                        + ([f"p{l}" for l in LETTERS] + ["top_prob", "verdict"] if doubt else []))
        for n, problem in enumerate(problems, 1):
            result = solve(problem, index=index, k=args.rag,
                           doubt=True if args.doubt else None)
            row = [problem["id"], problem.get("language", ""), result["answer"],
                   result["seconds"], result["status"]]
            if doubt:  # empty cells when the server sent no usable logprobs
                probs = result.get("probs") or {}
                confidence = result.get("confidence") or {}
                row += [f"{probs[l]:.4f}" if l in probs else "" for l in LETTERS]
                row += [f"{confidence['prob']:.4f}" if confidence else "",
                        confidence.get("verdict", "")]
            writer.writerow(row)
            out.flush()
            print(f"[{n}/{len(problems)}] {problem['id']} -> {result['answer']} "
                  f"in {result['seconds']:.1f} s ({result['status']})", file=sys.stderr)


if __name__ == "__main__":
    main()
