#!/usr/bin/env python3
"""Measure accuracy and latency on the board, the way the official run does.

    python3 eval.py --limit 50                    # baseline on 50 validation problems
    python3 eval.py --limit 50 --rag 1 --name rag1
    python3 eval.py --languages cat --limit 100
    python3 eval.py --limit 50 --strategy vote_safe --name improved

Every run writes per-problem results to results/ and adds one line to
results/ablations.csv, so you can build your ablation table as you go.
"""
import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import time
from collections import Counter, defaultdict

from copilot.confidence import LETTERS, calibration_text, enabled as doubt_enabled
from copilot.config import CONFIG
from copilot.led_matrix import enabled as led_enabled, show_cross, show_probs
from copilot.pipeline import TIME_LIMIT, solve


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def table(title, groups):
    print(f"\n{title}")
    for key in sorted(groups, key=str):
        hits = groups[key]
        print(f"  {str(key):12} {sum(hits):4}/{len(hits):<4} {100 * sum(hits) / len(hits):5.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/val.jsonl")
    p.add_argument("--limit", type=int, default=50, help="0 means every problem")
    p.add_argument("--languages", default="", help="e.g. eng,cat")
    p.add_argument("--rag", type=int, default=0, metavar="K")
    p.add_argument("--index", default="data/index.db")
    p.add_argument("--name", default="", help="a label for this experiment in ablations.csv")
    p.add_argument("--doubt", action="store_true",
                   help="record the letter probabilities (pA..pE) and print the calibration table; "
                        "DOUBT_DETECTOR=1 in config.env turns it on without this flag")
    p.add_argument("--led", action="store_true",
                   help="show pA..pE live on the board's 8x13 LED matrix (needs the led_app "
                        "app running; LED_MATRIX=1 in config.env turns it on without this flag)")
    p.add_argument("--strategy", default=None,
                   help="solving strategy: auto, baseline (default), vote, vote_safe, "
                        "fewshot, calibrated, fallback")
    args = p.parse_args()
    led = args.led or led_enabled()  # the matrix needs the probabilities too
    doubt = args.doubt or doubt_enabled() or led

    with open(args.input) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    languages = set(filter(None, args.languages.split(",")))
    if languages:
        problems = [x for x in problems if x["language"] in languages]
    random.Random(42).shuffle(problems)  # the same sample every run, so runs compare
    # NOTE: --limit counts problems with VALID probabilities (pA..pE > 0).
    # Problems whose probability reading comes back zero/empty are discarded
    # on the fly (see the loop): warned on the console, shown as a cross on
    # the matrix when --led, never counted in any statistic or in the quota.
    # Problems with empty options are NOT discarded: if the model still
    # returns valid probabilities (the options may be inside the statement
    # text), they are evaluated like any other problem.

    index = None
    if args.rag:
        from copilot.rag import Index
        index = Index(args.index)

    os.makedirs("results", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = args.name or (f"rag{args.rag}" if args.rag else "baseline")
    detail_path = f"results/eval-{stamp}-{name}.csv"

    rows = []
    evaluated = 0   # problems with valid probabilities: the only ones counted
    discarded = 0   # zero/empty probabilities (no reliable reading): excluded
    with open(detail_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["id", "language", "level", "has_options", "label", "answer",
                         "correct", "seconds", "status", "prompt_tokens", "retrieved"]
                        + ([f"p{l}" for l in LETTERS] + ["top_prob", "verdict"] if doubt else []))
        for problem in problems:
            if args.limit and evaluated >= args.limit:
                break  # --limit counts problems with valid probs; discards never fill it
            result = solve(problem, index=index, k=args.rag, exclude_id=problem["id"],
                           doubt=True if (args.doubt or led) else None,
                           strategy=args.strategy)
            probs = result.get("probs") or {}
            has_valid_probs = any(v > 0 for v in probs.values())
            if not (led or args.doubt or doubt_enabled()):
                # no probabilities were requested (--led/--doubt off): there is
                # no reading to judge, so the problem is kept whatever its
                # options look like — the official behaviour is untouched
                has_valid_probs = True
            if not has_valid_probs:
                discarded += 1
                print(f"[descartado] {problem['id']} ({problem.get('language', '')}): "
                      "probabilidades a 0 (sin lectura fiable), excluido de estadísticas",
                      flush=True)
                if led:  # the matrix shows a cross instead of empty bars
                    show_cross()
                continue
            evaluated += 1
            correct = result["answer"] == problem["label"]
            if led:  # live bars on the matrix; fails silently when it is absent
                show_probs(probs, answer=result["answer"], correct=correct)
            row = {**problem, **result, "correct": correct, "has_options": bool(problem.get("options"))}
            rows.append(row)
            values = [problem["id"], problem["language"], problem["level"], row["has_options"],
                      problem["label"], result["answer"], int(correct), result["seconds"],
                      result["status"], result["prompt_tokens"], result["retrieved"]]
            if doubt:  # empty cells when the server sent no usable logprobs
                probs = result.get("probs") or {}
                confidence = result.get("confidence") or {}
                values += [f"{probs[l]:.4f}" if l in probs else "" for l in LETTERS]
                values += [f"{confidence['prob']:.4f}" if confidence else "",
                           confidence.get("verdict", "")]
            writer.writerow(values)
            out.flush()
            running = 100 * sum(r["correct"] for r in rows) / len(rows)
            print(f"[{evaluated}/{args.limit if args.limit else evaluated}] {problem['id']} "
                  f"{problem['language']}: {result['answer']} (truth {problem['label']}) "
                  f"{result['seconds']:.1f} s, running accuracy {running:.1f}%", flush=True)

    if not rows:
        # every problem came back with zero/empty probabilities: nothing to
        # statisticar, and the summary below would divide by zero
        print(f"\nSin problemas con probabilidades válidas: {discarded} descartes y "
              "ninguna fila que estadisticar. Revisa el servidor de modelos y "
              "vuelve a ejecutar; no se escribe ninguna estadística nueva.")
        print(f"Details: {detail_path} (sin filas)\nAblation log: sin fila nueva")
        return

    hits = [r["correct"] for r in rows]
    seconds = [r["seconds"] for r in rows]
    floor = sum(r["label"] == CONFIG["FALLBACK_ANSWER"] for r in rows)
    over = sum(s > TIME_LIMIT for s in seconds)
    accuracy = 100 * sum(hits) / len(hits)

    print("\n" + "=" * 60)
    print(f"Accuracy: {sum(hits)}/{len(hits)} = {accuracy:.1f}%")
    print(f"  floor, always {CONFIG['FALLBACK_ANSWER']}: {100 * floor / len(rows):.1f}%   random guess: 20.0%")
    print(f"Latency: mean {statistics.mean(seconds):.1f} s, median {statistics.median(seconds):.1f} s, "
          f"worst {max(seconds):.1f} s, over {TIME_LIMIT:.0f} s: {over}")
    print(f"Status: {dict(Counter(r['status'] for r in rows))}")
    print(f"Descartados por opciones ilegibles (sin A-E): {discarded} — "
          "excluidos de todas las estadísticas y del cupo de --limit")

    for field, title in (("language", "By language"), ("level", "By level"),
                         ("has_options", "By whether OCR found the options")):
        groups = defaultdict(list)
        for r in rows:
            groups[r[field]].append(r["correct"])
        table(title, groups)

    predicted = Counter(r["answer"] for r in rows)
    truth = Counter(r["label"] for r in rows)
    print("\nLetters   " + "  ".join(f"{l}:{predicted[l]:>3}/{truth[l]:<3}" for l in "ABCDE"))
    print("          (predicted/true; a model that loves one letter shows up here)")

    if doubt:  # the doubt detector's question: does sure mean right?
        print()
        print(calibration_text(rows))

    log = "results/ablations.csv"
    new = not os.path.exists(log)
    with open(log, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["time", "name", "commit", "input", "languages", "n", "rag_k",
                             "accuracy", "mean_s", "worst_s", "over_limit"])
        writer.writerow([stamp, name, git_commit(), args.input, args.languages or "all", len(rows),
                         args.rag, f"{accuracy:.1f}", f"{statistics.mean(seconds):.1f}",
                         f"{max(seconds):.1f}", over])
    print(f"\nDetails: {detail_path}\nAblation log: {log}")


if __name__ == "__main__":
    main()
