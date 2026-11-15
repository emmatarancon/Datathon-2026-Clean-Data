#!/usr/bin/env python3
"""Phase 2 confirmation: ONE pass over val per winning correction.

    python3 bias/final.py --configs baseline,calibrate,permute \
        --calibration-file results/bias-calibration-contextual.json   # board
    python3 bias/final.py --limit 6 --mock --configs baseline,calibrate  # laptop

Runs each configuration once over data/val.jsonl through copilot.pipeline.solve
(the exact official path), with DEBIAS_MODE switched per configuration and the
parameters fixed beforehand (never tuned on val, rule R6). Reports N, hits,
Wilson 95%, letters predicted vs true with chi-square, latency mean/worst,
reserve answers and the paired comparison with the baseline. With N < 100 it
prints 'diferencia no significativa'. Writes results/bias-final-<hora>.md.
"""
import argparse
import csv
import json
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__))))

from copilot import pipeline
from copilot.config import CONFIG
from bias import mock, stats

MODES = ("off", "calibrate", "permute", "calibrate+permute")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/val.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="val problems (0 = all 281)")
    parser.add_argument("--configs", default="baseline,calibrate,permute",
                        help="comma-separated: baseline, calibrate, permute, calibrate+permute")
    parser.add_argument("--calibration-file", default="",
                        help="the bias JSON fixed beforehand (needed by calibrate)")
    parser.add_argument("--mock", action="store_true", help="simulated model (not a measurement)")
    return parser.parse_args()


def set_mode(name, args):
    """Switch the pipeline into one configuration (in process, env-like)."""
    mode = "off" if name == "baseline" else name
    if mode not in MODES:
        sys.exit(f"unknown config {name!r}; choose from baseline,{', '.join(MODES[1:])}")
    CONFIG["DEBIAS_MODE"] = mode
    CONFIG["CALIBRATION_FILE"] = args.calibration_file or ""


def run_config(name, problems, args):
    """One pass of the official solve() over val with this configuration."""
    set_mode(name, args)
    rows = []
    started = time.perf_counter()
    for n, problem in enumerate(problems, 1):
        result = pipeline.solve(problem)
        rows.append({"config": name, "id": problem["id"],
                     "answer": result.get("answer", ""), "label": problem.get("label", ""),
                     "correct": result.get("answer") == problem.get("label"),
                     "seconds": result.get("seconds", 0.0),
                     "status": result.get("status", ""),
                     "debias": result.get("debias", "")})
        if n % 20 == 0 or n == len(problems):
            print(f"  [{name}] {n}/{len(problems)} "
                  f"~{(time.perf_counter() - started) / n:.1f} s/problem", flush=True)
    return rows


def summarize(name, rows, baseline_rows):
    """The standard block plus the paired line and the letters moved."""
    lines = stats.describe(name, rows)
    if baseline_rows is not None and name != "baseline":
        counts = stats.paired([r["correct"] for r in rows],
                              [r["correct"] for r in baseline_rows])
        lines.append(f"- pareada vs baseline: solo {name} {counts['only_a']}, "
                     f"solo baseline {counts['only_b']}, ambas {counts['both']}, "
                     f"ninguna {counts['neither']}")
        moved = {letter: stats.letter_counts(rows)[letter] -
                 stats.letter_counts(baseline_rows)[letter] for letter in stats.LETTERS}
        lines.append("- letras movidas frente al baseline: " +
                     "  ".join(f"{l}:{moved[l]:+d}" for l in stats.LETTERS if moved[l]))
    debias_rows = [r for r in rows if r.get("debias")]
    if debias_rows:
        lines.append(f"- filas corregidas por el gancho: {len(debias_rows)}/{len(rows)}")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.mock:
        mock.install()
    with open(args.input) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    problems = [p for p in problems if p.get("language") == "eng"]
    if args.limit:
        problems = problems[:args.limit]
    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    print(f"val: {len(problems)} problems (eng); configs: {names}")
    est = len(names) * len(problems) * 9.0 / 60.0
    if args.mock:
        print(f"[mock] {len(names)} simulated passes: seconds, NOT a measurement.")
    else:
        print(f"WARNING: {len(names)} passes x {len(problems)} problems "
              f"≈ {est:.0f} min on the board. Ctrl+C now to cancel.")
    saved = dict(CONFIG)
    all_rows, report = [], ["# bias: confirmacion en val (via oficial solve)", "",
                            f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}  "
                            f"Mock: {'SI (no son medidas)' if args.mock else 'no'}",
                            f"Parametros fijados de antemano: CALIBRATION_FILE="
                            f"{args.calibration_file or '(vacia)'}", "", ""]
    baseline_rows = None
    try:
        for name in names:
            print(f"\n== {name} (DEBIAS_MODE={'off' if name == 'baseline' else name}) ==")
            rows = run_config(name, problems, args)
            all_rows += rows
            if name == "baseline":
                baseline_rows = rows
            report.append(summarize(name, rows, baseline_rows))
    finally:
        CONFIG.clear()
        CONFIG.update(saved)
    report.append("## Aviso de honestidad")
    report.append("Un modelo casi uniforme bien calibrado acierta cerca del azar "
                  "(~20-25%): si el acierto sigue ahi tras quitar el sesgo, es un "
                  "resultado valido para el analisis de errores, no un fallo del trabajo.")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = f"results/bias-final-{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "id", "answer", "label",
                                               "correct", "seconds", "status", "debias"])
        writer.writeheader()
        writer.writerows(all_rows)
    md_path = f"results/bias-final-{stamp}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(report))
    print(f"\ndetails: {csv_path}\nsummary: {md_path}")
    if args.mock:
        print("[mock] simulated numbers: plumbing only, measure on the board.")


if __name__ == "__main__":
    main()