#!/usr/bin/env python3
"""The one confirmation pass over the full val set, through the official path.

    python3 rag2/final.py --gate-tau 0.85 --hybrid-w 0.7 --tau-dup 0.97   # board
    python3 rag2/final.py --limit 10 --mock                               # laptop

Runs each configuration once over data/val.jsonl through copilot.pipeline.solve
(the exact function predict.py and eval.py call), with the parameters FIXED by
rag2/tune.py on dev (never tuned here, rule R6). Configurations: baseline,
best gate, best hybrid, and the chosen combination with and without dedup.
Writes results/rag2-final-<hora>.csv and the summary results/rag2-resumen.md
with Wilson intervals, paired comparisons and latencies. Real measurements
only on the board: --mock numbers are plumbing, not measurements.
"""
import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot import pipeline
from copilot.config import CONFIG
from copilot.llm import format_problem
from rag2 import data, mock, protocol


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="val problems (0 = all 281)")
    parser.add_argument("--mock", action="store_true",
                        help="simulated model and embeddings (NOT a measurement)")
    parser.add_argument("--index", default=data.INDEX_PATH)
    parser.add_argument("--gate-tau", type=float, default=0.85,
                        help="the tau fixed by rag2/tune.py on dev")
    parser.add_argument("--hybrid-w", type=float, default=0.7,
                        help="the w fixed by rag2/tune.py on dev")
    parser.add_argument("--tau-dup", type=float, default=0.97,
                        help="the tau_dup fixed by rag2/tune.py on dev")
    parser.add_argument("--budget", type=float, default=35.0)
    parser.add_argument("--configs", default="baseline,gate,hybrid,gate+dedup,hybrid+dedup",
                        help="comma-separated subset of the configurations")
    return parser.parse_args()


def set_mode(mode, args):
    """Switch the pipeline into one rag2 configuration (in process)."""
    CONFIG["RAG_MODE"] = mode
    CONFIG["RAG2_TAU"] = str(args.gate_tau)
    CONFIG["RAG2_W"] = str(args.hybrid_w)
    CONFIG["RAG2_TAU_DUP"] = str(args.tau_dup)
    CONFIG["RAG2_BUDGET_S"] = str(args.budget)


def build_index(args):
    if args.mock:
        problems = data.load_train(languages=("eng",))
        fit, _dev = data.split_train(problems)
        return data.MemIndex.from_problems(fit, mock.mock_embed(
            [format_problem(p) for p in fit]))
    if not os.path.exists(args.index):
        sys.exit(f"{args.index} not found: build_index.py runs on the board. "
                 "For laptop plumbing use --mock.")
    from copilot.rag import Index
    return Index(args.index)


def run_config(name, mode, problems, index, args):
    """One pass of the official solve() over val with this configuration."""
    set_mode(mode, args)
    rows = []
    started = time.perf_counter()
    for n, problem in enumerate(problems, 1):
        result = pipeline.solve(problem, index=index, k=1, exclude_id=problem["id"])
        rows.append({"config": name, "id": problem["id"],
                     "language": problem.get("language", ""),
                     "answer": result["answer"], "label": problem.get("label", ""),
                     "correct": result["answer"] == problem.get("label"),
                     "seconds": result["seconds"], "status": result["status"],
                     "rag2_path": result.get("rag2_path", ""),
                     "retrieved": result.get("retrieved", "")})
        if n % 20 == 0 or n == len(problems):
            rate = (time.perf_counter() - started) / n
            print(f"  [{name}] {n}/{len(problems)} ~{rate:.1f} s/problem", flush=True)
    return rows


def summarize(name, rows, baseline_rows):
    lines = [f"## {name}", ""]
    n = len(rows)
    hits = sum(1 for r in rows if r["correct"])
    lo, hi = protocol.wilson(hits, n)
    seconds = [r["seconds"] for r in rows]
    over = sum(1 for s in seconds if s > float(CONFIG["TIME_LIMIT_SECONDS"]))
    lines.append(f"- N={n}, aciertos={hits}, acc={protocol.fmt_pct(hits / n)}, "
                 f"Wilson95=[{protocol.fmt_pct(lo)}, {protocol.fmt_pct(hi)}]")
    lines.append(f"- latencia: media {statistics.mean(seconds):.1f} s, "
                 f"mediana {statistics.median(seconds):.1f} s, "
                 f"peor {max(seconds):.1f} s, sobre el limite: {over}")
    letters = {l: sum(1 for r in rows if str(r["answer"]).strip().upper()[:1] == l)
               for l in "ABCDE"}
    truth = {l: sum(1 for r in rows if str(r["label"]).strip().upper()[:1] == l)
             for l in "ABCDE"}
    lines.append("- letras pred vs true: " + "  ".join(
        f"{l}:{letters[l]}/{truth[l]}" for l in "ABCDE"))
    if n < protocol.SIGNIFICANT_N:
        lines.append("- diferencia no significativa")
    if baseline_rows is not None and name != "baseline":
        counts = protocol.paired([r["correct"] for r in rows],
                                 [r["correct"] for r in baseline_rows])
        lines.append(f"- pareada vs baseline: solo {name} {counts['only_a']}, "
                     f"solo baseline {counts['only_b']}, ambas {counts['both']}, "
                     f"ninguna {counts['neither']}")
    dedup_rows = [r for r in rows if r["rag2_path"] == "dedup"]
    d_n = len(dedup_rows)
    if d_n:
        d_hits = sum(1 for r in dedup_rows if r["correct"])
        d_lo, d_hi = protocol.wilson(d_hits, d_n)
        lines.append(f"- camino dedup: {d_n}/{n} = {100.0 * d_n / n:.1f}% de problemas, "
                     f"precision {d_hits}/{d_n} = {100.0 * d_hits / d_n:.1f}%, "
                     f"Wilson95=[{protocol.fmt_pct(d_lo)}, {protocol.fmt_pct(d_hi)}]")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    modes = {"baseline": "baseline", "gate": "gate", "hybrid": "hybrid",
             "gate+dedup": "gate+dedup", "hybrid+dedup": "hybrid+dedup"}
    for name in names:
        if name not in modes:
            sys.exit(f"unknown config {name!r}; choose from {sorted(modes)}")
    if args.mock:
        mock.install()
    problems = data.load_val(languages=("eng",))
    if args.limit:
        problems = problems[:args.limit]
    print(f"val: {len(problems)} problems (eng).")
    est = len(names) * len(problems) * 15.0 / 60.0
    if args.mock:
        print(f"[mock] {len(names)} simulated passes, seconds, not a measurement.")
    else:
        print(f"WARNING: {len(names)} passes x {len(problems)} problems, roughly "
              f"{est:.0f} min on the board at ~15 s/call. Ctrl+C now to cancel.")
    index = build_index(args)
    os.environ["RAG2_PATHS_FILE"] = protocol.results_path("rag2-paths-final")
    all_rows = []
    summary = ["# rag2: confirmacion en val (via oficial, copilot.pipeline.solve)", "",
               f"Parametros fijados por rag2/tune.py en dev: gate tau={args.gate_tau}, "
               f"hybrid w={args.hybrid_w}, tau_dup={args.tau_dup}, presupuesto {args.budget} s.",
               f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}  Mock: {'SI (no son medidas)' if args.mock else 'no'}"
               "", ""]
    baseline_rows = None
    for name in names:
        mode = modes[name]
        print(f"\n== {name} (RAG_MODE={mode}) ==")
        rows = run_config(name, mode, problems, index if mode != "baseline" else None, args)
        all_rows += rows
        if name == "baseline":
            baseline_rows = rows
        summary.append(summarize(name, rows, baseline_rows))
    fields = ["config", "id", "language", "answer", "label", "correct", "seconds",
              "status", "rag2_path", "retrieved"]
    final_csv = protocol.results_path("rag2-final")
    with open(final_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    summary_path = os.path.join(os.path.dirname(final_csv), "rag2-resumen.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary))
    print(f"\ndetails: {final_csv}\nsummary: {summary_path}")
    if args.mock:
        print("[mock] simulated numbers: plumbing only, measure on the board.")


if __name__ == "__main__":
    main()
