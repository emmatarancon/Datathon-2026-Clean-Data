#!/usr/bin/env python3
"""Sweep rag2 parameters on the dev split only (never on val, rule R6).

    python3 rag2/tune.py --technique gate --limit 60        # on the board
    python3 rag2/tune.py --technique hybrid --limit 60
    python3 rag2/tune.py --technique dedup --limit 60
    python3 rag2/tune.py --technique gate --limit 5 --mock  # laptop, plumbing only

The tuning index is IN MEMORY, built from the vectors already stored in
data/index.db with ALL dev ids filtered out (train_fit only; embeddings are
never recomputed). Every configuration reports N, hits, Wilson 95%, latency
and letter counts; with N < 100 it says the difference is not significant.
Real measurements only on the board: --mock numbers are not measurements.
"""
import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.llm import answer_letter, format_problem
from rag2 import data, dedup, gate, hybrid, mock, protocol

TAUS_GATE = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
WS_HYBRID = (1.0, 0.85, 0.7, 0.5)
TAUS_DUP = (0.95, 0.97, 0.98, 0.99)
AVG_SECONDS_PER_CALL = 15.0  # rough board estimate for the time warning


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--technique", required=True, choices=("gate", "hybrid", "dedup"))
    parser.add_argument("--limit", type=int, default=60, help="dev problems (0 = all)")
    parser.add_argument("--mock", action="store_true",
                        help="simulated model and embeddings (NOT a measurement)")
    parser.add_argument("--index", default=data.INDEX_PATH)
    parser.add_argument("--budget", type=float, default=35.0)
    parser.add_argument("--input", default=data.TRAIN_PATH)
    return parser.parse_args()


def warn_time(calls, mock):
    minutes = calls * AVG_SECONDS_PER_CALL / 60.0
    if mock:
        print(f"[mock] {calls} simulated calls, seconds, not a measurement.")
    else:
        print(f"WARNING: this run makes about {calls} model calls, roughly "
              f"{minutes:.0f} min on the board at ~{AVG_SECONDS_PER_CALL:.0f} s/call. "
              "Ctrl+C now to cancel.")


def build_index(args, fit, dev_ids_all):
    """The in-memory tuning index: stored vectors, dev ids filtered out."""
    if args.mock:
        vectors = mock.mock_embed([format_problem(p) for p in fit])
        return data.MemIndex.from_problems(fit, vectors)
    if not __import__("os").path.exists(args.index):
        sys.exit(f"{args.index} not found: build_index.py runs on the board. "
                 "For laptop plumbing use --mock.")
    index = data.MemIndex.from_db(args.index, exclude_ids=dev_ids_all)
    print(f"index: {len(index)} rows loaded from {args.index} "
          f"(excluded {len(dev_ids_all)} dev ids)")
    return index


def make_runner(args):
    if args.mock:
        return mock.mock_answer
    return lambda problem, examples: answer_letter(problem, examples, timeout=60)


def answer_row(runner, problem, examples, config, technique, seconds=None):
    start = time.perf_counter()
    letter, _timings = runner(problem, examples)
    seconds = time.perf_counter() - start if seconds is None else seconds
    return {"technique": technique, "config": config, "id": problem["id"],
            "answer": letter, "label": problem.get("label", ""),
            "correct": letter == problem.get("label"), "seconds": round(seconds, 2),
            "examples_n": len(examples)}


def write_csv(path, rows):
    fields = ["technique", "config", "id", "answer", "label", "correct",
              "seconds", "examples_n"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\ndetails: {path}")


def report(config, rows, baseline_rows):
    """The standard block plus coverage and the paired line vs baseline."""
    protocol.describe(f"{config}", rows)
    if baseline_rows:
        flags = [r["correct"] for r in rows]
        base = [r["correct"] for r in baseline_rows]
        print("  " + protocol.paired_text(config, flags, "baseline", base))


def run_baseline(runner, dev, technique="tune"):
    rows = []
    for problem in dev:
        rows.append(answer_row(runner, problem, (), "baseline", technique))
    return rows


def tune_gate(args, runner, dev, index, baseline_rows):
    rows = []
    # the current behaviour, --rag 1 with the full example format, for reference
    rag1_rows = []
    for problem in dev:
        hits = index.search(problem, k=1, exclude_id=problem["id"])
        examples = [p for _s, p in hits]
        rag1_rows.append(answer_row(runner, problem, examples, "rag1", "gate"))
    report("rag1 (current full format)", rag1_rows, baseline_rows)
    rows += rag1_rows
    for tau in TAUS_GATE:
        config_rows = []
        covered = 0
        for problem in dev:
            examples, info = gate.select_examples(
                problem, index, exclude_id=problem["id"], tau=tau, budget_s=args.budget)
            covered += bool(info["coverage"])
            config_rows.append(answer_row(runner, problem, examples, f"tau={tau}", "gate"))
        report(f"gate tau={tau}", config_rows, baseline_rows)
        print(f"  cobertura: {covered}/{len(dev)} = {100.0 * covered / len(dev):.1f}%")
        rows += config_rows
    return rows


def tune_hybrid(args, runner, dev, index, baseline_rows):
    bm25, elapsed, cached = hybrid.build_bm25(index, cache_path=_bm25_cache())
    print(f"BM25 over {bm25.n} docs in {elapsed:.2f} s "
          f"({'pickle cache written' if cached else 'no cache written'})")
    # (a) retrieval quality without the model: top-1 neighbour per method
    methods = [("cosine", None)] + [(f"w={w}", w) for w in WS_HYBRID] + [("rrf", "rrf")]
    quality = {name: {"level_year": 0, "related": 0} for name, _w in methods}
    for problem in dev:
        cos = hybrid.cosine_pairs(index, problem, exclude_id=problem["id"])
        bm = hybrid.bm25_pairs(bm25, problem)
        tops = {"cosine": cos[:1]}
        for w in WS_HYBRID:
            tops[f"w={w}"] = hybrid.fuse(cos, bm, w=w)[:1]
        tops["rrf"] = hybrid.rrf(cos, bm)[:1]
        for name, pairs in tops.items():
            if not pairs:
                continue
            neighbor = index.rows[pairs[0][0]][1]
            info = hybrid.top1_related(problem, neighbor)
            quality[name]["level_year"] += info["same_level_year"]
            quality[name]["related"] += info["related"]
    print("\n(a) calidad de recuperacion en dev (vecino no 1):")
    for name, _w in methods:
        stats = quality[name]
        print(f"  {name:8} mismo nivel y ano: {stats['level_year']}/{len(dev)}"
              f" = {100.0 * stats['level_year'] / len(dev):.1f}%  "
              f"relacionado (o numeros J>=0.5): {stats['related']}/{len(dev)}"
              f" = {100.0 * stats['related'] / len(dev):.1f}%")
    # (b) final accuracy with the same budget as the gate
    rows = []
    warn_time(len(dev) * (len(WS_HYBRID) + 1), args.mock)
    for w in WS_HYBRID:
        config_rows = []
        for problem in dev:
            examples, _info = hybrid.select_examples(
                problem, index, bm25, exclude_id=problem["id"], w=w, budget_s=args.budget)
            config_rows.append(answer_row(runner, problem, examples, f"w={w}", "hybrid"))
        report(f"hybrid w={w}", config_rows, baseline_rows)
        rows += config_rows
    config_rows = []
    for problem in dev:
        examples, _info = hybrid.select_examples(
            problem, index, bm25, exclude_id=problem["id"], method="rrf",
            budget_s=args.budget)
        config_rows.append(answer_row(runner, problem, examples, "rrf", "hybrid"))
    report("hybrid rrf(k=60)", config_rows, baseline_rows)
    rows += config_rows
    return rows


def tune_dedup(args, runner, dev, index, baseline_rows):
    rows = []
    print(f"dedup: transfer checks over {len(dev)} dev problems (no model calls).")
    for tau_dup in TAUS_DUP:
        transferred = []
        reasons = {}
        for problem, base in zip(dev, baseline_rows):
            letter, info = dedup.try_transfer(problem, index,
                                              exclude_id=problem["id"],
                                              tau_dup=tau_dup)
            if letter is None:
                reasons[info["reason"]] = reasons.get(info["reason"], 0) + 1
                continue
            transferred.append({"technique": "dedup", "config": f"tau_dup={tau_dup}",
                                "id": problem["id"], "answer": letter,
                                "label": problem.get("label", ""),
                                "correct": letter == problem.get("label"),
                                "seconds": 0.0, "examples_n": 0,
                                "model_correct_same_problems": base["correct"]})
        n = len(transferred)
        hits = sum(1 for r in transferred if r["correct"])
        lo, hi = protocol.wilson(hits, n) if n else (0.0, 1.0)
        model_hits = sum(1 for r in transferred if r["model_correct_same_problems"])
        print(f"\n== dedup tau_dup={tau_dup} ==")
        print(f"cobertura: {n}/{len(dev)} = {100.0 * n / len(dev):.1f}%  "
              f"precision al transferir: {hits}/{n}", end="")
        if n:
            print(f" = {100.0 * hits / n:.1f}%  Wilson95=[{protocol.fmt_pct(lo)}, "
                  f"{protocol.fmt_pct(hi)}]  (modelo en esos mismos problemas: "
                  f"{model_hits}/{n} = {100.0 * model_hits / n:.1f}%)")
            if n < 30:
                print("  N < 30 en esta rama: no se puede concluir.")
            activated = lo > (model_hits / n)
            print(f"  regla de activacion (Wilson lo > modelo): "
                  f"{'ACTIVADA' if activated else 'NO activada'}")
        else:
            print()
        if reasons:
            top = sorted(reasons.items(), key=lambda kv: -kv[1])[:3]
            print("  razones de no transferir: " + ", ".join(f"{k}: {v}" for k, v in top))
        rows += transferred
    return rows


def _bm25_cache():
    import os
    os.makedirs(data.RAG2_DATA_DIR, exist_ok=True)
    return os.path.join(data.RAG2_DATA_DIR, "bm25.pkl")


def main():
    args = parse_args()
    if args.mock:
        mock.install()  # deterministic embeddings/answers, no servers
    problems = data.load_problems(args.input, languages=("eng",))
    fit, dev_all = data.split_train(problems)
    dev_ids_all = data.dev_ids(p["id"] for p in problems)
    dev = dev_all[:args.limit] if args.limit else dev_all
    print(f"train {len(problems)} problems (eng): train_fit {len(fit)}, "
          f"dev {len(dev_all)} (evaluating {len(dev)}), languages eng only.")
    index = build_index(args, fit, dev_ids_all)
    runner = make_runner(args)
    rows = []
    print("\n== baseline (no RAG) ==")
    warn_time(len(dev), args.mock)
    baseline_rows = run_baseline(runner, dev)
    report("baseline", baseline_rows, None)
    rows += baseline_rows
    if args.technique == "gate":
        warn_time(len(dev) * (len(TAUS_GATE) + 1), args.mock)
        rows += tune_gate(args, runner, dev, index, baseline_rows)
    elif args.technique == "hybrid":
        rows += tune_hybrid(args, runner, dev, index, baseline_rows)
    else:
        rows += tune_dedup(args, runner, dev, index, baseline_rows)
    write_csv(protocol.results_path(f"rag2-tune-{args.technique}"), rows)
    if args.mock:
        print("\n[mock] simulated numbers: plumbing only, measure on the board.")


if __name__ == "__main__":
    main()

