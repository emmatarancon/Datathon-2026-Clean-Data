#!/usr/bin/env python3
"""Phase 1: measure WHERE the letter bias (A/D) comes from. Changes nothing.

    python3 bias/diagnose.py --limit 60          # on the board (~15-25 min)
    python3 bias/diagnose.py --limit 8 --mock    # laptop, plumbing only

Works on the deterministic dev split of train (int(sha1(id)[:8],16) % 5 == 0),
problems WITH read options for the main pass, plus a small subset WITHOUT
options. Every claim prints its numbers:

D1 artifact separation (H3): letters on status=ok rows vs fallback rows;
   letters with options vs without options.
D2 true letter distribution in train/dev, predicted letters with chi-square
   (df=4, critical 9.49 at 5%) against uniformity and against the truth.
D3 5x5 confusion matrix and precision/recall per letter.
D4 permutation sensitivity: cyclic shifts of the options - letter stickiness
   vs content consistency (H1/H2).
D5 logprob distribution: entropy vs ln 5, share of problems above 50% on one
   letter, mean probability per letter (corrigible prior or not).
D6 cost: latency mean/worst and prompt_n reuse across option shifts.

Writes results/bias-diagnose-<hora>.csv and results/bias-diagnose-<hora>.md.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__))))

from copilot import pipeline
from copilot.confidence import LETTERS
from copilot.llm import build_messages
from bias import mock, permute as permute_mod, stats
from bias import calibration as bias_calibration
from bias.calibration import renormalize

LN5 = math.log(5.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/train.jsonl")
    parser.add_argument("--limit", type=int, default=60, help="dev problems WITH options")
    parser.add_argument("--permute-limit", type=int, default=30)
    parser.add_argument("--no-options-limit", type=int, default=20)
    parser.add_argument("--shifts", default="1,2,3", help="cyclic shifts for D4")
    parser.add_argument("--mock", action="store_true", help="simulated model (not a measurement)")
    args = parser.parse_args()
    args.shift_list = [int(s) for s in args.shifts.split(",") if s.strip()]
    return args


def is_dev(problem_id):
    """The deterministic dev rule, same as the rest of the reason work."""
    return int(hashlib.sha1(problem_id.encode()).hexdigest()[:8], 16) % 5 == 0


def load_dev(input_path):
    with open(input_path) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    dev = [p for p in problems if p.get("language") == "eng" and is_dev(p["id"])]
    with_options = [p for p in dev if p.get("options")]
    without_options = [p for p in dev if not p.get("options")]
    return problems, dev, with_options, without_options


def entropy(probs):
    """Shannon entropy of the letter probabilities (0..ln 5)."""
    total = sum(probs.get(l, 0.0) for l in LETTERS)
    if not total > 0.0:
        return None
    value = 0.0
    for letter in LETTERS:
        p = probs.get(letter, 0.0) / total
        if p > 0.0:
            value -= p * math.log(p)
    return value


def base_row(kind, problem, shift=0):
    return {"kind": kind, "id": problem["id"], "shift": shift, "status": "",
            "answer": "", "label": problem.get("label", ""), "correct": "",
            "seconds": 0.0, "prompt_n": 0,
            **{f"p{l}": "" for l in LETTERS}, "entropy": "",
            "sticky": "", "consistent": "", "content": ""}


def solve_row(problem):
    """One official solve() with the letter probabilities (D1/D5/D6 data)."""
    row = base_row("main", problem)
    start = time.perf_counter()
    result = pipeline.solve(problem, doubt=True)
    probs = result.get("probs") or {}
    row.update({"status": result.get("status", ""),
                "answer": result.get("answer", ""),
                "correct": result.get("answer") == problem.get("label"),
                "seconds": result.get("seconds", round(time.perf_counter() - start, 2)),
                "prompt_n": result.get("prompt_tokens", 0)})
    for letter in LETTERS:
        row[f"p{letter}"] = f"{probs[letter]:.4f}" if letter in probs else ""
    if probs:
        row["entropy"] = f"{entropy(probs):.3f}"
    return row


def no_options_row(problem):
    """One solve() of a problem whose options OCR could not read."""
    row = base_row("no-options", problem)
    result = pipeline.solve(problem, doubt=True)
    row.update({"status": result.get("status", ""), "answer": result.get("answer", ""),
                "correct": result.get("answer") == problem.get("label"),
                "seconds": result.get("seconds", 0.0),
                "prompt_n": result.get("prompt_tokens", 0)})
    return row


def warn_time(args, n_main, n_perm):
    requests = n_main + n_perm * (1 + len(args.shift_list))
    minutes = (n_main * 9.0 + n_perm * (1 + len(args.shift_list)) * 12.0) / 60.0
    if args.mock:
        print(f"[mock] about {requests} simulated requests: seconds, NOT a measurement.")
    else:
        print(f"WARNING: about {requests} model requests ≈ {minutes:.0f} min on the "
              "board. Ctrl+C now to cancel.")


def permutation_rows(problem, args, main_row):
    """D4: the letter answer re-asked with the options cyclically shifted.

    Returns (rows, summary): summary holds the sticky/consistent counts over
    the measured shifts (letter stickiness vs content consistency).
    """
    rows = []
    original_answer = stats.letter(main_row["answer"])
    n = len(problem["options"])
    sticky_letter = consistent_content = used = 0
    for shift in args.shift_list:
        row = base_row("permute", problem, shift)
        shifted = dict(problem,
                       options=permute_mod.shifted_options(problem["options"], shift))
        start = time.perf_counter()
        grouped, timings = bias_calibration.letter_logprobs(build_messages(shifted), timeout=12.0)
        row["seconds"] = round(time.perf_counter() - start, 2)
        row["prompt_n"] = timings.get("prompt_n", 0)
        if not grouped:
            row["status"] = "no-logprobs"
            rows.append(row)
            continue
        used += 1
        row["status"] = "ok"
        probs = renormalize(grouped)
        for letter in LETTERS:
            row[f"p{letter}"] = f"{probs.get(letter, 0.0):.4f}"
        row["entropy"] = f"{entropy(probs):.3f}"
        answered = max(LETTERS, key=lambda l: probs.get(l, 0.0))
        content = permute_mod.content_letter(answered, shift, n)
        sticky = answered == original_answer
        consistent = (problem["options"].get(content) ==
                      problem["options"].get(original_answer))
        sticky_letter += int(sticky)
        consistent_content += int(consistent)
        row.update({"answer": answered, "content": content,
                    "sticky": int(sticky), "consistent": int(consistent)})
        rows.append(row)
    return rows, {"sticky": sticky_letter, "consistent": consistent_content, "used": used}


def analyses_d1_d3(args, main_rows, no_options_rows, fallback_letter):
    """D1 (artifact separation), D2 (distributions) and D3 (confusion matrix)."""
    lines = []
    ok_rows = [r for r in main_rows if r["status"] == "ok"]
    fallback_rows = [r for r in main_rows if r["status"] != "ok"]
    lines.append("## D1 separacion de artefactos (H3)")
    lines.append(f"- filas status=ok: {len(ok_rows)}/{len(main_rows)}; "
                 f"respuestas de reserva (status!=ok): {len(fallback_rows)} "
                 f"({100.0 * len(fallback_rows) / max(len(main_rows), 1):.1f}%), "
                 f"todas de la letra {fallback_letter} "
                 f"-> letras de reserva: {stats.letter_counts(fallback_rows)}")
    lines.append(f"- letras con status=ok: {stats.letter_counts(ok_rows)}")
    lines.append(f"- letras de TODAS las filas (lo que veria eval.py): "
                 f"{stats.letter_counts(main_rows)}")
    lines.append(f"- problemas SIN opciones leidas (N={len(no_options_rows)}): "
                 f"{stats.letter_counts(no_options_rows)}")
    lines.append("")
    lines.append("## D2 distribucion verdadera y chi-cuadrado (df=4, critico 9.49)")
    lines.append(f"- etiquetas verdaderas del dev medido: "
                 f"{stats.letter_counts(main_rows, key='label')}")
    lines.append("")
    lines.append("## D3 matriz de confusion y precision/recall")
    matrix, per_letter = stats.confusion(ok_rows or main_rows)
    lines.append("```")
    lines.append("true\\pred " + " ".join(f"{l:>7}" for l in LETTERS))
    for true_l in LETTERS:
        lines.append(f"{true_l:>8}  " + " ".join(f"{matrix[true_l][p]:>7}" for p in LETTERS))
    lines.append("```")
    for letter, info in per_letter.items():
        precision = f"{100 * info['precision']:.0f}%" if info["precision"] is not None else "-"
        recall = f"{100 * info['recall']:.0f}%" if info["recall"] is not None else "-"
        lines.append(f"- {letter}: pred {info['predicted']}, true {info['true']}, "
                     f"precision {precision}, recall {recall}")
    lines.append("")
    return lines


def analyses_d4_d6(args, main_rows, perm_rows, perm_summaries):
    """D4 (stickiness vs consistency), D5 (entropy) and D6 (cost/prefix reuse)."""
    lines = []
    ok_rows = [r for r in main_rows if r["status"] == "ok"]
    lines.append("## D4 sensibilidad a la permutacion (H1 frente a H2)")
    total_shifts = sum(s["used"] for s in perm_summaries)
    if total_shifts:
        sticky = 100.0 * sum(s["sticky"] for s in perm_summaries) / total_shifts
        consistent = 100.0 * sum(s["consistent"] for s in perm_summaries) / total_shifts
        lines.append(f"- pegajosidad de LETRA (misma letra tras el desplazamiento): "
                     f"{sticky:.1f}% sobre {total_shifts} peticiones desplazadas")
        lines.append(f"- consistencia de CONTENIDO (mismo contenido elegido): {consistent:.1f}%")
        if sticky >= 60.0 and consistent <= 40.0:
            lines.append("- lectura: letra pegajosa y contenido inconsistente => el modelo "
                         "ignora el contenido (H1/H2); la permutacion puede corregirlo.")
        elif consistent >= 70.0:
            lines.append("- lectura: contenido consistente => el modelo mira el contenido "
                         "y el sesgo es leve; la permutacion aportaria poco.")
        else:
            lines.append("- lectura: intermedio; decide con la matriz y D5.")
    else:
        lines.append("- sin peticiones desplazadas medibles.")
    lines.append("")
    lines.append("## D5 distribucion de probabilidades (prior corregible?)")
    entropies = [float(r["entropy"]) for r in ok_rows if r["entropy"]]
    if entropies:
        mean_entropy = statistics.mean(entropies)
        share_above = 100.0 * sum(
            1 for r in ok_rows
            if max((float(r[f"p{l}"]) for l in LETTERS if r[f"p{l}"]), default=0.0) > 0.5
        ) / len(entropies)
        mean_probs = {l: statistics.mean([float(r[f"p{l}"]) for r in ok_rows if r[f"p{l}"]]
                                         or [0.0]) for l in LETTERS}
        lines.append(f"- entropia media {mean_entropy:.3f} de un maximo {LN5:.3f} (ln 5): "
                     f"{100.0 * mean_entropy / LN5:.0f}% del maximo")
        lines.append(f"- problemas con una letra por encima del 50%: {share_above:.1f}%")
        lines.append("- probabilidad media por letra: " +
                     "  ".join(f"{l}:{p:.3f}" for l, p in mean_probs.items()))
        lines.append("- lectura: " + ("casi uniforme => prior de letra leve y CORREGIBLE "
                                      "con calibracion (o acierto esperable ~ azar)."
                                      if mean_entropy / LN5 > 0.85 else
                                      "distribuciones marcadas => si el modelo esta seguro "
                                      "de letras equivocadas, la calibracion ayuda poco."))
    else:
        lines.append("- sin logprobs utilizables en la corrida principal.")
    lines.append("")
    lines.append("## D6 coste y reutilizacion del prefijo")
    seconds = [r["seconds"] for r in main_rows if r["seconds"] is not None]
    if seconds:
        lines.append(f"- peticion principal: media {statistics.mean(seconds):.1f} s, "
                     f"peor {max(seconds):.1f} s")
    perm_seconds = [r["seconds"] for r in perm_rows if r["status"] == "ok"]
    if perm_seconds:
        lines.append(f"- peticion desplazada: media {statistics.mean(perm_seconds):.1f} s, "
                     f"peor {max(perm_seconds):.1f} s")
        lines.append(f"- coste de permute con k={len(args.shift_list)}: "
                     f"~{len(args.shift_list) * max(perm_seconds):.0f} s extra por problema "
                     "(el gancho pasa el presupuesto de tiempo restante)")
    orig_n = [r["prompt_n"] for r in main_rows if r["prompt_n"]]
    shift_n = [r["prompt_n"] for r in perm_rows if r["status"] == "ok" and r["prompt_n"]]
    if orig_n and shift_n:
        reuse = statistics.mean(shift_n) <= 0.8 * statistics.mean(orig_n)
        lines.append(f"- prompt_n medio: original {statistics.mean(orig_n):.0f}, "
                     f"desplazado {statistics.mean(shift_n):.0f} => "
                     + ("indicio de reutilizacion del prefijo" if reuse
                        else "relee casi todo (las opciones van al final del mensaje)"))
    return lines


def fallback_report(problems, dev_ids):
    """Correccion A data: the most frequent letter of train_fit vs FALLBACK."""
    from collections import Counter
    fit = [p for p in problems if p["id"] not in dev_ids and p.get("label")]
    counts = Counter(p["label"] for p in fit)
    top, n_top = counts.most_common(1)[0]
    share = 100.0 * n_top / len(fit)
    return (f"- Correccion A: letra mas frecuente de train_fit = {top} "
            f"({n_top}/{len(fit)} = {share:.1f}%); FALLBACK_ANSWER=D; "
            "propon el valor en el informe, no edites config.env")


def main():
    args = parse_args()
    if args.mock:
        mock.install()
    problems, dev, with_options, without_options = load_dev(args.input)
    dev_ids = {p["id"] for p in dev}
    dev_opt = with_options[:args.limit]
    no_opt = without_options[:args.no_options_limit]
    perm_problems = [p for p in dev_opt if permute_mod.distinct_contents(p)][:args.permute_limit]
    if not dev_opt:
        sys.exit("no dev problems with read options")
    print(f"dev: {len(dev)} ids, {len(with_options)} con opciones (midiendo {len(dev_opt)}), "
          f"{len(without_options)} sin opciones (midiendo {len(no_opt)}), "
          f"muestra permute {len(perm_problems)}")
    warn_time(args, len(dev_opt) + len(no_opt), len(perm_problems))
    main_rows, no_options_rows = [], []
    for problem in dev_opt:
        main_rows.append(solve_row(problem))
    for problem in no_opt:
        no_options_rows.append(no_options_row(problem))
    perm_rows, perm_summaries = [], []
    for problem in perm_problems:
        main_row = next(r for r in main_rows if r["id"] == problem["id"])
        if main_row["status"] != "ok" or not main_row["answer"]:
            continue
        new_rows, summary = permutation_rows(problem, args, main_row)
        perm_rows += new_rows
        perm_summaries.append(summary)
    rows = main_rows + no_options_rows + perm_rows
    print()
    report = ["# bias: diagnostico del sesgo de letra (fase 1)", "",
              f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}  "
              f"Mock: {'SI (no son medidas)' if args.mock else 'no'}", ""]
    report += stats.describe("principal (con opciones, via oficial solve)", main_rows)
    if no_options_rows:
        report += stats.describe("sin opciones leidas", no_options_rows)
    report += analyses_d1_d3(args, main_rows, no_options_rows, pipeline.FALLBACK)
    report += analyses_d4_d6(args, main_rows, perm_rows, perm_summaries)
    report += ["## Correccion A (para la fase 2)", fallback_report(problems, dev_ids), ""]
    if args.mock:
        report += ["AVISO: mock, no son medidas.", ""]
    fields = ["kind", "id", "shift", "status", "answer", "label", "correct", "seconds",
              "prompt_n", "entropy", "sticky", "consistent", "content"] + [f"p{l}" for l in LETTERS]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = f"results/bias-diagnose-{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md_path = f"results/bias-diagnose-{stamp}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(report))
    print(f"\ndetails: {csv_path}\nsummary: {md_path}")
    if args.mock:
        print("[mock] simulated numbers: plumbing only, measure on the board.")


if __name__ == "__main__":
    main()


