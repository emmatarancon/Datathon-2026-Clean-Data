"""Statistics for the bias study: Wilson intervals, chi-square (df=4),
letter counting, the 5x5 confusion matrix, and the standard report block.
Pure standard library; every measurement prints the same numbers.
"""
import math
import statistics

LETTERS = "ABCDE"
CHI2_CRITICAL_DF4 = 9.49  # 5% critical value for 4 degrees of freedom
SIGNIFICANT_N = 100


def wilson(hits, n, z=1.96):
    """95% Wilson score interval (lo, hi) for a proportion."""
    if not n:
        return (0.0, 1.0)
    share = hits / n
    denom = 1.0 + z * z / n
    center = (share + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(share * (1.0 - share) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def letter(value):
    return str(value or "").strip().upper()[:1]


def letter_counts(rows, key="answer"):
    """{letter: count} over rows (unknown letters counted nowhere)."""
    counts = {l: 0 for l in LETTERS}
    for row in rows:
        current = letter(row.get(key))
        if current in counts:
            counts[current] += 1
    return counts


def chi_square(observed, expected):
    """The Pearson chi-square statistic, sum((o-e)^2/e); df = cells - 1."""
    cells = [(observed[key], expected[key]) for key in observed if expected.get(key)]
    if not cells:
        return 0.0
    return sum((o - e) ** 2 / e for o, e in cells)


def uniform_expected(counts):
    """Expected counts under uniformity over the letters present."""
    keys = [key for key in counts if counts.get(key)]
    share = sum(counts.values()) / max(len(keys), 1)
    return {key: share for key in keys}


def confusion(rows, answer_key="answer", label_key="label"):
    """5x5 matrix[true][predicted] plus precision/recall per letter."""
    matrix = {true: {pred: 0 for pred in LETTERS} for true in LETTERS}
    for row in rows:
        true_l, pred_l = letter(row.get(label_key)), letter(row.get(answer_key))
        if true_l in LETTERS and pred_l in LETTERS:
            matrix[true_l][pred_l] += 1
    per_letter = {}
    for target in LETTERS:
        predicted = sum(matrix[t][target] for t in LETTERS)
        true = sum(matrix[target].values())
        hits = matrix[target][target]
        per_letter[target] = {
            "precision": hits / predicted if predicted else None,
            "recall": hits / true if true else None,
            "predicted": predicted, "true": true, "hits": hits,
        }
    return matrix, per_letter


def paired(flags_a, flags_b):
    """Paired comparison: only_a (A right, B wrong), only_b, both, neither."""
    counts = {"only_a": 0, "only_b": 0, "both": 0, "neither": 0}
    for a, b in zip(flags_a, flags_b):
        counts["both" if a and b else "only_a" if a else "only_b" if b else "neither"] += 1
    return counts


def describe(title, rows, seconds_key="seconds", status_key="status"):
    """N, hits, Wilson, letters pred vs true with chi-square, latency, fallbacks."""
    n = len(rows)
    lines = [f"== {title} =="]
    if not n:
        lines.append("N=0: nothing measured")
        print("\n".join(lines))
        return lines
    hits = sum(1 for r in rows if r.get("correct"))
    lo, hi = wilson(hits, n)
    lines.append(f"N={n}  aciertos={hits}  acc={100.0 * hits / n:.1f}%  "
                 f"Wilson95=[{100.0 * lo:.1f}%, {100.0 * hi:.1f}%]")
    predicted = letter_counts(rows)
    truth = letter_counts(rows, key="label")
    lines.append("letras pred vs true: " + "  ".join(
        f"{l}:{predicted[l]}/{truth[l]}" for l in LETTERS))
    chi_uniform = chi_square(predicted, uniform_expected(predicted))
    chi_true = chi_square(predicted, {l: truth[l] for l in LETTERS})
    verdict = "sesgo" if chi_uniform > CHI2_CRITICAL_DF4 else "compatible con uniforme"
    lines.append(f"chi2(df=4) pred vs uniforme = {chi_uniform:.2f} (critico 9.49: {verdict}); "
                 f"vs distribucion verdadera = {chi_true:.2f}")
    seconds = [float(r[seconds_key]) for r in rows if r.get(seconds_key) is not None]
    if seconds:
        lines.append(f"latencia: media {statistics.mean(seconds):.1f} s, "
                     f"mediana {statistics.median(seconds):.1f} s, peor {max(seconds):.1f} s")
    fallbacks = sum(1 for r in rows if status_key in r and r[status_key] != "ok")
    if status_key in rows[0]:
        lines.append(f"respuestas de reserva (status != ok): {fallbacks}/{n}")
    if n < SIGNIFICANT_N:
        lines.append("diferencia no significativa")
    print("\n".join(lines))
    return lines
