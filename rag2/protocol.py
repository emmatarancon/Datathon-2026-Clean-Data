"""Measurement protocol: Wilson intervals, paired comparison, reporting,
token estimation and the latency budget. Everything tune.py/final.py print
goes through here so every measurement reports the same numbers.
"""
import math
import os
import statistics
import time

from copilot.config import CONFIG

# Board measurements (given by the organizers): ~11 tokens/s read, ~6 written.
READ_TPS = 11.0
CHARS_PER_TOKEN = 3.5  # the fallback estimate embed.py also uses
BASE_OVERHEAD_S = 1.0  # embedding, search and writing the answer
SIGNIFICANT_N = 100    # below this, print that the difference is not significant

_TOKENIZE = {"checked": False, "available": False}


def _tokenize_url():
    return f"http://127.0.0.1:{CONFIG.get('LLM_PORT', 8081)}/tokenize"


def tokenize_available():
    """One real POST /tokenize probe per process; False means chars/3.5.

    On the laptop (no servers) this fails immediately; on the board it should
    succeed against llama-server. Never raises.
    """
    if not _TOKENIZE["checked"]:
        _TOKENIZE["checked"] = True
        try:
            from copilot.client import post
            response = post(_tokenize_url(), {"content": "hello world"}, timeout=3)
            _TOKENIZE["available"] = len(response.get("tokens", [])) > 0
        except Exception:
            _TOKENIZE["available"] = False
    return _TOKENIZE["available"]


def count_tokens(text):
    """Real token count via /tokenize when reachable, else ceil(chars/3.5)."""
    if tokenize_available():
        try:
            from copilot.client import post
            response = post(_tokenize_url(), {"content": text}, timeout=5)
            return len(response["tokens"])
        except Exception:
            pass
    return int(math.ceil(len(text) / CHARS_PER_TOKEN))


def truncate_tokens(text, max_tokens):
    """Cut text to about max_tokens (chars/3.5), never mid-word."""
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    if cut <= 0:
        cut = max_chars
    return text[:cut]


def messages_tokens(messages):
    """Estimated token count of a chat prompt (list of {"role","content"})."""
    total = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        else:  # multimodal content lists: count the text parts only
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += count_tokens(part.get("text", ""))
    return total


def estimate_latency_seconds(tokens, read_tps=READ_TPS):
    """The prompt is read at ~11 tokens/s; add a fixed 1 s of overhead."""
    return tokens / read_tps + BASE_OVERHEAD_S


def wilson(k, n, z=1.96):
    """95% Wilson score interval (lo, hi) for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def paired(flags_a, flags_b):
    """Paired comparison of two answer sets over the same problems, in order.

    Returns the counts: only_a (A right, B wrong), only_b, both, neither.
    """
    only_a = only_b = both = neither = 0
    for a, b in zip(flags_a, flags_b):
        if a and b:
            both += 1
        elif a:
            only_a += 1
        elif b:
            only_b += 1
        else:
            neither += 1
    return {"only_a": only_a, "only_b": only_b, "both": both, "neither": neither}


def fmt_pct(x):
    return f"{100.0 * x:.1f}%"


def _letter(value):
    return str(value or "").strip().upper()[:1]


def describe(title, rows, label_key="label", answer_key="answer", seconds_key="seconds"):
    """The standard measurement block: N, hits, Wilson, latency, letters.

    rows: dicts with the answer, the true label, a "correct" bool and seconds.
    With N < 100 it prints the literal 'diferencia no significativa'.
    """
    n = len(rows)
    lines = [f"== {title} =="]
    if n == 0:
        lines.append("N=0: nothing measured")
        print("\n".join(lines))
        return
    hits = sum(1 for r in rows if r.get("correct"))
    lo, hi = wilson(hits, n)
    lines.append(f"N={n}  aciertos={hits}  acc={fmt_pct(hits / n)}  "
                 f"Wilson95=[{fmt_pct(lo)}, {fmt_pct(hi)}]")
    seconds = [float(r[seconds_key]) for r in rows if r.get(seconds_key) is not None]
    if seconds:
        lines.append(f"latencia: media {statistics.mean(seconds):.1f} s, "
                     f"mediana {statistics.median(seconds):.1f} s, "
                     f"peor {max(seconds):.1f} s")
    predicted = {letter: sum(1 for r in rows if _letter(r.get(answer_key)) == letter)
                 for letter in "ABCDE"}
    truth = {letter: sum(1 for r in rows if _letter(r.get(label_key)) == letter)
             for letter in "ABCDE"}
    lines.append("letras pred vs true: " + "  ".join(
        f"{l}:{predicted[l]}/{truth[l]}" for l in "ABCDE"))
    if n < SIGNIFICANT_N:
        lines.append("diferencia no significativa")
    print("\n".join(lines))


def paired_text(name_a, flags_a, name_b, flags_b):
    """The paired line: who is right only in A, only in B."""
    counts = paired(flags_a, flags_b)
    return (f"pareada {name_a} vs {name_b}: solo {name_a} {counts['only_a']}, "
            f"solo {name_b} {counts['only_b']}, ambas {counts['both']}, ninguna {counts['neither']}")


def results_path(name, stamp=None):
    """results/<name>-<stamp>.csv next to the other experiment outputs."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(root, "results", f"{name}-{stamp}.csv")


def config_get(key, default):
    """CONFIG value as float, falling back to default when unset/invalid."""
    raw = CONFIG.get(key, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)
