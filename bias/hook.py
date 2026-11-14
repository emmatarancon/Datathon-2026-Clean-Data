"""The pipeline hook: DEBIAS_MODE in config.env decides the correction.

- off: the hook is never called (pipeline checks first).
- calibrate: argmax(log p - bias) with the bias JSON from CALIBRATION_FILE
  (results/bias-calibration-<method>.json).
- permute: answer by permuted options scored per content (bias/permute.py),
  within the remaining time budget.
- calibrate+permute: both — the per-content scores subtract the letter bias.

Any exception or missing logprobs falls back to the model's letter with a
warning, never the other way round. Cost of calibrate ≈ 0 (same request).
"""
import sys
import time

from copilot.config import CONFIG
from copilot.confidence import LETTERS
from copilot.llm import URL
from bias import permute as permute_mod
from bias.calibration import debiased_argmax, load_bias

MODES = ("calibrate", "permute", "calibrate+permute")


def options_letters(problem):
    return {str(l).strip().upper() for l in (problem.get("options") or {})}


def debias_letter(problem, letter, probs, mode="calibrate", examples=(),
                  image_b64=None, llm_url=URL, remaining=30.0):
    """(letter, note) or None to keep the model's letter (with a warning)."""
    if mode not in MODES:
        return None
    if not probs:
        print("WARNING bias: no letter logprobs from the server, keeping the "
              "model's letter", file=sys.stderr)
        return None
    bias = None
    if "calibrate" in mode:
        bias = load_bias(str(CONFIG.get("CALIBRATION_FILE", "") or "").strip())
        if not bias:
            print("WARNING bias: CALIBRATION_FILE missing or unusable, "
                  "calibration skipped", file=sys.stderr)
    allowed = options_letters(problem) or None
    if "permute" in mode and (problem.get("options") or {}):
        deadline = time.monotonic() + max(remaining - 1.0, 0.0)
        chosen, info = permute_mod.permuted_letter(
            problem, examples=examples, image_b64=image_b64, url=llm_url,
            bias=bias, deadline=deadline)
        if chosen is None:
            print(f"WARNING bias: permutation gave no answer ({info.get('reason')}); "
                  "falling back", file=sys.stderr)
            if "calibrate" not in mode:
                return None
        else:
            note = f"permute: {chosen} (shifts {info.get('used_shifts')})"
            return chosen, note
    if bias:  # plain calibrate, or permute fell back within calibrate+permute
        chosen = debiased_argmax(probs, allowed, bias)
        if chosen is None:
            print("WARNING bias: no candidate letters, keeping the model's letter",
                  file=sys.stderr)
            return None
        if chosen != letter:
            print(f"NOTE bias: the model said {letter}, the debias prefers {chosen}",
                  file=sys.stderr)
        return chosen, f"calibrate: {chosen}"
    return None
