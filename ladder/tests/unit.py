"""Sanity checks for the doubt detector, with synthetic server responses."""
import sys

sys.path.insert(0, "/Users/emmataranconestrada/Documents/UPCdades/2n/datathon")
from copilot.confidence import (calibration_text, bucket_label, bucket_labels,  # noqa: E402
                                enabled, extract_letter_probs, sure_threshold,
                                summarize, bars)
from copilot.config import CONFIG  # noqa: E402


def check(name, got, want):
    ok = got == want
    print(f"{name}: {'ok' if ok else 'FAIL (got ' + repr(got) + ', want ' + repr(want) + ')'}")
    assert ok, name


# 1. a server without logprobs support -> None, nothing breaks
check("no logprobs -> None", extract_letter_probs({"choices": [{"message": {"content": "A"}}]}), None)
check("empty top_logprobs -> None",
      extract_letter_probs({"choices": [{"logprobs": {"content": [{"token": "A", "logprob": 0, "top_logprobs": []}]}}]}),
      None)
# 2. a single letter is not a usable distribution
check("single letter -> None",
      extract_letter_probs({"choices": [{"logprobs": {"content": [{"token": "A", "logprob": -0.1, "top_logprobs": [
          {"token": "A", "logprob": -0.1}]}]}}]}), None)
# 3. two letters renormalize to a sum of 1
r = {"choices": [{"logprobs": {"content": [{"token": "B", "logprob": -0.4, "top_logprobs": [
    {"token": "B", "logprob": -0.4}, {"token": "A", "logprob": -0.4}]}]}}]}
p = extract_letter_probs(r)
check("letters filled", sorted(p), ["A", "B", "C", "D", "E"])
check("sums to 1", round(sum(p.values()), 9), 1.0)
check("tie -> 0.5", round(p["A"], 4), 0.5)
# 4. -inf (grammar-masked) and non-letter entries are skipped
r = {"choices": [{"logprobs": {"content": [{"token": "B", "logprob": -0.1, "top_logprobs": [
    {"token": "B", "logprob": -0.1}, {"token": "A", "logprob": -0.1}, {"token": "C", "logprob": float("-inf")},
    {"token": "zz", "logprob": -0.2}]}]}}]}
p = extract_letter_probs(r)
check("-inf skipped", (round(p["B"], 6), round(p["A"], 6), p["C"]), (0.5, 0.5, 0.0))
# 5. whitespace inside the token text is tolerated
r = {"choices": [{"logprobs": {"content": [{"token": "A", "logprob": -0.2, "top_logprobs": [
    {"token": " a ", "logprob": -0.2}, {"token": "E", "logprob": -0.2}]}]}}]}
p = extract_letter_probs(r)
check("token stripped", (p["A"], p["E"]), (0.5, 0.5))
# 6. NaN logprob skipped, the rest still normalize
r = {"choices": [{"logprobs": {"content": [{"token": "A", "logprob": 0.0, "top_logprobs": [
    {"token": "A", "logprob": 0.0}, {"token": "B", "logprob": float("nan")}, {"token": "C", "logprob": -0.5}]}]}}]}
p = extract_letter_probs(r)
check("NaN skipped, sums to 1", round(sum(p.values()), 9), 1.0)

# verdicts
sure = {"A": 0.58, "B": 0.26, "C": 0.09, "D": 0.05, "E": 0.02}
doubt = {"A": 0.22, "B": 0.21, "C": 0.20, "D": 0.19, "E": 0.18}
info = summarize(sure)
check("sure verdict", info["verdict"], "sure")
check("top letter", info["top"], "A")
check("margin", info["margin"], 0.32)
check("doubt verdict", summarize(doubt)["verdict"], "doubt")
check("uniform entropy 1.0",
      summarize({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2})["entropy"], 1.0)
check("threshold override", summarize(sure, threshold=0.9)["verdict"], "doubt")

# config switch
check("default off", enabled(), False)
check("default threshold", sure_threshold(), 0.5)
print("config.env ->", "DOUBT_DETECTOR =", CONFIG.get("DOUBT_DETECTOR"),
      "| DOUBT_SURE_PROB =", CONFIG.get("DOUBT_SURE_PROB"))

print("\n" + bars(sure) + "\n")
print(bars(doubt))

check("bucket labels", bucket_labels(),
      ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"])
check("bucket 1.0", bucket_label(1.0), "[0.8-1.0]")
check("bucket 0.579", bucket_label(0.579), "[0.4-0.6)")

# a calibrated model: sure answers right, doubting answers wrong
rows = [{"probs": sure, "correct": True}, {"probs": dict(sure), "correct": True},
        {"probs": doubt, "correct": False}, {"probs": dict(doubt), "correct": False}]
print("\n" + calibration_text(rows))
print("\n" + calibration_text([]))
print("\nALL UNIT CHECKS PASSED")