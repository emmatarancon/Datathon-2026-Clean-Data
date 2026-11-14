"""Tests for bias/calibration.py: token grouping, debiased argmax, bias files."""
import json
import math
import os
import tempfile
import unittest

from bias.calibration import (debiased_argmax, group_letter_logprobs, load_bias,
                              renormalize, save_bias)


def probs_from(grouped):
    return renormalize(grouped)


class GroupingTest(unittest.TestCase):
    def test_variants_of_the_same_letter_add_up(self):
        entries = [{"token": "A", "logprob": math.log(0.3)},
                   {"token": " A", "logprob": math.log(0.2)},
                   {"token": "D", "logprob": math.log(0.5)},
                   {"token": "foo", "logprob": math.log(0.1)}]
        probs = probs_from(group_letter_logprobs(entries))
        self.assertAlmostEqual(probs["A"], 0.5, places=6)
        self.assertAlmostEqual(probs["D"], 0.5, places=6)
        self.assertEqual(set(probs), {"A", "D"})

    def test_single_token_is_preserved(self):
        grouped = group_letter_logprobs([{"token": "B", "logprob": math.log(0.7)}])
        self.assertAlmostEqual(math.exp(grouped["B"]), 0.7, places=6)

    def test_nan_and_garbage_are_skipped(self):
        entries = [{"token": "C", "logprob": float("nan")},
                   {"token": "C!", "logprob": math.log(0.9)},
                   {"logprob": math.log(0.9)},
                   {"token": "C", "logprob": "not-a-number"}]
        self.assertEqual(group_letter_logprobs(entries), {})

    def test_renormalize_sums_to_one(self):
        grouped = group_letter_logprobs([{"token": l, "logprob": math.log(p)}
                                         for l, p in (("A", 0.2), ("D", 0.5), ("E", 0.3))])
        self.assertAlmostEqual(sum(probs_from(grouped).values()), 1.0, places=9)


class DebiasedArgmaxTest(unittest.TestCase):
    def test_bias_changes_the_letter_when_it_should(self):
        probs = {"A": 0.34, "B": 0.33, "C": 0.33}
        self.assertEqual(debiased_argmax(probs, None, None), "A")
        # dividing by the prior: D is rare a priori (log(0.05) bias), so its
        # 0.22 beats B's 0.22/0.1 = 2.2 with 0.22/0.05 = 4.4
        bias = {"A": math.log(0.5), "B": math.log(0.1), "C": math.log(0.1),
                "D": math.log(0.05)}
        probs_d = {"A": 0.34, "B": 0.22, "C": 0.22, "D": 0.22}
        self.assertEqual(debiased_argmax(probs_d, None, bias), "D")

    def test_respects_letters_absent_from_the_problem(self):
        bias = {"A": 0.0, "B": -5.0}  # B heavily favoured by the bias
        probs = {"A": 0.5, "B": 0.5}
        self.assertEqual(debiased_argmax(probs, allowed_letters={"A"}, bias=bias), "A")

    def test_missing_bias_falls_back_to_raw_argmax(self):
        probs = {"A": 0.4, "D": 0.35}
        self.assertEqual(debiased_argmax(probs, None, {"A": 0.0}), "A")
        self.assertEqual(debiased_argmax(probs, None, None), "A")

    def test_no_usable_probs(self):
        self.assertIsNone(debiased_argmax({}, None, {"A": 0.0}))
        self.assertIsNone(debiased_argmax({"A": 0.0}, None, {"A": 0.0}))


class SaveLoadBiasTest(unittest.TestCase):
    def test_roundtrip_and_garbage(self):
        record = {"method": "contextual", "date": "2026-11-14", "seed": 42,
                  "n": 5, "bias": {"A": -1.2, "D": -0.8, "Z": 9, "B": "oops"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cal.json")
            save_bias(record, path)
            with open(path) as f:
                stored = json.load(f)
            bias = load_bias(path)
        self.assertEqual(stored["method"], "contextual")
        self.assertEqual(stored["seed"], 42)
        self.assertEqual(bias, {"A": -1.2, "D": -0.8})  # garbage dropped per letter

    def test_missing_or_broken_file(self):
        self.assertIsNone(load_bias(""))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "missing.json")
            self.assertIsNone(load_bias(path))
            with open(path, "w") as f:
                f.write("{not json")
            self.assertIsNone(load_bias(path))


if __name__ == "__main__":
    unittest.main()
