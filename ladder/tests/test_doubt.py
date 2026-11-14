"""unittest checks for the doubt detector (copilot/confidence.py), all offline.

    python3 -m unittest discover -s ladder/tests

Synthetic server responses only: no M3Kang data, no network.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from copilot.confidence import (calibration_text, bucket_label,  # noqa: E402
                                bucket_labels, extract_letter_probs, summarize)


def response(top):
    """A chat completion whose logprobs look like the server's documented shape."""
    entries = [{"token": token, "logprob": value} for token, value in top]
    return {"choices": [{"message": {"content": entries[0]["token"]},
                         "logprobs": {"content": [{"token": entries[0]["token"],
                                                   "logprob": entries[0]["logprob"],
                                                   "top_logprobs": entries}]}}]}


class ParserTests(unittest.TestCase):
    def test_no_logprobs_returns_none(self):
        self.assertIsNone(extract_letter_probs({"choices": [{"message": {"content": "A"}}]}))

    def test_empty_or_broken_shapes_return_none(self):
        self.assertIsNone(extract_letter_probs({}))
        self.assertIsNone(extract_letter_probs({"choices": []}))
        self.assertIsNone(extract_letter_probs({"choices": [{"logprobs": {"content": []}}]}))

    def test_single_letter_is_unusable(self):
        self.assertIsNone(extract_letter_probs(response([("A", -0.1)])))

    def test_renormalized_to_one_with_missing_letters_at_zero(self):
        probs = extract_letter_probs(response([("B", -0.4), ("A", -0.4)]))
        self.assertEqual(sorted(probs), ["A", "B", "C", "D", "E"])
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=9)
        self.assertAlmostEqual(probs["A"], 0.5, places=6)
        self.assertEqual(probs["C"], 0.0)

    def test_inf_nan_and_non_letter_tokens_are_skipped(self):
        probs = extract_letter_probs(response([("B", -0.1), ("A", -0.1),
                                               ("C", float("-inf")), ("zz", -0.2),
                                               ("D", float("nan"))]))
        self.assertAlmostEqual(probs["B"], 0.5, places=6)
        self.assertAlmostEqual(probs["A"], 0.5, places=6)
        self.assertEqual(probs["C"], 0.0)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=9)

    def test_token_text_is_stripped(self):
        probs = extract_letter_probs(response([(" a ", -0.2), ("E", -0.2)]))
        self.assertAlmostEqual(probs["A"], 0.5, places=6)
        self.assertAlmostEqual(probs["E"], 0.5, places=6)


class VerdictTests(unittest.TestCase):
    SURE = {"A": 0.58, "B": 0.26, "C": 0.09, "D": 0.05, "E": 0.02}
    DOUBT = {"A": 0.22, "B": 0.21, "C": 0.20, "D": 0.19, "E": 0.18}

    def test_sure_when_one_bar_dominates(self):
        info = summarize(self.SURE)
        self.assertEqual(info["verdict"], "sure")
        self.assertEqual(info["top"], "A")
        self.assertAlmostEqual(info["margin"], 0.32, places=9)

    def test_doubt_when_bars_are_close(self):
        self.assertEqual(summarize(self.DOUBT)["verdict"], "doubt")

    def test_uniform_entropy_is_one(self):
        uniform = {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}
        self.assertAlmostEqual(summarize(uniform)["entropy"], 1.0, places=9)

    def test_threshold_override(self):
        self.assertEqual(summarize(self.SURE, threshold=0.9)["verdict"], "doubt")


class CalibrationTests(unittest.TestCase):
    def test_bucket_labels_in_order(self):
        self.assertEqual(bucket_labels(),
                         ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"])
        self.assertEqual(bucket_label(1.0), "[0.8-1.0]")
        self.assertEqual(bucket_label(0.579), "[0.4-0.6)")

    def test_sure_right_and_doubting_wrong(self):
        sure = {"A": 0.58, "B": 0.26, "C": 0.09, "D": 0.05, "E": 0.02}
        doubt = {"A": 0.22, "B": 0.21, "C": 0.20, "D": 0.19, "E": 0.18}
        rows = [{"probs": sure, "correct": True}, {"probs": dict(sure), "correct": True},
                {"probs": doubt, "correct": False}, {"probs": doubt, "correct": False}]
        report = calibration_text(rows)
        self.assertIn("2/2", report)
        self.assertIn("0/2", report)
        self.assertIn("sure", report)
        self.assertIn("doubting", report)
        self.assertIn("ECE", report)

    def test_rows_without_probs_are_skipped(self):
        self.assertIn("--doubt", calibration_text([{"probs": {}, "correct": True}]))


if __name__ == "__main__":
    unittest.main()