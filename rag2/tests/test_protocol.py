"""Tests for rag2/protocol.py: Wilson, paired, describe, token estimation."""
import contextlib
import io
import unittest

from rag2 import protocol


class WilsonTest(unittest.TestCase):
    def test_known_values(self):
        lo, hi = protocol.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)

    def test_zero_hits(self):
        lo, hi = protocol.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=3)

    def test_perfect(self):
        lo, hi = protocol.wilson(10, 10)
        self.assertEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.7224, places=3)

    def test_empty(self):
        self.assertEqual(protocol.wilson(0, 0), (0.0, 1.0))

    def test_monotone_in_k(self):
        lo1, _ = protocol.wilson(1, 50)
        lo2, _ = protocol.wilson(10, 50)
        self.assertGreater(lo2, lo1)


class PairedTest(unittest.TestCase):
    def test_counts(self):
        counts = protocol.paired([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(counts, {"only_a": 1, "only_b": 1, "both": 1, "neither": 1})

    def test_paired_text(self):
        text = protocol.paired_text("gate", [1, 0], "base", [0, 1])
        self.assertIn("solo gate 1", text)
        self.assertIn("solo base 1", text)


class DescribeTest(unittest.TestCase):
    def test_describe_block(self):
        rows = [{"answer": "A", "label": "A", "correct": True, "seconds": 9.0},
                {"answer": "B", "label": "A", "correct": False, "seconds": 11.0}]
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            protocol.describe("test run", rows)
        out = buffer.getvalue()
        self.assertIn("N=2", out)
        self.assertIn("aciertos=1", out)
        self.assertIn("Wilson95=[", out)
        self.assertIn("media 10.0 s", out)
        self.assertIn("peor 11.0 s", out)
        self.assertIn("A:1/2", out)  # predicted A once, truth has A twice
        self.assertIn("diferencia no significativa", out)

    def test_no_warning_at_n_100(self):
        rows = [{"answer": "D", "label": "D", "correct": True, "seconds": 1.0}] * 100
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            protocol.describe("big", rows)
        self.assertNotIn("diferencia no significativa", buffer.getvalue())


class TokenEstimationTest(unittest.TestCase):
    def setUp(self):
        # deterministic: pretend the /tokenize probe already failed (laptop)
        self._old = dict(protocol._TOKENIZE)
        protocol._TOKENIZE.update(checked=True, available=False)

    def tearDown(self):
        protocol._TOKENIZE.clear()
        protocol._TOKENIZE.update(self._old)

    def test_count_tokens_fallback(self):
        self.assertEqual(protocol.count_tokens("a" * 35), 10)  # ceil(35/3.5)

    def test_truncate_tokens(self):
        text = " ".join(f"word{i}" for i in range(300))
        cut = protocol.truncate_tokens(text, 120)
        self.assertLessEqual(len(cut), 420)
        self.assertLess(len(cut), len(text))
        # never mid-word: the cut lands on a space (or the original end)
        self.assertTrue(text[len(cut):].startswith(" ") or len(cut) == len(text))
        self.assertEqual(protocol.truncate_tokens("short", 120), "short")

    def test_estimate_latency(self):
        self.assertAlmostEqual(protocol.estimate_latency_seconds(110), 11.0)
        self.assertAlmostEqual(protocol.estimate_latency_seconds(0), 1.0)

    def test_messages_tokens(self):
        messages = [{"role": "system", "content": "a" * 35},
                    {"role": "user", "content": "b" * 7}]
        self.assertEqual(protocol.messages_tokens(messages), 10 + 2)


if __name__ == "__main__":
    unittest.main()
