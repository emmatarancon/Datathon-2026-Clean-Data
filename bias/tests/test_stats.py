"""Tests for bias/stats.py: Wilson and chi-square with known values."""
import unittest

from bias import stats


class WilsonTest(unittest.TestCase):
    def test_known_values(self):
        lo, hi = stats.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)

    def test_zero_hits_and_empty(self):
        lo, hi = stats.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=3)
        self.assertEqual(stats.wilson(0, 0), (0.0, 1.0))


class ChiSquareTest(unittest.TestCase):
    def test_uniform_is_zero(self):
        counts = {"A": 10, "B": 10, "C": 10, "D": 10, "E": 10}
        self.assertAlmostEqual(stats.chi_square(counts, stats.uniform_expected(counts)), 0.0)

    def test_known_statistic(self):
        # 20/0/0/0/0 vs uniform 4 each: (16^2/4) + 4*(4^2/4) = 64 + 16 = 80
        counts = {"A": 20, "B": 0, "C": 0, "D": 0, "E": 0}
        expected = {l: 4.0 for l in "ABCDE"}
        self.assertAlmostEqual(stats.chi_square(counts, expected), 80.0)

    def test_critical_value_discriminates(self):
        counts = {"A": 30, "B": 5, "C": 5, "D": 5, "E": 5}
        self.assertGreater(stats.chi_square(counts, stats.uniform_expected(counts)),
                           stats.CHI2_CRITICAL_DF4)  # 50 > 9.49
        even = {l: 10 for l in "ABCDE"}
        self.assertLess(stats.chi_square(even, stats.uniform_expected(even)),
                        stats.CHI2_CRITICAL_DF4)


class LettersTest(unittest.TestCase):
    def test_letter_counts_ignores_unknown(self):
        rows = [{"answer": "D"}, {"answer": " A "}, {"answer": "X"}, {"answer": ""}]
        counts = stats.letter_counts(rows)
        self.assertEqual(counts["D"], 1)
        self.assertEqual(counts["A"], 1)
        self.assertEqual(sum(counts.values()), 2)

    def test_confusion_matrix_and_metrics(self):
        rows = [{"answer": "D", "label": "D"}, {"answer": "D", "label": "A"},
                {"answer": "A", "label": "A"}]
        matrix, per_letter = stats.confusion(rows)
        self.assertEqual(matrix["A"]["D"], 1)
        self.assertEqual(matrix["A"]["A"], 1)
        self.assertAlmostEqual(per_letter["D"]["precision"], 0.5)
        self.assertAlmostEqual(per_letter["D"]["recall"], 1.0)
        self.assertAlmostEqual(per_letter["A"]["recall"], 0.5)

    def test_paired(self):
        counts = stats.paired([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(counts, {"only_a": 1, "only_b": 1, "both": 1, "neither": 1})


if __name__ == "__main__":
    unittest.main()
