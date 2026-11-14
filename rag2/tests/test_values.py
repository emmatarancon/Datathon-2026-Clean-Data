"""Tests for rag2/values.py: canonical values and statement number multisets."""
import unittest
from collections import Counter

from rag2 import values


class CanonicalValueTest(unittest.TestCase):
    def test_half_forms_are_equal(self):
        # documented default: 1/2, 0,5 and 0.5 are the same value
        self.assertEqual(values.canonical_value("1/2"),
                         values.canonical_value("0,5"))
        self.assertEqual(values.canonical_value("1/2"),
                         values.canonical_value("0.5"))
        self.assertEqual(values.canonical_value("0,5"), "1/2")

    def test_percent_is_not_half(self):
        self.assertNotEqual(values.canonical_value("50%"),
                            values.canonical_value("1/2"))

    def test_thousands_separators_removed(self):
        self.assertEqual(values.canonical_value("1,000"), values.canonical_value("1000"))
        self.assertEqual(values.canonical_value("1 000"), values.canonical_value("1000"))
        self.assertEqual(values.canonical_value("12,345"), "12345")
        # a comma with one or two decimals is a decimal comma, not thousands
        self.assertEqual(values.canonical_value("12,5"), values.canonical_value("12.5"))

    def test_exact_fractions(self):
        self.assertEqual(values.canonical_value("2.50"), "5/2")
        self.assertEqual(values.canonical_value("3"), "3")
        self.assertEqual(values.canonical_value("3.0"), "3")

    def test_units_survive(self):
        self.assertEqual(values.canonical_value("81 cm²"), values.canonical_value("81cm²"))
        self.assertNotEqual(values.canonical_value("81 cm²"), values.canonical_value("81"))
        self.assertNotEqual(values.canonical_value("81 cm²"), values.canonical_value("81 cm"))

    def test_multiset_of_option_values(self):
        a = {"A": "10", "B": "12", "C": "0,5"}
        b = {"A": "12", "B": "0.5", "C": "10"}
        self.assertEqual(values.values_multiset(a), values.values_multiset(b))
        c = {"A": "10", "B": "12", "C": "0.4"}
        self.assertNotEqual(values.values_multiset(a), values.values_multiset(c))


class StatementNumbersTest(unittest.TestCase):
    def test_same_numbers_same_multiset(self):
        self.assertEqual(values.statement_numbers("3, 5, 7"),
                         values.statement_numbers("3 5 7"))
        self.assertEqual(values.statement_numbers("3, 5, 7"),
                         Counter({"3": 1, "5": 1, "7": 1}))

    def test_decimal_and_fraction_tokens(self):
        self.assertEqual(values.statement_numbers("3.5 kg"), Counter({"7/2": 1}))
        self.assertEqual(values.statement_numbers("1/2 of the cake"), Counter({"1/2": 1}))

    def test_thousands_in_statement(self):
        self.assertEqual(values.statement_numbers("1,000 apples"), Counter({"1000": 1}))

    def test_one_number_change_breaks_equality(self):
        a = values.statement_numbers("Tom has 3 bags with 4 apples and eats 2.")
        b = values.statement_numbers("Tom has 3 bags with 5 apples and eats 2.")
        self.assertNotEqual(a, b)

    def test_multiset_not_set(self):
        self.assertNotEqual(values.statement_numbers("3 cats and 3 dogs"),
                            values.statement_numbers("3 cats and dogs"))

    def test_extractor_keeps_every_number(self):
        # the generic extractor keeps the leading exam number as well; dedup
        # strips it with normalized_text before comparing multisets
        self.assertEqual(values.statement_numbers("13. For 3 days."),
                         Counter({"13": 1, "3": 1}))


if __name__ == "__main__":
    unittest.main()
