"""Tests for bias/permute.py: content scoring across option shifts.

The model is simulated: no board and no servers.
"""
import math
import unittest

from bias import permute as permute_mod


class PermuteTest(unittest.TestCase):
    def test_shifted_options_maps_any_order(self):
        options = {"A": "8", "B": "9", "C": "10", "D": "12", "E": "14"}
        shifted = permute_mod.shifted_options(options, 1)
        self.assertEqual(shifted["A"], "14")  # A takes E's content
        self.assertEqual(shifted["B"], "8")
        self.assertEqual(shifted["E"], "12")
        two = permute_mod.shifted_options(options, 2)
        self.assertEqual(two["A"], "12")
        self.assertEqual(two["C"], "8")

    def test_content_letter_roundtrip(self):
        self.assertEqual(permute_mod.content_letter("B", 1, 5), "A")
        self.assertEqual(permute_mod.content_letter("A", 1, 5), "E")
        self.assertEqual(permute_mod.content_letter("C", 3, 5), "E")

    def test_distinct_contents_rejects_bad_problems(self):
        self.assertFalse(permute_mod.distinct_contents({"options": {}}))
        self.assertFalse(permute_mod.distinct_contents(
            {"options": {"A": "1", "B": "2", "C": "3", "D": "4"}}))
        self.assertFalse(permute_mod.distinct_contents(
            {"options": {"A": "1", "B": "1", "C": "3", "D": "4", "E": "5"}}))
        self.assertTrue(permute_mod.distinct_contents(
            {"options": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}}))

    def test_permuted_letter_scores_content_not_position(self):
        # the simulated model always picks the letter that holds the value "10"
        # in the prompt it sees; that content must win in the original order
        problem = {"id": "p", "text": "How many?",
                   "options": {"A": "8", "B": "9", "C": "10", "D": "12", "E": "14"},
                   "label": "C"}

        def fake_logprobs(messages, timeout=12.0, url=None, with_grammar=True):
            text = messages[-1]["content"]
            letter = next(l for l in "ABCDE" if f"{l}) 10" in text)
            return {l: math.log(0.9 if l == letter else 0.025) for l in "ABCDE"}, \
                {"prompt_n": 100}

        original = permute_mod.letter_logprobs
        permute_mod.letter_logprobs = fake_logprobs
        try:
            letter, info = permute_mod.permuted_letter(problem, shifts=(1, 2, 3))
        finally:
            permute_mod.letter_logprobs = original
        self.assertEqual(letter, "C")
        self.assertEqual(info["used_shifts"], [1, 2, 3])

    def test_budget_exhausted_returns_none(self):
        problem = {"id": "p", "text": "How many?",
                   "options": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}}
        letter, info = permute_mod.permuted_letter(problem, deadline=0.0)
        self.assertIsNone(letter)
        self.assertEqual(info["used_shifts"], [])


if __name__ == "__main__":
    unittest.main()
