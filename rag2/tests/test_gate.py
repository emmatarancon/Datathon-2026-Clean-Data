"""Tests for rag2/gate.py: threshold, budget, ordering, baseline identity."""
import unittest

from copilot.llm import build_messages
from rag2 import data, gate, protocol


def unit(x, y):
    """A 2-d unit vector, so the cosine with [1, 0] is exactly x."""
    norm = (x * x + y * y) ** 0.5
    return [x / norm, y / norm]


def make_index():
    # cosines with the query vector [1, 0]: near1 0.99, near2 0.95, mid 0.90
    problems = [
        {"id": "near1", "language": "eng", "text": "Anna has 3 bags with 4 apples.",
         "options": {"A": "8", "B": "9", "C": "10"}, "label": "C"},
        {"id": "near2", "language": "eng", "text": "Anna has 3 bags of 4 apples each.",
         "options": {"A": "7", "B": "12"}, "label": "B"},
        {"id": "mid", "language": "eng", "text": "A train travels 12 km in 20 minutes.",
         "options": {"A": "36 km", "B": "24 km"}, "label": "A"},
        {"id": "far", "language": "eng", "text": "What is 7 times 8?",
         "options": {"A": "56", "B": "54"}, "label": "A"},
        {"id": "nopts", "language": "eng", "text": "Anna has 3 bags with 4 apples, shown.",
         "options": {}, "label": "B"},
    ]
    vectors = [unit(0.99, 0.14071), unit(0.95, 0.31225), unit(0.90, 0.43589),
               unit(0.0, 1.0), unit(0.93, 0.36756)]
    return data.MemIndex.from_problems(problems, vectors)


class GateTestBase(unittest.TestCase):
    def setUp(self):
        self._old = data._MOCK_EMBED
        data._MOCK_EMBED = lambda texts: [unit(1, 0) for _ in texts]
        # deterministic token estimates: pretend the /tokenize probe failed
        self._old_tokenize = dict(protocol._TOKENIZE)
        protocol._TOKENIZE.update(checked=True, available=False)
        self.index = make_index()
        self.problem = {"id": "q", "language": "eng", "text": "Anna has 3 bags with 4 apples. She eats 2. How many are left?",
                        "options": {"A": "8", "B": "9", "C": "10"}, "label": "C"}

    def tearDown(self):
        data._MOCK_EMBED = self._old
        protocol._TOKENIZE.clear()
        protocol._TOKENIZE.update(self._old_tokenize)


class GateSelectionTest(GateTestBase):
    def test_nobody_passes_means_baseline_identical_prompt(self):
        # the best cosine is 0.99: above 0.995 nobody passes the gate
        examples, info = gate.select_examples(self.problem, self.index, tau=0.995)
        self.assertEqual(examples, [])
        self.assertFalse(info["coverage"])
        self.assertEqual(gate.prompt_bytes(self.problem, examples),
                         build_messages(self.problem, ()))
        self.assertTrue(gate.baseline_identical(examples))

    def test_keeps_at_most_two_most_similar(self):
        examples, info = gate.select_examples(self.problem, self.index, tau=0.9)
        self.assertEqual(len(examples), 2)
        # cosine 1.0 and ~0.95 pass; the 0.9 neighbour is third and stays out
        self.assertNotIn("mid", info["kept"])
        self.assertIn("near1", info["kept"])

    def test_most_similar_last(self):
        examples, _info = gate.select_examples(self.problem, self.index, tau=0.9)
        self.assertEqual(examples[-1]["id"], "near1")  # cosine 1.0, next to the question
        self.assertEqual(examples[0]["id"], "near2")

    def test_examples_without_options_are_dropped(self):
        examples, _info = gate.select_examples(self.problem, self.index, tau=0.0)
        self.assertNotIn("nopts", [e["id"] for e in examples])

    def test_exclude_id_removes_the_identical_row(self):
        examples, _info = gate.select_examples(self.problem, self.index, tau=0.0,
                                               exclude_id="q")
        self.assertNotIn("q", [e["id"] for e in examples])


class GateBudgetTest(GateTestBase):
    def test_budget_drops_least_similar_first(self):
        # two trimmed examples (120 tokens each) cost about 31 s with the
        # chars/3.5 estimate: a 25 s budget drops the least similar first
        long_text = "Anna shares apples with her friends. " * 30
        problems = [
            {"id": "near1", "language": "eng", "text": long_text,
             "options": {"A": "8", "B": "9", "C": "10"}, "label": "C"},
            {"id": "near2", "language": "eng", "text": long_text + " Then she walks home.",
             "options": {"A": "7", "B": "12"}, "label": "B"},
        ]
        vectors = [unit(0.99, 0.14071), unit(0.95, 0.31225)]
        index = data.MemIndex.from_problems(problems, vectors)
        examples, info = gate.select_examples(self.problem, index, tau=0.9,
                                              budget_s=25.0)
        self.assertEqual(info["dropped_by_budget"], 1)
        self.assertEqual([e["id"] for e in examples], ["near1"])
        self.assertLessEqual(info["estimated_seconds"], 25.0)

    def test_no_drop_when_short(self):
        examples, info = gate.select_examples(self.problem, self.index, tau=0.9)
        self.assertEqual(info["dropped_by_budget"], 0)
        self.assertEqual(len(examples), 2)


class CompactFormatTest(GateTestBase):
    def test_statement_trimmed_to_token_budget(self):
        example = {"id": "long", "text": " ".join(f"word{i}" for i in range(500)),
                   "options": {"A": "1", "B": "2"}, "label": "A"}
        view = gate.compact_example(example)
        self.assertLessEqual(len(view["text"]), 420)
        self.assertEqual(view["label"], "A")

    def test_compact_example_renders_like_the_existing_pipeline(self):
        example = {"id": "e", "text": "Two plus two?", "options": {"A": "3", "B": "4"},
                   "label": "B"}
        view = gate.compact_example(example)
        messages = build_messages(self.problem, [view])
        user = messages[1]["content"]
        self.assertIn("Two plus two?", user)
        self.assertIn("A) 3", user)
        self.assertIn("B) 4", user)
        self.assertTrue(user.endswith("Answer:"))
        self.assertEqual(messages[2]["content"], "B")

    def test_without_options_returns_none(self):
        self.assertIsNone(gate.compact_example({"id": "x", "text": "t", "options": {}}))


if __name__ == "__main__":
    unittest.main()
