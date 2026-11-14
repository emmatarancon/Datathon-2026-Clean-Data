"""The hook contract: baseline identity, dedup shortcut, safe fallback.

No servers: the model call is captured with a fake answer_letter, so the
tests compare the exact messages the pipeline would send.
"""
import os
import tempfile
import unittest

from copilot import pipeline
from copilot.config import CONFIG
from copilot.llm import build_messages, format_problem
from rag2 import data, hook

Q_VEC = [1.0]


def unit(x, y):
    norm = (x * x + y * y) ** 0.5
    return [x / norm, y / norm]


TWIN = {
    "id": "train-7", "language": "eng", "level": 1, "year": 2019,
    "text": "7. Tom has 3 bags with 4 apples in each bag. He eats 2 apples. How many apples does he have left?",
    "options": {"A": "12", "B": "9", "C": "10", "D": "8", "E": "14"},
    "label": "C",
}
CURRENT = {
    "id": "val-9", "language": "eng", "level": 2, "year": 2024,
    "text": "13. Tom has 3 bags with 4 apples in each bag. He eats 2 apples. How many apples does he have left?",
    "options": {"A": "12", "B": "10", "C": "8", "D": "9", "E": "14"},
}


class HookTestBase(unittest.TestCase):
    def setUp(self):
        self._old_mock = data._MOCK_EMBED
        data._MOCK_EMBED = lambda texts: [Q_VEC for _ in texts]
        self.index = data.MemIndex.from_problems([TWIN], [Q_VEC])
        self.captured = []
        self._saved_config = dict(CONFIG)
        self._old_answer = pipeline.answer_letter
        self._old_probs = pipeline.answer_letter_probs

        def fake_answer(problem, examples=(), timeout=60, image_b64=None, url=None):
            self.captured.append(build_messages(problem, examples))
            return "A", {"prompt_n": 10}

        pipeline.answer_letter = fake_answer
        pipeline.answer_letter_probs = fake_answer  # must never be called here
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAG2_PATHS_FILE"] = os.path.join(self.tmp.name, "paths.csv")
        hook._STATE["warned"] = False

    def tearDown(self):
        data._MOCK_EMBED = self._old_mock
        pipeline.answer_letter = self._old_answer
        pipeline.answer_letter_probs = self._old_probs
        CONFIG.clear()
        CONFIG.update(self._saved_config)
        os.environ.pop("RAG2_PATHS_FILE", None)
        self.tmp.cleanup()
        hook._STATE["bm25_rows"] = None
        hook._STATE["bm25"] = None


class BaselineIdentityTest(HookTestBase):
    def test_baseline_mode_returns_none(self):
        CONFIG["RAG_MODE"] = "baseline"
        self.assertIsNone(pipeline._rag2(CURRENT, self.index, "val-9", None))

    def test_baseline_prompt_is_byte_identical_to_the_original(self):
        CONFIG["RAG_MODE"] = "baseline"
        result = pipeline.solve(CURRENT, index=self.index, k=1, exclude_id="val-9")
        # the original path with k=1: full-format example from index.search
        expected = build_messages(CURRENT, [TWIN])
        self.assertEqual(self.captured, [expected])
        self.assertNotIn("rag2_path", result)
        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["status"], "ok")

    def test_baseline_without_rag_is_untouched(self):
        CONFIG["RAG_MODE"] = "baseline"
        result = pipeline.solve(CURRENT)
        self.assertEqual(self.captured, [build_messages(CURRENT, ())])
        self.assertEqual(result["retrieved"], "")


class GateHookTest(HookTestBase):
    def test_gate_with_nobody_passing_equals_baseline_prompt(self):
        CONFIG["RAG_MODE"] = "gate"
        CONFIG["RAG2_TAU"] = "0.995"
        # point the mock query away from the twin: nobody passes the gate now
        data._MOCK_EMBED = lambda texts: [unit(0.0, 1.0) for _ in texts]
        result = pipeline.solve(CURRENT, index=self.index, k=1, exclude_id="val-9")
        self.assertEqual(self.captured, [build_messages(CURRENT, ())])
        self.assertEqual(result["rag2_path"], "gate")

    def test_gate_uses_compact_examples(self):
        CONFIG["RAG_MODE"] = "gate"
        CONFIG["RAG2_TAU"] = "0.9"
        # a long statement makes the 120-token trim actually shorten the example
        twin = dict(TWIN, text=TWIN["text"] + " " + "Anna also shares pears. " * 60)
        index = data.MemIndex.from_problems([twin], [Q_VEC])
        result = pipeline.solve(CURRENT, index=index, k=1, exclude_id="val-9")
        self.assertEqual(result["rag2_path"], "gate")
        sent = self.captured[0]
        self.assertEqual(len(sent), 4)  # system, example user, example answer, question
        self.assertEqual(sent[2]["content"], "C")
        # the example statement is trimmed: shorter than the full rendering
        self.assertLess(len(sent[1]["content"]),
                        len(format_problem(twin)) + len("\nAnswer:"))

    def test_unknown_mode_falls_back_to_the_original_path(self):
        CONFIG["RAG_MODE"] = "bogus"
        result = pipeline.solve(CURRENT, index=self.index, k=1, exclude_id="val-9")
        self.assertEqual(self.captured, [build_messages(CURRENT, [TWIN])])
        self.assertNotIn("rag2_path", result)

    def test_hook_exception_falls_back_to_the_original_path(self):
        CONFIG["RAG_MODE"] = "gate"

        def boom(*args, **kwargs):
            raise RuntimeError("rag2 exploded")

        original_decide = hook.decide
        hook.decide = boom
        try:
            result = pipeline.solve(CURRENT, index=self.index, k=1, exclude_id="val-9")
        finally:
            hook.decide = original_decide
        self.assertEqual(self.captured, [build_messages(CURRENT, [TWIN])])
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("rag2_path", result)

    def test_gate_without_index_returns_to_the_original_path(self):
        CONFIG["RAG_MODE"] = "gate"
        result = pipeline.solve(CURRENT, index=None, k=0)
        self.assertEqual(self.captured, [build_messages(CURRENT, ())])
        self.assertNotIn("rag2_path", result)


class DedupHookTest(HookTestBase):
    def test_dedup_answers_without_calling_the_model(self):
        CONFIG["RAG_MODE"] = "gate+dedup"
        CONFIG["RAG2_TAU_DUP"] = "0.97"

        def forbidden(*args, **kwargs):
            raise AssertionError("the model must not be called on a dedup hit")

        pipeline.answer_letter = forbidden
        pipeline.answer_letter_probs = forbidden
        result = pipeline.solve(CURRENT, index=self.index, k=1, exclude_id="val-9")
        self.assertEqual(result["answer"], "B")  # the twin's "10" sits at B here
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["rag2_path"], "dedup")

    def test_paths_csv_records_the_path(self):
        CONFIG["RAG_MODE"] = "gate+dedup"
        pipeline.solve(CURRENT, index=self.index, k=1, exclude_id="val-9")
        with open(os.environ["RAG2_PATHS_FILE"]) as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertEqual(lines[0], "id,path,mode,n_examples,answer,seconds,status")
        self.assertIn("val-9,dedup,gate+dedup,0,B", lines[1])

    def test_dedup_miss_continues_to_the_model_with_gate_examples(self):
        CONFIG["RAG_MODE"] = "gate+dedup"
        CONFIG["RAG2_TAU_DUP"] = "0.97"
        # break the twin: one number changes, so dedup cannot transfer
        twin = dict(TWIN, text=TWIN["text"].replace("4 apples", "5 apples"))
        index = data.MemIndex.from_problems([twin], [Q_VEC])
        result = pipeline.solve(CURRENT, index=index, k=1, exclude_id="val-9")
        self.assertEqual(result["answer"], "A")  # the model answered (fake)
        self.assertEqual(result["rag2_path"], "gate")
        self.assertEqual(len(self.captured), 1)


if __name__ == "__main__":
    unittest.main()

