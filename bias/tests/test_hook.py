"""Tests for the pipeline hook: DEBIAS_MODE=off identity, calibration,
permute, and the no-logprobs fallback. The model is simulated.
"""
import math
import os
import tempfile
import unittest

from copilot import pipeline
from copilot.config import CONFIG
from copilot.llm import build_messages
from bias import permute as permute_mod
from bias.calibration import save_bias


class HookTestBase(unittest.TestCase):
    def setUp(self):
        self.captured = []
        self._saved = dict(CONFIG)
        self._old_answer = pipeline.answer_letter
        self._old_probs = pipeline.answer_letter_probs
        self.problem = {"id": "s1", "language": "eng", "text": "2 + 3 * 4?",
                        "options": {"A": "20", "B": "14", "C": "24", "D": "9", "E": "10"},
                        "label": "B"}

    def tearDown(self):
        pipeline.answer_letter = self._old_answer
        pipeline.answer_letter_probs = self._old_probs
        CONFIG.clear()
        CONFIG.update(self._saved)

    def capture(self, letter="D", probs=None):
        def fake_probs(problem, examples=(), timeout=60, image_b64=None, url=None):
            self.captured.append(build_messages(problem, examples))
            return letter, {"prompt_n": 10}, probs

        def fake_answer(problem, examples=(), timeout=60, image_b64=None, url=None):
            self.captured.append(build_messages(problem, examples))
            return letter, {"prompt_n": 10}

        pipeline.answer_letter = fake_answer
        pipeline.answer_letter_probs = fake_probs


class OffIdentityTest(HookTestBase):
    def test_off_is_identical_to_the_current_flow(self):
        self.capture(letter="D", probs={"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.3, "E": 0.1})
        CONFIG["DEBIAS_MODE"] = "off"
        result = pipeline.solve(self.problem)
        self.assertEqual(result["answer"], "D")
        self.assertNotIn("debias", result)
        self.assertNotIn("probs", result)  # doubt stays off: same output as before
        self.assertEqual(len(self.captured), 1)
        self.assertEqual(self.captured[0], build_messages(self.problem, ()))

    def test_mode_parsing_rejects_unknown(self):
        for raw in ("bogus", "", "OFF-but-wrong"):
            CONFIG["DEBIAS_MODE"] = raw
            self.assertEqual(pipeline.debias_mode(), "off")
        CONFIG["DEBIAS_MODE"] = None
        self.assertEqual(pipeline.debias_mode(), "off")


class CalibrateHookTest(HookTestBase):
    def test_calibrate_changes_the_letter_from_the_bias_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cal.json")
            # D is picked a lot even without content; B the least: after
            # subtracting the prior, B wins over the sampled D
            save_bias({"method": "contextual",
                       "bias": {"A": math.log(0.05), "B": math.log(0.02),
                                "C": math.log(0.2), "D": math.log(0.45)}}, path)
            self.capture(letter="D",
                         probs={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25, "E": 0.0})
            CONFIG["DEBIAS_MODE"] = "calibrate"
            CONFIG["CALIBRATION_FILE"] = path
            result = pipeline.solve(self.problem)
        self.assertEqual(result["answer"], "B")
        self.assertIn("calibrate", result["debias"])

    def test_no_logprobs_falls_back_to_the_raw_letter(self):
        self.capture(letter="D", probs=None)
        CONFIG["DEBIAS_MODE"] = "calibrate"
        CONFIG["CALIBRATION_FILE"] = "nowhere.json"
        result = pipeline.solve(self.problem)
        self.assertEqual(result["answer"], "D")  # original letter, no crash
        self.assertNotIn("debias", result)


class PermuteHookTest(HookTestBase):
    def test_permute_answers_by_content_within_the_budget(self):
        problem = self.problem  # the value "10" sits at E
        self.capture(letter="D", probs={"A": 0.1, "D": 0.9})  # >= 2 letters

        def fake_logprobs(messages, timeout=12.0, url=None, with_grammar=True):
            text = messages[-1]["content"]
            letter = next(l for l in "ABCDE" if f"{l}) 10" in text)
            return {l: math.log(0.9 if l == letter else 0.025) for l in "ABCDE"}, \
                {"prompt_n": 100}

        original = permute_mod.letter_logprobs
        permute_mod.letter_logprobs = fake_logprobs
        CONFIG["DEBIAS_MODE"] = "permute"
        try:
            result = pipeline.solve(problem)
        finally:
            permute_mod.letter_logprobs = original
        self.assertEqual(result["answer"], "E")
        self.assertIn("permute", result["debias"])


if __name__ == "__main__":
    unittest.main()
