"""Unit tests for the letter-bias calibration, with simulated responses.

No server, no board needed: the HTTP client and the two LLM entry points of
copilot.pipeline are replaced with fakes, like the ladder tests do.

    python3 -m unittest discover -s reason/tests -v
"""
import json
import math
import os
import sys
import tempfile
import unittest
import urllib.error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import copilot.pipeline as pipeline                          # noqa: E402
from copilot.llm import build_messages                       # noqa: E402
from reason import calibration as cal                        # noqa: E402


def logprobs_response(token, top):
    """A llama-server-style response: one token plus its top_logprobs."""
    entry = {"token": token, "logprob": top[0][1], "top_logprobs": [
        {"token": t, "logprob": v} for t, v in top]}
    return {"choices": [{"message": {"role": "assistant", "content": token},
                         "logprobs": {"content": [entry]}}],
            "timings": {"prompt_n": 42}}


class FakePost:
    """Collects the request payloads and replays queued responses."""

    def __init__(self, responses):
        self.payloads = []
        self.responses = list(responses)

    def __call__(self, url, payload, timeout):
        self.payloads.append(dict(payload))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TokenTests(unittest.TestCase):
    def test_letter_of_token_trims_and_uppercases(self):
        self.assertEqual((cal.letter_of_token("A"), cal.letter_of_token(" A"),
                          cal.letter_of_token("e\n"), cal.letter_of_token("AB"),
                          cal.letter_of_token("D)")), ("A", "A", "E", None, None))

    def test_variants_of_the_same_letter_are_summed(self):
        # two spellings of A: p(A) = 0.7 + 0.1 = 0.8
        grouped = cal.group_letter_logprobs([{"token": " A", "logprob": math.log(0.7)},
                                             {"token": "A", "logprob": math.log(0.1)}])
        self.assertAlmostEqual(math.exp(grouped["A"]), 0.8, places=6)

    def test_non_letters_and_bad_values_are_skipped(self):
        grouped = cal.group_letter_logprobs([
            {"token": "The", "logprob": -0.1}, {"token": "C", "logprob": float("nan")},
            {"token": "B", "logprob": -0.7}, {"token": "E"}])
        self.assertEqual(sorted(grouped), ["B"])

    def test_renormalize_sums_to_one(self):
        probs = cal.renormalize({"A": -0.4, "D": -1.2})
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertEqual(sorted(probs), ["A", "D"])

    def test_empty_entries_give_nothing(self):
        self.assertEqual((cal.group_letter_logprobs([]), cal.group_letter_logprobs(None)),
                         ({}, {}))


class LetterLogprobsTests(unittest.TestCase):
    def tearDown(self):
        cal.post = self._original_post

    def _replace(self, fake):
        self._original_post = cal.post
        cal.post = fake

    def test_grammar_request_with_logprobs(self):
        fake = FakePost([logprobs_response("B", [("B", -0.4), ("A", -1.2), ("E", -2.0)])])
        self._replace(fake)
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        grouped = cal.letter_logprobs(messages, timeout=5)
        payload = fake.payloads[0]
        self.assertEqual(payload["grammar"], "root ::= [A-E]")
        self.assertEqual((payload["max_tokens"], payload["temperature"]), (1, 0))
        self.assertEqual(payload["seed"], 42)
        self.assertTrue(payload["logprobs"] and payload["top_logprobs"] == 5)
        self.assertTrue(payload["cache_prompt"])
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(grouped, {"B": -0.4, "A": -1.2, "E": -2.0})

    def test_retry_without_grammar_when_no_letters_come(self):
        # first answer: no logprobs block at all; second: letters without grammar
        fake = FakePost([{"choices": [{"message": {"content": "A"}}]},
                         logprobs_response("A", [("A", -0.2), ("D", -0.3)])])
        self._replace(fake)
        grouped = cal.letter_logprobs([{"role": "user", "content": "u"}])
        self.assertEqual(("grammar" in fake.payloads[0], "grammar" in fake.payloads[1]),
                         (True, False))
        self.assertEqual(grouped, {"A": -0.2, "D": -0.3})

    def test_no_retry_when_disabled(self):
        fake = FakePost([{"choices": [{"message": {"content": "A"}}]}])
        self._replace(fake)
        self.assertIsNone(cal.letter_logprobs([{"role": "user", "content": "u"}],
                                              retry_without_grammar=False))
        self.assertEqual(len(fake.payloads), 1)

    def test_none_when_the_server_ignores_logprobs(self):
        fake = FakePost([{"choices": [{"message": {"content": "C"}}]},
                         {"choices": [{"message": {"content": "C"}}]}])
        self._replace(fake)
        self.assertIsNone(cal.letter_logprobs([{"role": "user", "content": "u"}]))
        self.assertEqual(len(fake.payloads), 2)


class BiasFileTests(unittest.TestCase):
    def setUp(self):
        self._original_post = cal.post
        cal.post = FakePost([])  # never used here, but defined
        self.folder = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.folder.name, "calibration-test.json")

    def tearDown(self):
        cal.post = self._original_post
        self.folder.cleanup()
        cal._BIAS_CACHE.clear()

    def test_prior_shape_roundtrip(self):
        record = {"method": "prior", "date": "2026", "seed": 42, "n": 3,
                  "bias": {"A": -1.0, "B": -2.5, "C": -0.9, "D": -0.4, "E": -1.7}}
        with open(self.path, "w") as f:
            json.dump(record, f)
        self.assertEqual(cal.load_bias(self.path), record["bias"])

    def test_missing_file(self):
        self.assertIsNone(cal.load_bias(self.path))

    def test_corrupt_file(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        self.assertIsNone(cal.load_bias(self.path))

    def test_non_letters_are_dropped(self):
        with open(self.path, "w") as f:
            json.dump({"method": "contextual", "bias": {"A": -0.5, "F": -0.2, "C": "x"}}, f)
        self.assertEqual(cal.load_bias(self.path), {"A": -0.5})


class CorrectedLetterTests(unittest.TestCase):
    def setUp(self):
        self.probs = {"A": 0.6, "B": 0.4, "C": 0.0, "D": 0.0, "E": 0.0}
        self.bias = {"A": -0.5, "B": -2.0, "C": -1.0, "D": -1.0, "E": -1.0}

    def test_bias_changes_the_winner(self):
        # raw argmax is A; log(0.6)-(-0.5) beats B's log(0.4)-(-2.0)? No: -0.011 vs 1.084 -> B
        self.assertEqual(cal.corrected_letter(self.probs, None, self.bias), "B")

    def test_without_bias_the_argmax_is_the_sampled_letter(self):
        self.assertEqual(cal.corrected_letter(self.probs, None,
                        {l: 0.0 for l in "ABCDE"}), "A")

    def test_absent_letters_never_win(self):
        probs = {l: 0.0 for l in "ABCDE"}
        probs["D"] = 0.55
        probs["E"] = 0.45
        bias = {l: -1.0 for l in "ABCDE"}
        bias["D"] = 0.0
        self.assertEqual(cal.corrected_letter(probs, None, bias), "E")

    def test_fewer_than_five_options_limits_candidates(self):
        self.assertEqual(cal.corrected_letter(self.probs, {"B", "C"}, self.bias), "B")
        # E has the top raw probability here but is not an option of this problem
        probs = {"A": 0.05, "B": 0.05, "C": 0.05, "D": 0.05, "E": 0.8}
        bias = {l: 0.0 for l in "ABCDE"}
        self.assertEqual(cal.corrected_letter(probs, {"A", "B", "C"}, bias), "A")

    def test_empty_options_allow_every_letter(self):
        probs = {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.6, "E": 0.1}
        bias = {l: 0.0 for l in "ABCDE"}
        self.assertEqual(cal.corrected_letter(probs, None, bias), "D")

    def test_bias_missing_a_present_letter_falls_back_to_raw(self):
        bias = dict(self.bias)
        del bias["A"]  # A is present in probs, so correction is apples-to-oranges
        self.assertEqual(cal.corrected_letter(self.probs, None, bias), "A")

    def test_no_usable_probs_gives_none(self):
        self.assertIsNone(cal.corrected_letter({}, None, self.bias))
        self.assertIsNone(cal.corrected_letter({l: 0.0 for l in "ABCDE"}, None, self.bias))
        self.assertIsNone(cal.corrected_letter(self.probs, None, None))


class WilsonTests(unittest.TestCase):
    def test_edges(self):
        self.assertEqual(cal.wilson_interval(0, 0), (0.0, 1.0))
        low, _ = cal.wilson_interval(0, 10)
        high, _ = cal.wilson_interval(10, 10, 1.96)
        self.assertEqual(low, 0.0)
        _, high_full = cal.wilson_interval(10, 10)
        self.assertEqual(high_full, 1.0)

    def test_typical_interval(self):
        low, high = cal.wilson_interval(9, 40)  # the organizers' 22.5% baseline
        self.assertAlmostEqual(low, 0.123, places=3)
        self.assertAlmostEqual(high, 0.375, places=3)


PROBLEM = {"id": "p1", "language": "eng", "level": 0, "text": "What is 2 + 2?",
           "options": {"A": "3", "B": "4", "C": "5", "D": "6", "E": "7"}, "label": "B"}
FULL_PROBS = {"A": 0.05, "B": 0.70, "C": 0.05, "D": 0.10, "E": 0.10}
D_BIAS = {"A": 0.0, "B": 0.0, "C": 0.0, "D": -0.5, "E": 0.0}  # trims D only


class SolveFixture(unittest.TestCase):
    """solve() fixtures: the two LLM entry points replaced with fakes."""

    def setUp(self):
        self.config_backup = dict(pipeline.CONFIG)
        self.answer_backup = (pipeline.answer_letter, pipeline.answer_letter_probs)
        self.calls = []

        def answer_letter(problem, examples=(), timeout=60, image_b64=None, url=None):
            self.calls.append(("answer_letter", problem["id"]))
            return "D", {"prompt_n": 10}

        def answer_letter_probs(problem, examples=(), timeout=60, image_b64=None, url=None):
            self.calls.append(("answer_letter_probs", problem["id"]))
            return "D", {"prompt_n": 10}, dict(FULL_PROBS)

        pipeline.answer_letter = answer_letter
        pipeline.answer_letter_probs = answer_letter_probs

    def tearDown(self):
        pipeline.CONFIG.clear()
        pipeline.CONFIG.update(self.config_backup)
        pipeline.answer_letter, pipeline.answer_letter_probs = self.answer_backup
        cal._BIAS_CACHE.clear()


class SolveTests(SolveFixture):
    """Off keeps the old code path; on answers with the corrected argmax."""

    def test_off_by_default_is_the_old_code_path(self):
        pipeline.CONFIG["CALIBRATION_FILE"] = ""
        result = pipeline.solve(PROBLEM, doubt=False)
        self.assertEqual(self.calls, [("answer_letter", "p1")])
        self.assertEqual(result["answer"], "D")
        self.assertNotIn("calibration", result)
        self.assertNotIn("probs", result)

    def test_on_corrects_the_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bias.json")
            with open(path, "w") as f:
                json.dump({"method": "prior", "bias": {"A": -2.9, "B": -0.1, "C": -1.0,
                                                       "D": -0.5, "E": -1.0}}, f)
            pipeline.CONFIG["CALIBRATION_FILE"] = path
            result = pipeline.solve(PROBLEM, doubt=False)
        # score = log(p) - bias: A -2.996+2.9 = -0.096 beats B -0.357+0.1 = -0.257
        self.assertEqual(self.calls, [("answer_letter_probs", "p1")])
        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["calibration"], "applied: A")
        # additive: raw probs are still reported (predict/eval write them only with --doubt)
        self.assertEqual(result["probs"], {"A": 0.05, "B": 0.7, "C": 0.05, "D": 0.1, "E": 0.1})
        self.assertIn("confidence", result)

    def test_on_with_doubt_keeps_confidence_and_corrects(self):
        uniform = {letter: math.log(0.2) for letter in "ABCDE"}
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bias.json")
            with open(path, "w") as f:
                json.dump({"method": "contextual", "bias": uniform}, f)
            pipeline.CONFIG["CALIBRATION_FILE"] = path
            result = pipeline.solve(PROBLEM, doubt=True)
        self.assertEqual(result["answer"], "B")  # a uniform bias keeps the raw argmax
        self.assertEqual(result["calibration"], "applied: B")
        self.assertIn("probs", result)
        self.assertIn("confidence", result)

    def test_on_without_logprobs_keeps_the_letter(self):
        import contextlib
        import io

        def no_probs(problem, examples=(), timeout=60, image_b64=None, url=None):
            self.calls.append(("answer_letter_probs", problem["id"]))
            return "D", {"prompt_n": 10}, None

        pipeline.answer_letter_probs = no_probs
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bias.json")
            with open(path, "w") as f:
                json.dump({"method": "prior", "bias": D_BIAS}, f)
            pipeline.CONFIG["CALIBRATION_FILE"] = path
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = pipeline.solve(PROBLEM, doubt=False)
        self.assertEqual(result["answer"], "D")
        self.assertEqual(result["calibration"], "kept: no logprobs")
        self.assertIn("WARNING calibration", stderr.getvalue())


class SolveEdgeCaseTests(SolveFixture):
    """The border cases of step 4, on top of the same fakes."""

    def test_fewer_than_five_options_restricts_the_choice(self):
        problem = dict(PROBLEM, options={"B": "4", "D": "6", "E": "7"})
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bias.json")
            with open(path, "w") as f:
                json.dump({"method": "prior",
                           "bias": {"A": 0.0, "B": 0.0, "C": 0.0, "D": -1.0, "E": -3.0}}, f)
            pipeline.CONFIG["CALIBRATION_FILE"] = path
            result = pipeline.solve(problem, doubt=False)
        # E: log(.10)+3 = +1.697 beats B (-0.357); A and C are not options of this problem
        self.assertEqual(result["answer"], "E")

    def test_empty_options_allow_every_letter(self):
        def spread(problem, examples=(), timeout=60, image_b64=None, url=None):
            self.calls.append(("answer_letter_probs", problem["id"]))
            return "D", {"prompt_n": 10}, {"A": 0.1, "B": 0.15, "C": 0.2, "D": 0.3, "E": 0.25}

        pipeline.answer_letter_probs = spread
        problem = dict(PROBLEM, options={})
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bias.json")
            with open(path, "w") as f:
                json.dump({"method": "prior", "bias": {"A": 0.0, "B": 0.0, "C": 0.0,
                                                       "D": 0.0, "E": -0.4}}, f)
            pipeline.CONFIG["CALIBRATION_FILE"] = path
            result = pipeline.solve(problem, doubt=False)
        # score log(p)-bias: E -1.386+0.4 = -0.986 beats D -1.204 and C -1.609
        self.assertEqual(result["answer"], "E")

    def test_server_error_keeps_the_fallback(self):
        def broken(problem, examples=(), timeout=60, image_b64=None, url=None):
            raise urllib.error.URLError(OSError(61, "Connection refused"))

        pipeline.answer_letter_probs = broken
        pipeline.CONFIG["CALIBRATION_FILE"] = "whatever.json"
        result = pipeline.solve(PROBLEM, doubt=False)
        self.assertEqual(result["answer"], "D")
        self.assertEqual(result["status"], "error: [Errno 61] Connection refused")

    def test_empty_options_with_no_letters_reported_stays_raw(self):
        def sparse(problem, examples=(), timeout=60, image_b64=None, url=None):
            self.calls.append(("answer_letter_probs", problem["id"]))
            # only one letter usable: the extractor turns it into None upstream,
            # here it is already None before the hook
            return "C", {"prompt_n": 10}, None

        pipeline.answer_letter_probs = sparse
        problem = dict(PROBLEM, options={})
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bias.json")
            with open(path, "w") as f:
                json.dump({"method": "prior", "bias": D_BIAS}, f)
            pipeline.CONFIG["CALIBRATION_FILE"] = path
            result = pipeline.solve(problem, doubt=False)
        self.assertEqual(result["answer"], "C")
        self.assertEqual(result["calibration"], "kept: no logprobs")


class CalibrationPromptTests(unittest.TestCase):
    def test_the_calibration_prompt_is_a_pipeline_prompt(self):
        problem = cal.empty_problem()
        self.assertEqual(problem["text"], "N/A")
        self.assertEqual(problem["options"], {letter: "N/A" for letter in "ABCDE"})
        messages = build_messages(problem)  # the real builder, no fakes
        self.assertEqual(len(messages), 2)
        self.assertIn("N/A\nA) N/A\nB) N/A", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()




