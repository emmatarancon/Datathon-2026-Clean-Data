"""Offline unittest checks for the hint ladder: a fake index, a fake model server.

    python3 -m unittest discover -s ladder/tests

Every problem here is invented by the team for the tests: no M3Kang data.
"""
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ladder.ladder import (first_sentence, hint_sentence, leaks,  # noqa: E402
                           neighbourhood, public, rung, topic_of)

TARGET = {"id": "demo-1", "language": "eng", "level": 3, "year": 2020,
          "text": "Invented target: share 24 stickers among 4 friends.",
          "options": {"A": "4", "B": "6", "C": "8", "D": "12", "E": "24"}, "label": "B"}


def neighbour(nid, level, text, label="A"):
    return {"id": nid, "language": "eng", "level": level, "text": text,
            "options": {}, "label": label}


# The target itself (id demo-1, easy score) sits in the index to prove that
# exclude_id keeps a problem from finding itself in another language.
HITS = [
    (0.99, neighbour("demo-1", 1, "The target in another language, must be excluded.")),
    (0.91, neighbour("demo-10", 2, "Share 18 stickers among 3 friends, clock faces and angles.")),
    (0.85, neighbour("demo-11", 1, "Clock hands make angles; count the stickers to share.")),
    (0.80, neighbour("demo-12", 3, "Angles between clock hands, then share the total.")),
]


class FakeIndex:
    """Duck-typed like copilot.rag.Index: search(problem, k, exclude_id)."""

    def __init__(self, rows):
        self.rows = rows

    def search(self, problem, k=1, exclude_id=None):
        scored = [(s, p) for s, p in self.rows if p["id"] != exclude_id]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:k]


class FakeHintServer:
    """A tiny llama-server that answers every chat completion with one text."""

    def __init__(self, content):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class NeighbourhoodTests(unittest.TestCase):
    def setUp(self):
        self.index = FakeIndex(HITS)

    def test_public_strips_the_label(self):
        shown = public(TARGET)
        self.assertNotIn("label", shown)
        self.assertEqual(shown["text"], TARGET["text"])

    def test_neighbourhood_prefers_the_easier_neighbour(self):
        hits, relation = neighbourhood(TARGET, self.index)
        self.assertEqual(relation, "easier")
        self.assertEqual(hits[0][1]["id"], "demo-10")  # the target itself was excluded
        self.assertLess(hits[0][1]["level"], TARGET["level"])

    def test_neighbourhood_same_when_nothing_is_easier(self):
        index = FakeIndex([(0.9, neighbour("demo-20", 3, "Same level angles on clock faces.")),
                           (0.8, neighbour("demo-21", 4, "A harder one about angles."))])
        hits, relation = neighbourhood(TARGET, index)
        self.assertEqual(relation, "same")
        self.assertEqual(hits[0][1]["id"], "demo-20")

    def test_neighbourhood_without_index_is_empty(self):
        self.assertEqual(neighbourhood(TARGET, None), ([], None))

    def test_topic_of_counts_the_shared_words(self):
        hits, _relation = neighbourhood(TARGET, self.index)
        topic = topic_of(hits)
        words = [word for word, _n, _total in topic]
        self.assertIn("clock", words)  # present in both easier neighbours
        self.assertTrue(all(n <= total for _word, n, total in topic))
        self.assertLessEqual(len(topic), 3)


class LeakTests(unittest.TestCase):
    def test_option_values_are_leaks_even_in_a_sentence(self):
        self.assertTrue(leaks("You could compute 24 / 4 first.", TARGET))
        self.assertTrue(leaks("Try splitting them into 8 groups.", TARGET))  # any option's value

    def test_the_correct_letter_is_a_leak(self):
        self.assertTrue(leaks("Option B looks best.", TARGET))
        self.assertTrue(leaks("The answer is b.", TARGET))

    def test_empty_hints_are_unusable(self):
        self.assertTrue(leaks("", TARGET))
        self.assertTrue(leaks("   ", TARGET))

    def test_clean_method_hints_pass(self):
        self.assertFalse(leaks("Count how many friends share the stickers.", TARGET))
        self.assertFalse(leaks("Compare the 3 friends in the easier problem.", TARGET))


class HintTests(unittest.TestCase):
    def test_first_sentence_without_quotes_or_breaks(self):
        self.assertEqual(first_sentence("  \"Line one. Line two\"  "), "Line one.")
        self.assertEqual(first_sentence("no terminator here"), "no terminator here")

    def test_hint_sentence_cleans_up_the_model_text(self):
        server = FakeHintServer(" Count the corners of each shape, then compare the two totals. Extra. ")
        self.addCleanup(server.stop)
        hint, meta = hint_sentence(TARGET, url=server.url)
        self.assertEqual(meta, {})
        self.assertEqual(hint, "Count the corners of each shape, then compare the two totals.")


class RungTests(unittest.TestCase):
    def setUp(self):
        self.index = FakeIndex(HITS)
        self.server = FakeHintServer(" Count the corners of each shape, then compare the two totals. Extra. ")
        self.addCleanup(self.server.stop)

    def test_rung3_uses_the_model_when_the_hint_is_clean(self):
        out = rung(3, TARGET, index=self.index, url=self.server.url)
        self.assertEqual(out["source"], "model")
        self.assertEqual(out["hint"], "Count the corners of each shape, then compare the two totals.")
        self.assertNotIn("label", out)

    def test_rung3_falls_back_when_the_hint_leaks(self):
        server = FakeHintServer("The answer is 6 stickers, so option B.")
        self.addCleanup(server.stop)
        out = rung(3, TARGET, index=self.index, url=server.url)
        self.assertEqual(out["source"], "fallback")
        self.assertIn("leaked", out.get("reason", ""))
        self.assertNotIn("6", out["hint"])
        self.assertNotIn("label", out)

    def test_rung3_falls_back_when_the_server_is_down(self):
        out = rung(3, TARGET, index=self.index, url="http://127.0.0.1:1/v1/chat/completions",
                   hint_timeout=2.0)
        self.assertEqual(out["source"], "fallback")
        self.assertIn("model server", out.get("reason", ""))

    def test_rung1_and_2_never_carry_the_label(self):
        out1 = rung(1, TARGET, index=self.index)
        self.assertEqual(out1["relation"], "easier")
        self.assertNotIn("label", out1["neighbour"])
        self.assertIn("seconds", out1)
        out2 = rung(2, TARGET, index=self.index)
        self.assertTrue(out2["topic"])
        self.assertNotIn("label", out2)

    def test_rung1_needs_an_index(self):
        self.assertIn("error", rung(1, TARGET, index=None))
        self.assertIn("error", rung(2, TARGET, index=None))

    def test_rung4_reveals_the_answer_only_when_asked(self):
        out = rung(4, TARGET, index=self.index)
        self.assertEqual(out["answer"], "B")
        self.assertEqual(out["worked"]["label"], "A")  # the worked example keeps its label

    def test_unknown_rung_is_an_error(self):
        self.assertIn("error", rung(9, TARGET, index=self.index))


if __name__ == "__main__":
    unittest.main()