"""unittest checks for ladder problems whose options could not be read.

    python3 -m unittest discover -s ladder/tests

All offline: a fake index and a fake hint server, like test_ladder.py. The
invented problem has options {} (the OCR failed) but still carries a label —
which rung 4 must NOT show in this case.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ladder.ladder import public, rung  # noqa: E402

NO_OPTS = {"id": "no-opts-1", "language": "eng", "level": 3, "year": 2021,
           "text": "Invented target without readable options: share 24 stickers.",
           "options": {}, "label": "B"}


def neighbour(nid, level, text, label="A"):
    return {"id": nid, "language": "eng", "level": level, "text": text,
            "options": {}, "label": label}


HITS = [
    (0.91, neighbour("no-opts-10", 2, "Share 18 stickers among 3 friends, clock faces and angles.")),
    (0.85, neighbour("no-opts-11", 1, "Clock hands make angles; count the stickers to share.")),
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
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class NoOptionsTests(unittest.TestCase):
    def setUp(self):
        self.index = FakeIndex(HITS)

    def test_the_statement_is_still_public(self):
        shown = public(NO_OPTS)
        self.assertEqual(shown["text"], NO_OPTS["text"])
        self.assertEqual(shown["options"], {})
        self.assertNotIn("label", shown)

    def test_rung1_works_without_options(self):
        out = rung(1, NO_OPTS, index=self.index)
        self.assertNotIn("error", out)
        self.assertEqual(out["neighbour"]["id"], "no-opts-10")
        self.assertNotIn("label", out["neighbour"])

    def test_rung2_works_without_options(self):
        out = rung(2, NO_OPTS, index=self.index)
        self.assertNotIn("error", out)
        self.assertTrue(out["topic"])

    def test_rung3_works_without_options(self):
        server = FakeHintServer("Count how many friends share the stickers.")
        self.addCleanup(server.stop)
        out = rung(3, NO_OPTS, index=self.index, url=server.url)
        self.assertEqual(out["source"], "model")   # empty options: nothing to leak
        self.assertTrue(out["hint"])

    def test_rung4_explains_it_cannot_reach_the_solution(self):
        out = rung(4, NO_OPTS, index=self.index)
        self.assertTrue(out["no_options"])
        self.assertIn("cannot reach the solution", out["message"])
        self.assertIn("options", out["message"])
        self.assertNotIn("answer", out)            # never a resolved answer here
        self.assertNotIn("label", out)
        self.assertEqual(out["worked"]["id"], "no-opts-10")   # reference only

    def test_rung4_without_index_still_explains(self):
        out = rung(4, NO_OPTS, index=None)
        self.assertTrue(out["no_options"])
        self.assertIn("cannot reach the solution", out["message"])
        self.assertNotIn("answer", out)
        self.assertNotIn("worked", out)

    def test_rung4_with_options_is_unchanged(self):
        with_opts = dict(NO_OPTS, options={"A": "4", "B": "6", "C": "8", "D": "12", "E": "24"})
        out = rung(4, with_opts, index=self.index)
        self.assertNotIn("no_options", out)
        self.assertEqual(out["answer"], "B")


if __name__ == "__main__":
    unittest.main()
