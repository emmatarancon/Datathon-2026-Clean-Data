"""Tests for copilot/led_matrix.py: the 8x13 layout, the ASCII preview and
both delivery channels. Standard library only, no board and no App Lab.
"""
import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from copilot import led_matrix


class LayoutTest(unittest.TestCase):
    def test_dimensions(self):
        rows = led_matrix.frame_rows({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2})
        self.assertEqual(len(rows), led_matrix.ROWS)          # 8 rows
        self.assertTrue(all(len(r) == led_matrix.COLS for r in rows))  # 13 cols
        self.assertEqual(len(led_matrix.render_ascii({}, "").split("\n")), 8)  # 8 lines

    def test_full_bar_and_columns(self):
        rows = led_matrix.frame_rows({"A": 1.0}, "")  # p=1 -> height 7, rows 0..6
        self.assertEqual(rows[0][2], "#")
        for row in range(7):                    # rows 0..6 lit at column 2
            self.assertEqual(rows[row][2], "#")
        self.assertEqual(rows[7][2], ".")       # row 7 has no bar
        for row in range(7):                    # the other bars are empty
            self.assertEqual(rows[row][4], ".")

    def test_height_rounds_half_up(self):
        rows = led_matrix.frame_rows({"A": 0.5})  # 0.5*7 = 3.5 -> 4 pixels
        self.assertEqual([r[2] for r in rows[:7]].count("#"), 4)
        self.assertEqual(rows[3][2], "#")         # top of the bar at row 3
        self.assertEqual(rows[2][2], ".")

    def test_small_probability_lights_one_pixel(self):
        rows = led_matrix.frame_rows({"A": 0.03})  # 0.03*7 = 0.21 -> 0, but > 2%
        self.assertEqual(rows[6][2], "#")          # one pixel at row 6
        self.assertEqual(rows[5][2], ".")
        rows = led_matrix.frame_rows({"A": 0.0})   # 0%: no pixel at all
        self.assertEqual(rows[6][2], ".")

    def test_winner_and_hit_markers_on_row_7(self):
        probs = {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}
        rows = led_matrix.frame_rows(probs, answer="D", correct=True)
        self.assertEqual(rows[7][8], "^")      # D is the 4th bar column (index 3)
        self.assertEqual(rows[7][12], "*")     # correct answer
        rows = led_matrix.frame_rows(probs, answer="B", correct=False)
        self.assertEqual(rows[7][4], "^")
        self.assertEqual(rows[7][12], ".")     # wrong answer: pixel stays off

    def test_no_winner_without_a_valid_answer(self):
        for answer in ("", "F", None):
            rows = led_matrix.frame_rows({"A": 1.0}, answer=answer)
            self.assertEqual(rows[7].count("^"), 0)

    def test_scale_clamps(self):
        self.assertEqual(led_matrix.scale({"A": 0.1234, "B": -1, "C": 2, "E": "bad"}),
                         [12, 0, 100, 0, 0])
        self.assertEqual(led_matrix.scale(None), [0] * 5)
        self.assertEqual(led_matrix.winner_index("d"), 3)
        self.assertEqual(led_matrix.winner_index(""), -1)


class EnabledTest(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.pop("LED_MATRIX", None)

    def tearDown(self):
        if self._old is not None:
            os.environ["LED_MATRIX"] = self._old
        else:
            os.environ.pop("LED_MATRIX", None)

    def test_environment_wins_and_defaults_to_off(self):
        os.environ["LED_MATRIX"] = "1"
        self.assertTrue(led_matrix.enabled())
        os.environ["LED_MATRIX"] = "0"
        self.assertFalse(led_matrix.enabled())
        os.environ.pop("LED_MATRIX", None)
        self.assertFalse(led_matrix.enabled())  # config.env default is 0


class DebugPrintTest(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.pop("LED_MATRIX_DEBUG", None)

    def tearDown(self):
        if self._old is not None:
            os.environ["LED_MATRIX_DEBUG"] = self._old
        else:
            os.environ.pop("LED_MATRIX_DEBUG", None)

    def test_debug_prints_the_ascii_frame(self):
        import contextlib
        import io
        os.environ["LED_MATRIX_DEBUG"] = "1"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            led_matrix.show_probs({"A": 1.0}, answer="A", correct=False)
        out = buffer.getvalue()
        self.assertIn("#", out)
        self.assertIn("^", out)  # printed even though no channel is reachable
        self.assertNotIn("*", out)

    def test_no_debug_print_by_default(self):
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            led_matrix.show_probs({"A": 1.0}, answer="A", correct=False)
        self.assertEqual(buffer.getvalue(), "")


class HttpFallbackTest(unittest.TestCase):
    """Channel 2: a local HTTP bridge receives the frame and answers 200."""

    def setUp(self):
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                received["path"] = self.path
                received["body"] = json.loads(self.rfile.read(length))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.received = received

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_post_reaches_the_bridge(self):
        old_url = os.environ.get("LED_MATRIX_URL")
        os.environ["LED_MATRIX_URL"] = f"http://127.0.0.1:{self.port}/matrix"
        try:
            ok = led_matrix.show_probs({"A": 0.05, "B": 0.4, "C": 0.15, "D": 0.3, "E": 0.1},
                                       answer="B", correct=True)
        finally:
            if old_url is None:
                os.environ.pop("LED_MATRIX_URL", None)
            else:
                os.environ["LED_MATRIX_URL"] = old_url
        self.assertTrue(ok)
        self.assertEqual(self.received["path"], "/matrix")
        frame = self.received["body"]["frame"]      # row-major 8x13, values 0/7
        self.assertEqual(len(frame), 8 * 13)
        self.assertEqual(frame[6 * 13 + 2], 7)      # A: min pixel at row 6 (5%)
        self.assertEqual(frame[5 * 13 + 2], 0)
        self.assertEqual(frame[4 * 13 + 4], 7)      # B: top of a 3-pixel bar (40%)
        self.assertEqual(frame[3 * 13 + 4], 0)
        self.assertEqual(frame[6 * 13 + 4], 7)      # B base
        self.assertEqual(frame[7 * 13 + 4], 7)      # winner pixel under B
        self.assertEqual(frame[7 * 13 + 12], 7)     # correct answer pixel
        self.assertEqual(frame[0], 0)               # top-left stays off

    def test_missing_app_fails_silently_and_fast(self):
        os.environ["LED_MATRIX_URL"] = "http://127.0.0.1:9/matrix"  # nothing listens
        try:
            start = time.perf_counter()
            ok = led_matrix.show_probs({"A": 0.5, "D": 0.5}, answer="D", correct=True)
            elapsed = time.perf_counter() - start
        finally:
            os.environ.pop("LED_MATRIX_URL", None)
        self.assertFalse(ok)           # silent failure
        self.assertLess(elapsed, 1.5)  # the 0.25 s timeout keeps eval.py moving


if __name__ == "__main__":
    unittest.main()

