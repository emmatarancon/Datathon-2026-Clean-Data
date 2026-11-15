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

    def test_full_bar_over_all_eight_rows(self):
        rows = led_matrix.frame_rows({"A": 1.0}, "")  # p=1 -> height 8, rows 0..7
        for row in range(8):
            self.assertEqual(rows[row][2], "#")
        self.assertEqual(rows[7][0], "#")       # baseline lit outside the bars
        for row in range(8):                    # B's column: p=0, gap everywhere
            self.assertEqual(rows[row][4], ".")

    def test_height_rounds_half_up(self):
        # all five letters present (sum 1.0): 0.5 -> 0.5*8 = 4 pixels, rows 4..7
        rows = led_matrix.frame_rows({"A": 0.5, "B": 0.2, "C": 0.1, "D": 0.1, "E": 0.1})
        self.assertEqual([r[2] for r in rows].count("#"), 4)
        self.assertEqual(rows[4][2], "#")         # top of the bar at row 4
        self.assertEqual(rows[3][2], ".")

    def test_small_probability_lights_one_pixel(self):
        probs = {"A": 0.03, "B": 0.24, "C": 0.24, "D": 0.24, "E": 0.25}
        rows = led_matrix.frame_rows(probs)       # 0.03*8 = 0.24 -> 0, but > 2%
        self.assertEqual(rows[7][2], "#")          # one pixel at row 7
        self.assertEqual(rows[6][2], ".")
        rows = led_matrix.frame_rows({"A": 0.0, "B": 0.25, "C": 0.25,
                                      "D": 0.25, "E": 0.25})   # ~0%: gap
        self.assertEqual(rows[7][2], ".")
        self.assertEqual(rows[7][0], "#")          # the baseline itself stays lit

    def test_baseline_row_is_always_lit_outside_the_bars(self):
        rows = led_matrix.frame_rows({"A": 0.03})
        for column in (0, 1, 3, 5, 6, 7, 9, 11):
            self.assertEqual(rows[7][column], "#")  # the horizontal base line
        for column in (4, 8, 10, 12):               # bar columns with p=0 here
            self.assertEqual(rows[7][column], ".")

    def test_probs_are_normalized_to_sum_one(self):
        # the five probs sum 0.5: after normalization each is 0.2 -> height 2
        rows = led_matrix.frame_rows({"A": 0.1, "B": 0.1, "C": 0.1,
                                      "D": 0.1, "E": 0.1})
        for column in led_matrix.BAR_COLUMNS:
            self.assertEqual(rows[7][column], "#")
            self.assertEqual(rows[6][column], "#")
            self.assertEqual(rows[5][column], ".")

    def test_no_marker_pixels_on_the_frame(self):
        # the baseline replaced the old winner/correct markers
        rows = led_matrix.frame_rows({"A": 1.0}, answer="A", correct=True)
        self.assertNotIn("^", "\n".join(rows))
        self.assertNotIn("*", "\n".join(rows))

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
        probs = {"A": 0.05, "B": 0.4, "C": 0.15, "D": 0.3, "E": 0.1}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            led_matrix.show_probs(probs, answer="B", correct=False)
        # exactly the ASCII preview of the frame (plus the trailing newline)
        expected = led_matrix.render_ascii(probs, "B", False) + "\n"
        self.assertEqual(buffer.getvalue(), expected)
        self.assertIn("#", expected)  # sanity: the frame has lit pixels

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
        self.assertEqual(frame[7 * 13 + 2], 7)      # A: min pixel at row 7 (5%)
        self.assertEqual(frame[6 * 13 + 2], 0)
        self.assertEqual(frame[5 * 13 + 4], 7)      # B: 3-pixel bar, rows 5..7 (40%)
        self.assertEqual(frame[7 * 13 + 4], 7)
        self.assertEqual(frame[4 * 13 + 4], 0)
        self.assertEqual(frame[7 * 13 + 8], 7)      # C: 1 pixel (15%)
        self.assertEqual(frame[6 * 13 + 10], 7)     # D: 2 pixels, rows 6..7 (30%)
        self.assertEqual(frame[7 * 13 + 12], 7)     # E: 1 pixel (10%)
        for column in (0, 1, 3, 5, 6, 7, 9, 11):    # the lit baseline
            self.assertEqual(frame[7 * 13 + column], 7)

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

