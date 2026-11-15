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


def _join_blinker(timeout=2.0):
    """Wait for the module's blinker thread to exit (tests must not leak)."""
    thread = led_matrix._BLINK.get("thread")
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
    return thread is None or not thread.is_alive()


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
        for column in (0, 1, 3, 5, 7, 9, 11, 12):
            self.assertEqual(rows[7][column], "#")  # the horizontal base line
        for column in (4, 6, 8, 10):                # bar columns with p=0 here
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

    def test_normalize_empty_or_unusable_is_clean_zero(self):
        # safeguard: no division by zero, and the layout shows gaps, not garbage
        for probs in (None, {}, {"A": "x"}, {"A": -1}):
            normalized = led_matrix.normalize(probs)
            self.assertEqual(normalized, {l: 0.0 for l in "ABCDE"})
        rows = led_matrix.frame_rows(None)   # must not raise
        self.assertEqual(rows[7][2], ".")    # A gap in the baseline
        self.assertEqual(rows[7][0], "#")    # baseline still lit

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
        _join_blinker()  # a wrong answer starts a blinker: stop it here

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
        received = []
        _join_blinker()  # stale blinkers from earlier tests must not post here

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                received.append({"path": self.path,
                                 "body": json.loads(self.rfile.read(length))})
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
        self.assertEqual(self.received[-1]["path"], "/matrix")
        frame = self.received[-1]["body"]["frame"]  # row-major: byte k ->
        self.assertEqual(len(frame), 8 * 13)        # row k//13, column k%13
        self.assertEqual(frame[7 * 13 + 2], 7)      # A: min pixel at row 7 (5%)
        self.assertEqual(frame[6 * 13 + 2], 0)
        self.assertEqual(frame[5 * 13 + 4], 7)      # B: 3-pixel bar, rows 5..7 (40%)
        self.assertEqual(frame[7 * 13 + 4], 7)
        self.assertEqual(frame[4 * 13 + 4], 0)
        self.assertEqual(frame[7 * 13 + 6], 7)      # C: 1 pixel (15%)
        self.assertEqual(frame[6 * 13 + 8], 7)      # D: 2 pixels, rows 6..7 (30%)
        self.assertEqual(frame[7 * 13 + 10], 7)     # E: 1 pixel (10%)
        for column in (0, 1, 3, 5, 7, 9, 11, 12):   # the lit baseline
            self.assertEqual(frame[7 * 13 + column], 7)
        self.assertEqual(frame[91 + 12], 7)         # (7,12): clean baseline pixel
        self.assertEqual(self.received[-1]["body"]["probs"], [5, 40, 15, 30, 10])
        self.assertEqual(self.received[-1]["body"]["winner"], 1)
        self.assertEqual(self.received[-1]["body"]["correct"], 1)

    def test_buffer_is_row_major(self):
        # with all five letters at 20%: every bar is 2 pixels (rows 6..7) and
        # row 7 is fully lit (baseline + bar bottoms), so the bytes 91..103
        # (the bottom row) are all 7 under the row-major order.
        frame = led_matrix.frame_bytes({"A": 0.2, "B": 0.2, "C": 0.2,
                                        "D": 0.2, "E": 0.2})
        self.assertEqual(len(frame), 104)
        self.assertEqual(frame[6 * 13 + 6], 7)               # row 6, column 6 (C)
        self.assertEqual(frame[6 * 13 + 7], 0)               # gap between bars
        self.assertEqual(frame[0], 0)                        # top-left stays off
        for column in range(13):                             # the whole bottom row
            self.assertEqual(frame[7 * 13 + column], 7)
        self.assertEqual(sum(frame), (5 * 2 + 8) * 7)        # bars + baseline
                                                             # (baseline = 8 cols)

    def test_full_bar_serializes_row_major(self):
        # {"C": 1.0} normalizes to C=100%: the whole column 6 is lit and the
        # other bars are gaps in the baseline (E at 0% -> corner dark).
        frame = led_matrix.frame_bytes({"C": 1.0})
        for row in range(8):
            self.assertEqual(frame[row * 13 + 6], 7)
        self.assertEqual(frame[7 * 13 + 10], 0)              # E at 0%: gap
        self.assertEqual(sum(frame), (8 + 8) * 7)            # bar + baseline

    def test_row_order_fallback(self):
        old = os.environ.get("LED_MATRIX_ORDER")
        os.environ["LED_MATRIX_ORDER"] = "column"
        try:
            frame = led_matrix.frame_bytes({"C": 1.0})
        finally:
            if old is None:
                os.environ.pop("LED_MATRIX_ORDER", None)
            else:
                os.environ["LED_MATRIX_ORDER"] = old
        self.assertEqual(frame[6 * 8], 7)           # column-major: byte 48 = col 6
        self.assertEqual(frame[6], 0)               # row-major spot is dark

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


class ConfidenceSumTest(unittest.TestCase):
    """The probabilities the pipeline hands over always sum to 1.0."""

    def test_extract_letter_probs_sums_to_one(self):
        from copilot.confidence import extract_letter_probs
        response = {"choices": [{"logprobs": {"content": [
            {"top_logprobs": [
                {"token": "D", "logprob": __import__("math").log(0.6)},
                {"token": " A", "logprob": __import__("math").log(0.25)},
                {"token": "A", "logprob": __import__("math").log(0.15)},
                {"token": "foo", "logprob": -1.0},
            ]}]}}]}
        probs = extract_letter_probs(response)
        # confidence renormalizes over the letters it reports (sum 1.0); it
        # keeps only the FIRST variant per letter (" A" wins over "A"), the
        # logaddexp grouping of variants lives in bias/reason calibration.
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=9)
        self.assertAlmostEqual(probs["D"], 0.6 / 0.85, places=9)
        self.assertAlmostEqual(probs["A"], 0.25 / 0.85, places=9)

    def test_led_matrix_renormalizes_to_sum_one(self):
        led = led_matrix.normalize({"A": 0.3, "B": 0.3, "C": 0.3, "D": 0.3, "E": 0.3})
        self.assertAlmostEqual(sum(led.values()), 1.0, places=9)
        self.assertAlmostEqual(led["A"], 0.2, places=9)


class BlinkTest(unittest.TestCase):
    """Wrong answers blink fast; correct answers stay fixed."""

    PROBS = {"A": 0.05, "B": 0.4, "C": 0.15, "D": 0.3, "E": 0.1}

    def setUp(self):
        received = []
        self.assertTrue(_join_blinker())  # no stale blinker from other tests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                received.append({"path": self.path,
                                 "body": json.loads(self.rfile.read(length))})
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.received = received
        self.old_url = os.environ.get("LED_MATRIX_URL")
        self.old_interval = led_matrix.BLINK_INTERVAL_S
        os.environ["LED_MATRIX_URL"] = f"http://127.0.0.1:{self.port}/matrix"
        led_matrix.BLINK_INTERVAL_S = 0.05  # fast toggle, short test

    def tearDown(self):
        with led_matrix._BLINK["lock"]:
            led_matrix._BLINK["gen"] += 1  # stop any running blinker
        _join_blinker()
        self.server.shutdown()
        self.server.server_close()
        if self.old_url is None:
            os.environ.pop("LED_MATRIX_URL", None)
        else:
            os.environ["LED_MATRIX_URL"] = self.old_url
        led_matrix.BLINK_INTERVAL_S = self.old_interval

    def test_wrong_answer_blinks_fast(self):
        self.assertTrue(led_matrix.show_probs(self.PROBS, answer="B", correct=False))
        time.sleep(0.45)  # ~9 phases at the 0.05 s patched interval
        frames = [post["body"]["frame"] for post in self.received]
        self.assertGreaterEqual(len(frames), 4)                # several toggles
        self.assertTrue(any(all(v == 0 for v in f) for f in frames))  # off phase
        self.assertTrue(any(f[5 * 13 + 4] == 7 for f in frames))      # on phase
        # the spec fields ride along on the immediate post (blink posts carry
        # only the frame)
        self.assertEqual(self.received[0]["body"]["winner"], 1)

    def test_correct_answer_stays_fixed(self):
        self.assertTrue(led_matrix.show_probs(self.PROBS, answer="B", correct=True))
        time.sleep(0.4)
        self.assertEqual(len(self.received), 1)  # one frame, no toggling

    def test_unknown_correctness_stays_fixed(self):
        self.assertTrue(led_matrix.show_probs(self.PROBS, answer="B", correct=None))
        time.sleep(0.2)
        self.assertEqual(len(self.received), 1)  # no blink without a verdict

    def test_blinker_stops_when_the_channel_disappears(self):
        os.environ["LED_MATRIX_URL"] = "http://127.0.0.1:9/matrix"  # nothing listens
        self.assertFalse(led_matrix.show_probs(self.PROBS, answer="B", correct=False))
        start = time.perf_counter()
        alive = not _join_blinker(timeout=2.0)
        elapsed = time.perf_counter() - start
        self.assertFalse(alive)          # the worker gave up silently
        self.assertLess(elapsed, 2.0)    # after BLINK_MAX_FAILURES, quickly


if __name__ == "__main__":
    unittest.main()

