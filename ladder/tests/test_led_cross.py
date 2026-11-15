"""unittest checks for copilot.led_matrix.show_cross() (the no-options cross).

    python3 -m unittest discover -s ladder/tests

All offline: the delivery channel is captured, no board and no network.
"""
import contextlib
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from copilot import led_matrix  # noqa: E402


class ShowCrossTest(unittest.TestCase):
    def setUp(self):
        captured = []
        self.captured = captured
        self._old_send = led_matrix._send
        self._old_order = os.environ.get("LED_MATRIX_ORDER")
        self._old_debug = os.environ.pop("LED_MATRIX_DEBUG", None)
        led_matrix._send = lambda data, extra=None: (
            captured.append(bytes(data)) or True)

    def tearDown(self):
        led_matrix._send = self._old_send
        if self._old_order is None:
            os.environ.pop("LED_MATRIX_ORDER", None)
        else:
            os.environ["LED_MATRIX_ORDER"] = self._old_order
        if self._old_debug is not None:
            os.environ["LED_MATRIX_DEBUG"] = self._old_debug
        else:
            os.environ.pop("LED_MATRIX_DEBUG", None)

    def test_sends_104_bytes_with_the_x_corners_lit(self):
        self.assertTrue(led_matrix.show_cross())
        self.assertEqual(len(self.captured), 1)
        frame = self.captured[0]
        self.assertEqual(len(frame), 104)
        self.assertEqual(frame[0], 7)        # (0, 0)
        self.assertEqual(frame[12], 7)       # (0, 12)
        self.assertEqual(frame[91], 7)       # (7, 0)
        self.assertEqual(frame[103], 7)      # (7, 12)
        self.assertEqual(frame[3 * 13 + 5], 7)   # X arm, row 3
        self.assertEqual(frame[3 * 13 + 7], 7)   # X arm, row 3

    def test_it_is_not_a_bar_chart(self):
        led_matrix.show_cross()
        frame = self.captured[0]
        # no bar column with a full vertical run like the probability chart
        self.assertEqual(frame[6 * 13 + 4], 0)   # row 6, column 4 is dark
        self.assertEqual(frame[7 * 13 + 2], 0)   # row 7: the X leaves gaps

    def test_exactly_sixteen_lit_pixels_at_brightness_seven(self):
        led_matrix.show_cross()
        self.assertEqual(sum(self.captured[0]), 16 * 7)  # two diagonals of 8

    def test_debug_prints_the_cross(self):
        os.environ["LED_MATRIX_DEBUG"] = "1"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            led_matrix.show_cross()
        self.assertEqual(buffer.getvalue(),
                         led_matrix.render_cross_ascii() + "\n")
        self.assertEqual(buffer.getvalue().split("\n")[0].count("#"), 2)

    def test_respects_the_buffer_order_switch(self):
        os.environ["LED_MATRIX_ORDER"] = "column"
        led_matrix.show_cross()
        frame = self.captured[0]
        self.assertEqual(frame[0], 7)        # column-major: (0, 0) is byte 0
        self.assertEqual(frame[7], 7)        # (7, 0) is byte 7
        self.assertEqual(sum(frame), 16 * 7)

    def test_delivery_failure_is_silent(self):
        led_matrix._send = lambda data, extra=None: False
        self.assertFalse(led_matrix.show_cross())  # never raises


if __name__ == "__main__":
    unittest.main()
