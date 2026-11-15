"""Real-time letter probabilities on the board's 8x13 LED matrix.

Layout (row 0 at the top, row 7 at the bottom):
- one bar per option at columns 2, 4, 8, 10, 12 (A to E, 0-indexed, one
  column wide). The probabilities are normalized to sum exactly 1 and each
  bar grows bottom-up over all 8 rows: height = round(p * 8), so 100% lights
  the whole column; a probability above 0.02 that rounds to 0 still lights
  one pixel (row 7) as a reference;
- row 7 is otherwise a lit baseline (a horizontal line under the chart),
  except at those bar columns — a ~0% bar shows there as a gap in the line.

Only the standard library (board rule). The frame is built here (104 bytes,
values 0 = off / 7 = max brightness) and the sketch only draws it, following
the proven 02-led-matrix-frame pattern: "everything you show is decided in
Python". The buffer is serialized column-major (byte k -> column k // 8,
row k % 8), which is the order matrix.draw() expects on this board — that is
why the last byte is the bottom-right corner. Everything is non-blocking:
when neither the App Lab Bridge nor the local HTTP bridge is reachable,
show_probs fails silently in under 250 ms.
"""
import json
import math
import os
import urllib.request

from .config import CONFIG

LETTERS = "ABCDE"
BAR_COLUMNS = (2, 4, 8, 10, 12)  # 0-indexed, one bar column per option
ROWS, COLS = 8, 13
BRIGHTNESS = 7  # the maximum at the default setGrayscaleBits(3)
DEFAULT_URL = "http://127.0.0.1:8090/matrix"
TIMEOUT_S = 0.25


def enabled():
    """LED_MATRIX=1 in config.env or the environment turns the matrix on."""
    raw = str(os.environ.get("LED_MATRIX", CONFIG.get("LED_MATRIX", "0"))).strip().lower()
    return raw in ("1", "true", "yes", "on")


def debug():
    """LED_MATRIX_DEBUG=1 prints the ASCII preview on every frame."""
    return str(os.environ.get("LED_MATRIX_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on")


def scale(probs):
    """The five probabilities as ints 0..100 (missing letters count as 0)."""
    out = []
    for letter in LETTERS:
        try:
            value = float((probs or {}).get(letter, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        out.append(max(0, min(100, int(math.floor(value * 100 + 0.5)))))
    return out


def winner_index(answer):
    """0..4 for the model's chosen letter, -1 when there is none."""
    letter = str(answer or "").strip().upper()
    return LETTERS.index(letter) if len(letter) == 1 and letter in LETTERS else -1


def normalize(probs):
    """The five probabilities rescaled to sum exactly 1 (display only)."""
    values = []
    for letter in LETTERS:
        try:
            values.append(float((probs or {}).get(letter, 0.0)))
        except (TypeError, ValueError):
            values.append(0.0)
    values = [max(0.0, value) for value in values]
    total = sum(values)
    if total > 0.0:
        values = [value / total for value in values]
    return dict(zip(LETTERS, values))


def grid(probs, answer="", correct=None):
    """The 8x13 brightness grid (0 off, 7 lit), decided entirely in Python.

    Bars grow bottom-up over all 8 rows (height = round(p * 8), with p
    normalized to sum 1), and row 7 is a lit baseline everywhere except the
    bar columns — a ~0% bar shows there as a gap in the line. answer/correct
    are accepted for signature compatibility but no longer drawn: the
    baseline replaced the old winner/correct marker pixels.
    """
    scaled = scale(normalize(probs))
    cells = [[0] * COLS for _ in range(ROWS)]
    for column in range(COLS):  # the baseline, except at the bar columns
        if column not in BAR_COLUMNS:
            cells[7][column] = BRIGHTNESS
    for index, column in enumerate(BAR_COLUMNS):
        height = int(math.floor(scaled[index] / 100.0 * 8 + 0.5))  # 0..8
        if scaled[index] > 2 and height == 0:
            height = 1
        for row in range(8 - height, 8):  # bottom-up: h pixels ending at row 7
            cells[row][column] = BRIGHTNESS
    return cells


def buffer_order():
    """How the 104-byte buffer maps to the matrix (LED_MATRIX_ORDER env).

    'column' (default, what matrix.draw() expects on this board): byte k ->
    column k // 8, row k % 8 — the last byte is the bottom-right corner.
    'row': byte k -> row k // 13, column k % 13 (the raw grid order), kept
    only for comparison/debugging.
    """
    return str(os.environ.get("LED_MATRIX_ORDER", "column")).strip().lower()


def frame_bytes(probs, answer="", correct=None):
    """The display buffer the sketch draws: 104 bytes.

    Serialized column-major by default (byte k -> column k // 8, row k % 8):
    consecutive bytes run DOWN a column and then to the next column, so the
    last byte is the bottom-right corner of the matrix. LED_MATRIX_ORDER=row
    falls back to the raw row-major grid order for comparison.
    """
    cells = grid(probs, answer, correct)
    if buffer_order() == "row":
        return bytes(value for row in cells for value in row)
    return bytes(cells[row][column] for column in range(COLS) for row in range(ROWS))


def frame_rows(probs, answer="", correct=None):
    """The 8x13 layout as 8 strings ('#' lit, '.' off)."""
    return ["".join("#" if value else "." for value in row)
            for row in grid(probs, answer, correct)]


def render_ascii(probs, answer="", correct=None):
    """The 8x13 frame as text, for LED_MATRIX_DEBUG=1 and the tests."""
    return "\n".join(frame_rows(probs, answer, correct))


def show_probs(probs, answer="", correct=None):
    """Draw one frame; True when a channel accepted it. Never raises.

    Scales pA..pE to ints 0..100 (after normalizing them to sum 1), computes
    winner_idx (0..4 or -1) and hit (1 or 0). Channel 1: the App Lab Bridge,
    when eval.py runs inside the app. Channel 2: a POST to the local bridge
    of led_app (SSH/terminal runs), with timeout=0.25 so a missing app never
    delays the evaluation. The HTTP payload carries the spec fields
    ({"probs", "winner", "correct"}) plus the 104-byte frame the sketch draws.
    """
    scaled = scale(probs)
    winner = winner_index(answer)
    hit = 1 if correct else 0
    data = frame_bytes(probs, answer, correct)
    if debug():
        print(render_ascii(probs, answer, correct), flush=True)
    try:  # channel 1: direct Bridge call inside the App Lab app
        from arduino.app_utils import Bridge
        Bridge.call("draw_probs", data)
        return True
    except Exception:
        pass
    url = os.environ.get("LED_MATRIX_URL", DEFAULT_URL)
    try:  # channel 2: the led_app HTTP bridge
        payload = json.dumps({"probs": scaled, "winner": winner, "correct": hit,
                              "frame": list(data)}).encode()
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=TIMEOUT_S):
            return True
    except Exception:
        return False
