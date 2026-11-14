"""Real-time letter probabilities on the board's 8x13 LED matrix.

Layout (row 0 at the top, row 7 at the bottom):
- one bar per option at columns 2, 4, 6, 8, 10 (A to E, one column wide, dark
  columns between the bars for legibility). A bar of height h = round(p * 7)
  fills rows 7-h .. 6 (bottom-up, rows 0..6 are the bar area); a probability
  above 0.02 that rounds to 0 still lights one pixel (row 6) as a reference;
- row 7 carries the indicators: the pixel under the letter the model chose
  (column 2*i), and the bottom-right pixel (7, 12) when the answer is correct.

Only the standard library (board rule). The frame is built here (104 bytes,
values 0 = off / 7 = max brightness) and the sketch only draws it, following
the proven 02-led-matrix-frame pattern: "everything you show is decided in
Python". Everything is non-blocking: when neither the App Lab Bridge nor the
local HTTP bridge is reachable, show_probs fails silently in under 250 ms.
"""
import json
import math
import os
import urllib.request

from .config import CONFIG

LETTERS = "ABCDE"
BAR_COLUMNS = (2, 4, 6, 8, 10)
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


def grid(probs, answer="", correct=None):
    """The 8x13 brightness grid (0 off, 7 lit), decided entirely in Python."""
    scaled = scale(probs)
    winner = winner_index(answer)
    cells = [[0] * COLS for _ in range(ROWS)]
    for index, column in enumerate(BAR_COLUMNS):
        height = int(math.floor(scaled[index] / 100.0 * 7 + 0.5))
        if scaled[index] > 2 and height == 0:
            height = 1
        for row in range(7 - height, 7):  # bottom-up: h pixels ending at row 6
            cells[row][column] = BRIGHTNESS
    if 0 <= winner < len(BAR_COLUMNS):
        cells[7][BAR_COLUMNS[winner]] = BRIGHTNESS
    if correct:
        cells[7][COLS - 1] = BRIGHTNESS
    return cells


def frame_bytes(probs, answer="", correct=None):
    """The display buffer the sketch draws: 104 bytes, row-major 8x13."""
    return bytes(value for row in grid(probs, answer, correct) for value in row)


def frame_rows(probs, answer="", correct=None):
    """The 8x13 layout as 8 strings ('#' bar, '^' winner, '*' hit, '.' off)."""
    chars = [["." if value == 0 else "#" for value in row]
             for row in grid(probs, answer, correct)]
    winner = winner_index(answer)
    if 0 <= winner < len(BAR_COLUMNS):
        chars[7][BAR_COLUMNS[winner]] = "^"
    if correct:
        chars[7][COLS - 1] = "*"
    return ["".join(cells) for cells in chars]


def render_ascii(probs, answer="", correct=None):
    """The 8x13 frame as text, for LED_MATRIX_DEBUG=1 and the tests."""
    return "\n".join(frame_rows(probs, answer, correct))


def show_probs(probs, answer="", correct=None):
    """Draw one frame; True when a channel accepted it. Never raises.

    Scales pA..pE to ints 0..100, computes winner_idx (0..4 or -1) and hit
    (1 or 0). Channel 1: the App Lab Bridge, when eval.py runs inside the
    app. Channel 2: a POST to the local bridge of led_app (SSH/terminal
    runs), with timeout=0.25 so a missing app never delays the evaluation.
    The HTTP payload carries the spec fields ({"probs", "winner", "correct"})
    plus the 104-byte frame the sketch draws.
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
