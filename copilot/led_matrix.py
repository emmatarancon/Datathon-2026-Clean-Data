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
import threading
import time
import urllib.request

from .config import CONFIG

LETTERS = "ABCDE"
BAR_COLUMNS = (2, 4, 6, 8, 10)  # 0-indexed: A=2, B=4, C=6, D=8, E=10
ROWS, COLS = 8, 13
BRIGHTNESS = 7  # the maximum at the default setGrayscaleBits(3)
DEFAULT_URL = "http://127.0.0.1:8090/matrix"
TIMEOUT_S = 0.25
BLINK_INTERVAL_S = 0.2   # one phase: a full on/off cycle is 2.5 flashes/s
BLINK_MAX_FAILURES = 3   # channel gone: the blinker stops silently
BLINK_MAX_S = 90.0       # never blink longer than a problem plus margin


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
    """The five probabilities rescaled to sum exactly 1 (display only).

    Safeguard against a server that omits letters or sends unusable numbers:
    when probs is empty or the total is <= 0 the result is a clean all-zero
    dict (never a division by zero), which the layout shows as gaps in the
    baseline at the bar columns.
    """
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
    bar columns whose height is 0 (p <= 0.02) — those show as gaps in the
    line. answer/correct are accepted for signature compatibility but are no
    longer drawn: the baseline replaced the old winner/correct marker pixels,
    so (7, 12) is always a clean baseline pixel.
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

    'row' (default): byte k -> row k // 13, column k % 13, i.e. index =
    row * 13 + col with row 0 at the top and column 0 at the left — the row
    7 baseline occupies bytes 91..103. 'column' (byte k -> column k // 8,
    row k % 8) is kept only to compare on the board without editing code.
    """
    return str(os.environ.get("LED_MATRIX_ORDER", "row")).strip().lower()


def frame_bytes(probs, answer="", correct=None):
    """The display buffer the sketch draws: exactly 104 bytes, values 0..7.

    Row-major (index = row * 13 + col, no axis transposition), drawn by the
    sketch as an 8-row x 13-column grid. LED_MATRIX_ORDER=column switches to
    the alternative order for comparison on the board.
    """
    cells = grid(probs, answer, correct)
    if buffer_order() == "column":
        return bytes(cells[row][column] for column in range(COLS) for row in range(ROWS))
    return bytes(value for row in cells for value in row)


def frame_rows(probs, answer="", correct=None):
    """The 8x13 layout as 8 strings ('#' lit, '.' off)."""
    return ["".join("#" if value else "." for value in row)
            for row in grid(probs, answer, correct)]


def render_ascii(probs, answer="", correct=None):
    """The 8x13 frame as text, for LED_MATRIX_DEBUG=1 and the tests."""
    return "\n".join(frame_rows(probs, answer, correct))


def _send(data, extra=None):
    """One draw attempt through both channels; False when both are absent."""
    try:  # channel 1: direct Bridge call inside the App Lab app
        from arduino.app_utils import Bridge
        Bridge.call("draw_probs", data)
        return True
    except Exception:
        pass
    url = os.environ.get("LED_MATRIX_URL", DEFAULT_URL)
    try:  # channel 2: the led_app HTTP bridge
        payload = {"frame": list(data)}
        if extra:
            payload.update(extra)
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=TIMEOUT_S):
            return True
    except Exception:
        return False


_BLINK = {"gen": 0, "thread": None, "lock": threading.Lock()}


def _blink_worker(gen, data):
    """Toggle the frame on/off fast for a WRONG answer, in the background.

    Stops when a newer frame takes over (any show_probs call bumps the
    generation), when the channel disappears (BLINK_MAX_FAILURES) or after
    BLINK_MAX_S. Ends leaving the wrong-answer frame fixed.
    """
    started = time.monotonic()
    failures = 0
    show = True
    dead = False
    while True:
        with _BLINK["lock"]:
            if gen != _BLINK["gen"]:
                return  # a newer frame (or a correct answer) took over
        if time.monotonic() - started > BLINK_MAX_S:
            break
        if not _send(data if show else bytes(len(data))):
            failures += 1
            if failures >= BLINK_MAX_FAILURES:
                dead = True
                break
        else:
            failures = 0
        show = not show
        time.sleep(BLINK_INTERVAL_S)
    if not dead:  # when the blinking ends, leave the wrong frame fixed
        with _BLINK["lock"]:
            fresh = gen == _BLINK["gen"]
        if fresh:
            _send(data)


def show_probs(probs, answer="", correct=None):
    """Draw one frame; True when a channel accepted it. Never raises.

    Scales pA..pE to ints 0..100 (after normalizing them to sum 1), computes
    winner_idx (0..4 or -1) and hit (1 or 0). Channel 1: the App Lab Bridge,
    when eval.py runs inside the app. Channel 2: a POST to the local bridge
    of led_app (SSH/terminal runs), with timeout=0.25 so a missing app never
    delays the evaluation. The HTTP payload carries the spec fields
    ({"probs", "winner", "correct"}) plus the 104-byte frame the sketch draws.

    A WRONG answer (correct is False) blinks fast: a daemon thread toggles
    the frame on/off at 2.5 full flashes per second until the next frame
    takes over, the channel disappears or BLINK_MAX_S passes. A correct
    answer (or None) stays fixed, and no extra thread runs.
    """
    scaled = scale(probs)
    winner = winner_index(answer)
    hit = 1 if correct else 0
    data = frame_bytes(probs, answer, correct)
    if debug():
        print(render_ascii(probs, answer, correct), flush=True)
    with _BLINK["lock"]:  # any new frame replaces the previous blinker
        _BLINK["gen"] += 1
        gen = _BLINK["gen"]
    ok = _send(data, extra={"probs": scaled, "winner": winner, "correct": hit})
    if correct is False:
        thread = threading.Thread(target=_blink_worker, args=(gen, data), daemon=True)
        with _BLINK["lock"]:
            _BLINK["thread"] = thread
        thread.start()
    return ok


def cross_rows():
    """An X centred on the 8x13 grid: both diagonals, row 0 to row 7.

    Used for problems whose options could not be read from the image — a
    shape no probability chart can be confused with.
    """
    cells = [[0] * COLS for _ in range(ROWS)]
    for row in range(ROWS):
        main = round(row * (COLS - 1) / (ROWS - 1))
        cells[row][main] = BRIGHTNESS
        cells[row][COLS - 1 - main] = BRIGHTNESS
    return cells


def render_cross_ascii():
    """The cross as text, for LED_MATRIX_DEBUG=1 and the tests."""
    return "\n".join("".join("#" if value else "." for value in row)
                     for row in cross_rows())


def show_cross():
    """Draw a centred X for a problem whose options could not be read.

    Same channels and non-blocking guarantees as show_probs(); it also bumps
    the frame generation, so any running blinker stops at once. True when a
    channel accepted the frame.
    """
    cells = cross_rows()
    if debug():
        print(render_cross_ascii(), flush=True)
    with _BLINK["lock"]:
        _BLINK["gen"] += 1
    if buffer_order() == "column":
        data = bytes(cells[row][column] for column in range(COLS) for row in range(ROWS))
    else:
        data = bytes(value for row in cells for value in row)
    return _send(data)
