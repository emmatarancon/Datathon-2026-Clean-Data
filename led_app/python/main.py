"""LED probs: an HTTP bridge from eval.py --led to the LED matrix.

Runs on the Linux side of the App Lab app. A tiny HTTP server (127.0.0.1:8090
by default; LED_MATRIX_HOST=0.0.0.0 to accept eval.py runs from the laptop)
accepts POST /matrix with {"frame": [104 ints 0..7]} — the display buffer
copilot/led_matrix.py builds — and forwards it to the sketch through
Bridge.call. The Bridge call happens on the app's own thread through the
user_loop of App.run, the documented pattern, and App.run() is the last line.
"""
import json
import os
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

from arduino.app_utils import App, Bridge

HOST = os.environ.get("LED_MATRIX_HOST", "127.0.0.1")
PORT = int(os.environ.get("LED_MATRIX_PORT", "8090"))

FRAMES = queue.Queue(maxsize=4)  # only the latest frames matter during eval


class MatrixHandler(BaseHTTPRequestHandler):
    """POST /matrix: {"frame": [104 ints, 0 = off, 7 = max brightness]}."""

    def do_POST(self):
        status = 400
        try:
            if self.path == "/matrix":
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length) or b"{}")
                frame = bytes(int(v) for v in (data.get("frame") or []))
                if len(frame) != 8 * 13:
                    raise ValueError("frame must have 104 bytes")
                try:  # never blocks the handler: drop a frame if the queue is full
                    FRAMES.put_nowait(frame)
                except queue.Full:
                    pass
                status = 200
            else:
                status = 404
        except (ValueError, TypeError):
            status = 400
        self.send_response(status)
        self.end_headers()

    def log_message(self, *args):  # keep the app log quiet
        pass


def drain():
    """user_loop of App.run: forward queued frames to the sketch on this thread."""
    try:
        while True:
            frame = FRAMES.get_nowait()
            Bridge.call("draw_probs", frame)
    except queue.Empty:
        pass
    except Exception as error:  # a failed call never stops the app
        print("Bridge.call failed:", error, flush=True)


def serve():
    HTTPServer((HOST, PORT), MatrixHandler).serve_forever()


threading.Thread(target=serve, daemon=True).start()
print(f"LED probs: bridge listening on http://{HOST}:{PORT}/matrix", flush=True)
App.run(user_loop=drain)
