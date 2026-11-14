"""LED probs: an HTTP bridge from eval.py --led to the LED matrix.

Runs on the Linux side of the App Lab app. A tiny HTTP server on
127.0.0.1:8090 accepts POST /matrix with {"probs": [pA..pE 0..100],
"winner": 0..4|-1, "correct": 0|1} (copilot/led_matrix.py sends it) and
forwards the parameters to the sketch through the Bridge, which draws the
bars on the matrix. App.run() is the last line, as App Lab requires.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

from arduino.app_utils import App, Bridge

HOST, PORT = "127.0.0.1", 8090


class MatrixHandler(BaseHTTPRequestHandler):
    """POST /matrix: {"probs": [5 ints 0..100], "winner": int, "correct": 0|1}."""

    def do_POST(self):
        status = 400
        try:
            if self.path == "/matrix":
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length) or b"{}")
                probs = [int(p) for p in (data.get("probs") or [])][:5]
                probs += [0] * (5 - len(probs))
                winner = int(data.get("winner", -1))
                correct = int(bool(data.get("correct", 0)))
                Bridge.call("draw_probs", *probs, winner, correct)
                status = 200
            else:
                status = 404
        except (ValueError, TypeError):
            status = 400
        self.send_response(status)
        self.end_headers()

    def log_message(self, *args):  # keep the app log quiet
        pass


def serve():
    HTTPServer((HOST, PORT), MatrixHandler).serve_forever()


threading.Thread(target=serve, daemon=True).start()
App.run()
