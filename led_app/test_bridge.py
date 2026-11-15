#!/usr/bin/env python3
"""Send one test frame to the LED bridge: full-height bar of C (column 8)
over the lit baseline, using the real copilot.led_matrix.py layout.

On the board (from the repo root):
    python3 led_app/test_bridge.py
From the laptop, if the app listens outside (LED_MATRIX_HOST=0.0.0.0):
    LED_MATRIX_URL=http://<board-ip>:8090/matrix python3 led_app/test_bridge.py

Prints the HTTP status and the ASCII preview of what the matrix should show.
No curl and no quoting: pure standard library.
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.led_matrix import frame_bytes, render_ascii

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090/matrix"
PROBS = {"C": 1.0}  # pC = 100%: full-height bar, winner C, correct answer

payload = {"frame": list(frame_bytes(PROBS, answer="C", correct=True)),
           "probs": [0, 0, 100, 0, 0], "winner": 2, "correct": 1}
request = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(request, timeout=3) as response:
        print("HTTP", response.status, "->", URL)
except OSError as error:
    print("FAILED:", error)
    print("No hay nada escuchando en esa URL: arranca la app led-probs en la "
          "placa y mira led_app/README.md ('Si no ves luces').")
    sys.exit(1)

print(render_ascii(PROBS, "C", True))
print("^ esto debe verse en la matriz: barra central completa + indicadores")
