// LED probs: draw the copilot's pA..pE as bars on the 8x13 matrix.
//
// The frame is built in Python (copilot/led_matrix.py) and this sketch only
// draws it, following the official 02-led-matrix-frame example: Python calls
//   Bridge.call("draw_probs", frame_bytes)   // 104 bytes, row-major:
// index = row * 13 + col (row 0 top, col 0 left; byte 91..103 = bottom row)
// with values 0 = off and 7 = max brightness. The Bridge hands the bytes as
// std::vector<uint8_t>, copied with memcpy (binary, NOT a C-string), and
// matrix.draw() shows them on the 8x13 grid without transposing axes.
// Layout: bars for A-E at columns 2, 4, 6, 8, 10 growing bottom-up, and a
// lit baseline on row 7 except at bar columns whose height is 0.
#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include <vector>

Arduino_LED_Matrix matrix;

const size_t FRAME_SIZE = 8 * 13;
uint8_t frame[FRAME_SIZE] = {0};

void drawProbs(std::vector<uint8_t> newFrame) {
  memcpy(frame, newFrame.data(), min(newFrame.size(), (size_t)FRAME_SIZE));
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();
  Bridge.begin();
  Bridge.provide("draw_probs", drawProbs);
}

void loop() {
  matrix.draw(frame);
  delay(10);
}
