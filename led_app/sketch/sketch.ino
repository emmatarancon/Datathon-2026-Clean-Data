// LED probs: draw the copilot's pA..pE as bars on the 8x13 matrix.
//
// The frame is built in Python (copilot/led_matrix.py) and this sketch only
// draws it, following the official 02-led-matrix-frame example: Python calls
//   Bridge.call("draw_probs", frame_bytes)     // 104 bytes, row-major 8x13
// with values 0 = off and 7 = max brightness. Layout (row 0 top, row 7
// bottom): bars for A-E at columns 2, 4, 6, 8, 10 growing bottom-up, the
// pixel under the chosen letter on row 7, and the bottom-right pixel (7, 12)
// when the answer is correct.
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
