// LED probs: draw the copilot's pA..pE as bars on the 8x13 matrix.
//
// The sketch receives one call per evaluated problem through the Bridge:
//   Bridge.call("draw_probs", pA, pB, pC, pD, pE, winnerIdx, correct)
// with the probabilities as ints 0..100. Layout (row 0 top, row 7 bottom):
// - bars at columns 2, 4, 6, 8, 10 (A to E, one column wide, gaps between);
//   a bar of height h = round(p * 7) fills rows 7-h .. 6 (bottom-up), and a
//   probability above 2 (2%) that rounds to 0 still lights one pixel (row 6);
// - row 7: the pixel under the chosen letter, and the bottom-right pixel
//   (7, 12) when the answer is correct.
#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>

Arduino_LED_Matrix matrix;

const uint8_t BAR_COLUMNS[5] = {2, 4, 6, 8, 10};
const uint8_t ROWS = 8;
const uint8_t COLS = 13;

uint8_t frame[ROWS][COLS];

void drawProbs(int pA, int pB, int pC, int pD, int pE, int winnerIdx, int correct) {
  int probs[5] = {pA, pB, pC, pD, pE};
  memset(frame, 0, sizeof(frame));
  for (uint8_t i = 0; i < 5; i++) {
    int p = constrain(probs[i], 0, 100);
    int height = (p * 7 + 50) / 100;          // 0..7, round half up
    if (p > 2 && height == 0) height = 1;     // tiny but non-zero: one pixel
    for (uint8_t row = 7 - height; row < 7; row++) {
      frame[row][BAR_COLUMNS[i]] = 1;
    }
  }
  if (winnerIdx >= 0 && winnerIdx < 5) {
    frame[7][BAR_COLUMNS[winnerIdx]] = 1;     // under the chosen letter
  }
  if (correct) {
    frame[7][COLS - 1] = 1;                   // bottom-right: correct answer
  }
  matrix.renderBitmap(frame, ROWS, COLS);
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();
  Bridge.begin();
  Bridge.provide("draw_probs", drawProbs);
}

void loop() {
  delay(100);  // the frame stays on until the next draw_probs
}
