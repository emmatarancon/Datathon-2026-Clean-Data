# LED probs — pA..pE en vivo en la matriz 8x13

App de App Lab que dibuja las probabilidades del copilot (`pA..pE`) como 5
barras en la matriz LED 8x13 del UNO Q cada vez que `eval.py` evalúa un
problema, con el indicador de la letra elegida y un píxel de acierto.

## Layout de la matriz (fila 0 arriba, fila 7 abajo)

- **Barras A-E** en las columnas `2, 4, 6, 8, 10` (una columna por barra,
  columnas apagadas entre medias). La barra de la probabilidad `p` mide
  `round(p * 7)` píxeles y crece de abajo hacia arriba (filas `7-h .. 6`).
  Si `p > 2%` y la altura redondea a 0, se enciende 1 píxel (fila 6).
- **Fila 7**: píxel bajo la letra elegida por el modelo, y píxel inferior
  derecho `(7,12)` cuando la respuesta es correcta.

```
. # . . # . . # . . # . .      <- barra alta (p grande)
. # . . # . . # . . # . .
. . . . . . . . . . . . .
. # . . . . . . . . . . .
. # . . # . . . . . . . .
. # . . # . . # . . # . .
. # . . # . . # . . # . .      <- base de las barras (fila 6)
. . . . . . . . ^ . . . *      <- letra elegida (D) + acierto
```

## Cómo se conecta todo

1. `eval.py --led` pide las probabilidades al servidor (`pA..pE`) y llama a
   `copilot/led_matrix.py::show_probs(...)` después de cada problema.
2. `show_probs` prueba el **canal 1** (llamada directa `Bridge.call` cuando
   eval.py corre dentro de la app) y si no está disponible, el **canal 2**:
   `POST` JSON a `http://127.0.0.1:8090/matrix` con `timeout=0.25` s.
   Si nada responde, falla en silencio y la evaluación continúa.
3. `python/main.py` (esta app) recibe el POST y lo reenvía al sketch con
   `Bridge.call("draw_probs", pA..pE, winnerIdx, correct)`.
4. `sketch/sketch.ino` renderiza el bitmap 8x13 con `matrix.renderBitmap`.

## Arrancar la app en la placa

Desde App Lab crea/arranca la app, o en la placa:

```bash
arduino-app-cli app new "led-probs"
# copia led_app/ en /home/arduino/ArduinoApps/led-probs/ (app.yaml, python/,
# sketch/, assets/ no se usa: esta app no sirve página web)
arduino-app-cli app start user:led-probs
arduino-app-cli app logs /home/arduino/ArduinoApps/led-probs --all
```

El sketch compila al arrancar (puede tardar un minuto). Comprueba que el
puente HTTP escucha: `curl -s -X POST http://127.0.0.1:8090/matrix \
 -H 'Content-Type: application/json' -d '{"probs":[10,20,30,25,15],"winner":2,"correct":1}'`.

## Ejecutar la evaluación con la matriz

```bash
python3 eval.py --limit 50 --led          # barras en vivo durante eval.py
LED_MATRIX=1 python3 eval.py --limit 50   # lo mismo sin el flag
LED_MATRIX_DEBUG=1 python3 eval.py --limit 5 --led   # vista ASCII por consola
```

Sin `--led` ni `LED_MATRIX=1`, `eval.py` y `predict.py` se comportan exactamente
como antes (el formato del CSV no cambia). Vista previa sin hardware:

```bash
python3 -c "from copilot.led_matrix import render_ascii; \
print(render_ascii({'A':0.05,'B':0.4,'C':0.15,'D':0.3,'E':0.1}, 'B', True))"
```
