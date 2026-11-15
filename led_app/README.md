# LED probs — pA..pE en vivo en la matriz 13x8

App de App Lab que dibuja las probabilidades del copilot (`pA..pE`) como 5
barras en la matriz LED 13x8 del UNO Q cada vez que `eval.py` evalúa un
problema, sobre una línea base encendida en la fila inferior.

**Patrón del ejemplo oficial `02-led-matrix-frame`**: TODO el dibujo se decide
en Python — `copilot/led_matrix.py` construye el frame de 104 bytes (fila 0
arriba, fila 7 abajo, valores 0 = apagado, 7 = brillo máximo) y el sketch solo
lo dibuja. Así no hay lógica que mantener en C++.

## Layout de la matriz (columnas 0-12 de izquierda a derecha)

- **Línea base**: la fila 7 (la de abajo) está siempre encendida de lado a
  lado, **menos en las columnas de las barras cuya barra tenga altura 0**
  (`p <= 0.02`): ahí se ve un hueco. El píxel `(7, 12)` es parte limpia de la
  línea base (sin indicadores sueltos).
- **Barras A-E** en las columnas `2, 4, 6, 8, 10` (`A=2, B=4, C=6, D=8,
  E=10`, empezando en 0). Las probabilidades se **normalizan para sumar 1**
  y la barra mide `round(p * 8)` píxeles creciendo de abajo hacia arriba
  sobre las 8 filas: 100% = columna completa. Si `p > 2%` y la altura
  redondea a 0, se enciende 1 píxel (fila 7).
- **Orden del buffer**: row-major, `index = row * 13 + col` (fila 0 arriba,
  bytes 91..103 = fila inferior). Si en la placa se vieran los píxeles
  desplazados, compara sin editar código:
  `LED_MATRIX_ORDER=column python3 led_app/test_bridge.py`.

Ejemplo con `{"A":0.05, "B":0.4, "C":0.15, "D":0.3, "E":0.1}` (fila 0 arriba):

```
.............
.............
.............
.............
.............
....#........
....#.#.#.#..
#############   <- línea base + barras (C y E: 1 píxel, D: 2, B: 3)
```

## Cómo se conecta todo

1. `eval.py --led` pide las probabilidades al servidor (`pA..pE`) y llama a
   `copilot/led_matrix.py::show_probs(...)` después de cada problema.
2. `show_probs` prueba el **canal 1** (llamada directa `Bridge.call` cuando
   eval.py corre dentro de la app) y si no está disponible, el **canal 2**:
   `POST` JSON `{"frame": [104 enteros]}` a `http://127.0.0.1:8090/matrix`
   con `timeout=0.25` s. Si nada responde, falla en silencio y la evaluación
   continúa.
3. `python/main.py` (esta app) recibe el POST y encola el frame; el
   `user_loop` de `App.run` lo reenvía al sketch con
   `Bridge.call("draw_probs", frame_bytes)` en el hilo de la app.
4. `sketch/sketch.ino` copia el frame y `matrix.draw(frame)` lo muestra en
   cada vuelta del `loop()` (mismo patrón que el ejemplo oficial).

## Arrancar la app en la placa (paso imprescindible)

Los archivos `led_app/` están en el portátil: **hay que copiarlos a la placa**
y arrancar la app, o no habrá nada escuchando en el 8090 ni sketch que dibuje.
**Ojo: la placa no tiene `rsync`** — usa `scp` (no necesita nada en el remoto):

```bash
# desde el portátil (ajusta la IP; la primera vez crea la app en la placa):
ssh arduino@<IP-placa> "arduino-app-cli app new led-probs"
scp -r led_app/app.yaml led_app/python led_app/sketch \
    arduino@<IP-placa>:/home/arduino/ArduinoApps/led-probs/
# alternativa sin scp, con tar sobre ssh:
# tar -C led_app -cf - app.yaml python sketch | ssh arduino@<IP-placa> \
#     "tar -C /home/arduino/ArduinoApps/led-probs -xf -"

# arrancar (el sketch compila al arrancar: puede tardar ~1 minuto):
ssh arduino@<IP-placa> "arduino-app-cli app start user:led-probs"
# si YA habías arrancado el esqueleto antes de copiar: reinicia para que
# el contenedor coja main.py y el sketch se recompile:
ssh arduino@<IP-placa> "arduino-app-cli app restart user:led-probs"

# el log debe mostrar "LED probs: bridge listening on http://127.0.0.1:8090/matrix"
ssh arduino@<IP-placa> "arduino-app-cli app logs /home/arduino/ArduinoApps/led-probs --all | tail -20"
```

Pistas de que estás mirando el esqueleto vacío y no esta app: el log de
compilación lista las librerías del sketch y no aparece ninguna mención a
nuestro `main.py`, y el arranque no imprime "LED probs: bridge listening".
Recuerda: **solo una app de App Lab activa a la vez** — para otra app que
tengas arrancada.

## Comprobar que el puente funciona (sin eval.py)

La forma sencilla, sin problemas de comillas: usa el script de prueba
(desde la raíz del repo en la placa; imprime `HTTP 200` y enciende la columna
completa de C sobre la línea base):

```bash
python3 led_app/test_bridge.py
```

Si prefieres curl, este debe devolver `200` y encender la columna completa de
C (columna 6) sobre la línea base:

```bash
python3 -c "import json; print(json.dumps({'frame':[7 if c==6 or r==7 and c not in (2,4,6,8,10) else 0 for r in range(8) for c in range(13)]}))" > /tmp/frame.json
curl -s -X POST http://127.0.0.1:8090/matrix -H 'Content-Type: application/json' --data @/tmp/frame.json
```

## Ejecutar la evaluación con la matriz

```bash
# dentro de la placa (SSH), en la carpeta del repo:
python3 eval.py --limit 50 --led          # barras en vivo durante eval.py
LED_MATRIX=1 python3 eval.py --limit 50   # lo mismo sin el flag
LED_MATRIX_DEBUG=1 python3 eval.py --limit 5 --led   # vista ASCII por consola
```

**Acierto/fallo**: si la respuesta es correcta el frame se queda fijo; si es
incorrecta **parpadea rápido** (2,5 ciclos por segundo, dentro del límite del
skill de ≤3 destellos/s) hasta que llega el siguiente problema, sin bloquear
ni retrasar la evaluación (hilo en segundo plano; se detiene solo si el
puente deja de responder).

Si ejecutas `eval.py` **desde el portátil** (con los servidores reenviados por
SSH), apunta el canal 2 a la IP de la placa y haz que la app escuche fuera:

```bash
# en la placa (reinicia la app después):
LED_MATRIX_HOST=0.0.0.0 arduino-app-cli app restart user:led-probs
# desde el portátil:
LED_MATRIX_URL=http://<IP-placa>:8090/matrix python3 eval.py --limit 50 --led
```

Sin `--led` ni `LED_MATRIX=1`, `eval.py` y `predict.py` se comportan exactamente
como antes (el formato del CSV no cambia). Vista previa sin hardware:

```bash
python3 -c "from copilot.led_matrix import render_ascii; \
print(render_ascii({'A':0.05,'B':0.4,'C':0.15,'D':0.3,'E':0.1}, 'B', True))"
```

## Si no ves luces: checklist en orden

1. `LED_MATRIX_DEBUG=1 python3 eval.py --limit 2 --led` — ¿aparece el dibujo
   ASCII en la consola? Si no, `eval.py` no está llamando a la matriz (revisa
   `--led` / `LED_MATRIX=1`).
2. `curl` de la sección anterior en la placa — ¿devuelve `200`? Si da
   "connection refused", la app no está arrancada o los archivos no se
   copiaron a `/home/arduino/ArduinoApps/led-probs/`.
3. ¿Compila el sketch? Mira los logs: un error de compilación deja la app
   parada. Esta app usa solo el patrón del ejemplo oficial, así que compila.
4. ¿Hay otra app de App Lab arrancada? Solo puede haber una.
5. ¿Se ve muy tenue? Los píxeles van a brillo 7 (el máximo con
   `setGrayscaleBits(3)`); comprueba que el sketch de la placa es este.
