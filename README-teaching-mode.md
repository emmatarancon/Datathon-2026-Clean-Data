# Teaching Mode — OK / X on LED Matrix

## 📌 ¿Qué se cambió?

Se añadió **interactividad al modo Teaching existente** (`ladder/`) para que el estudiante pueda **seleccionar una respuesta (A–E)** y ver en la pantalla LED del Arduino UNO Q si acertó (**OK** en verde) o no (**X** en rojo).

**Se modificaron exactamente 2 archivos:**
| Archivo | Qué cambió |
|---|---|
| `ladder/server.py` | Se añadió el endpoint `POST /api/submit` + la función `send_to_leds()` que envía "OK" o "X" al microcontrolador vía Bridge |
| `ladder/index.html` | Se añadieron los botones A–E, la lógica de validación, el área de feedback y la barra de estadísticas |
| `ladder/ladder.py` | **NO se tocó** — se mantiene intacto al 100% |

## 🔄 Cómo funciona

```
┌──────────────────┐     POST /api/submit      ┌──────────────────────┐
│   Navegador       │ ────────────────────────  │  ladder/server.py   │
│   (Web UI)        │   {key, answer: "B"}      │  - Compara con      │
│                   │                           │    problem.label    │
│   Usuario hace clic│                           │  - Envía "OK" o "X" │
│   en la opción B  │                           │    vía Bridge       │
└──────────────────┘                           └──────────┬───────────┘
                                                         │
                                                    Bridge.send()
                                                         │
┌──────────────────┐     ──►  Pinta en la matriz LED  ┌─┴───────────┐
│  sketch.ino      │                                 │ Arduino UNO │
│  (Microcontrol.) │                                 │  Q (LEDs)   │
│  - Recibe "OK"   │                                 │             │
│  - Dibuja ✅     │                                 │  ✅ o ❌    │
│  - Recibe "X"    │                                 │  (verde/   │
│  - Dibuja ✗      │                                 │   rojo)     │
└──────────────────┘                                 └─────────────┘
```

### Flujo completo
1. El estudiante selecciona un problema de la lista desplegable
2. Aparecen los **5 botones** A, B, C, D, E
3. El estudiante hace clic en una opción (ej: "B")
4. El servidor compara `B == problem.label`
5. Si coincide: envía `"OK"` al Bridge → los LEDs muestran **✅** (verde)
6. Si no coincide: envía `"X"` al Bridge → los LEDs muestran **❌** (rojo)
7. La Web UI muestra el feedback visual y actualiza el contador de aciertos
8. El estudiante puede seguir escalando la escalera de pistas (rungs 1–4) como antes

## 🚀 Cómo ejecutar

### Opción A: Desde terminal SSH (la más rápida para probar)

```bash
# 1. Conectar a la placa
ssh arduino@<IP>

# 2. Ir al repo
cd ~/datathon-UPC-maths-qualcomm-arduino-unoq

# 3. Ejecutar el server del ladder
python3 ladder/server.py --input data/val.jsonl --index data/index.db

# Deberías ver:
# 281 problems from data/val.jsonl, 911 in the retrieval index
# Language model up (rung 3 needs it); rungs 1-2 need the embeddings server and the index
# Open http://0.0.0.0:8010 (Ctrl+C to stop)
```

Luego abrir en el navegador:
- **Desde la placa**: `http://127.0.0.1:8010`
- **Desde otro dispositivo**: `http://<IP-de-la-placa>:8010`
- **Port forwarding**: `ssh -L 8010:127.0.0.1:8010 arduino@<IP>` → `http://127.0.0.1:8010`

### Opción B: Dentro de App Lab (para deployment)

1. Crear una nueva App en App Lab
2. Añadir `ladder/index.html` como **HTML** (página principal)
3. Añadir `ladder/server.py` como **Python** (servidor)
4. Añadir `ladder/ladder.py` como **Python** (módulo)
5. Configurar el Bridge para permitir comunicación Python ↔ microcontrolador
6. Hacer clic en "Run"

## 📊 Interfaz de usuario

### Lo que ve el estudiante

```
┌──────────────────────────────────────────────────┐
│  Hint ladder                     language model  │
│                                        up ●      │
├──────────────────────────────────────────────────┤
│  Problem  [▼ L0 · lvl-0_2015_1 · eng]           │
│                                                  │
│  1. Find the missing piece for the house.        │
│                                                  │
│  A) Option 1     B) Option 2     C) Option 3    │
│  D) Option 4     E) Option 5                    │
│                                                  │
│  ┌─ Selecciona tu respuesta: ─────────────────┐  │
│  │  [A]  [B]  [C]  [D]  [E]                 │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  ✅ Correcto!                                    │
│     Tu respuesta: B     La correcta: B           │
│                                                  │
│  Aciertos: 3     Total: 5     Precisión: 60%    │
├──────────────────────────────────────────────────┤
│  Climb the ladder                                │
│  [1 · Similar, easier] [2 · The topic]          │
│  [3 · One hint]      [4 · The solution]         │
│                                                  │
│  Rung 1 · 0.23 s                                 │
│  [Problema similar ya resuelto...]               │
└──────────────────────────────────────────────────┘
```

### Teclas de teclado
| Tecla | Acción |
|---|---|
| **1, 2, 3, 4** | Subir un escalón en la escalera de pistas |
| **A, B, C, D, E** | Seleccionar respuesta y enviarla |
| **N** | Pasar al siguiente problema |

## 🔧 Configuración de los LEDs

### sketch.ino necesario (ejemplo mínimo)

Para que los LEDs funcionen, necesitas un sketch que reciba las señales del Bridge:

```cpp
#include <ArduinoLEDs.h>
#include <Bridge.h>

void setup() {
    ArduinoLEDs.begin();
    ArduinoLEDs.clear();
    Bridge.begin();
    ArduinoLEDs.print("OK");  // Pantalla de inicio
    ArduinoLEDs.display();
}

void loop() {
    while (Bridge.available() > 0) {
        String msg = Bridge.readStringUntil('\n');
        msg.trim();
        
        if (msg == "OK" || msg == "ok") {
            // Pantalla verde: acierto
            ArduinoLEDs.clear();
            ArduinoLEDs.color(0, 255, 0);  // Verde
            ArduinoLEDs.print("OK");
            ArduinoLEDs.display();
            delay(3000);  // Mostrar 3 segundos
        }
        else if (msg == "X" || msg == "x") {
            // Pantalla roja: error
            ArduinoLEDs.clear();
            ArduinoLEDs.color(255, 0, 0);  // Rojo
            ArduinoLEDs.print("X");
            ArduinoLEDs.display();
            delay(3000);  // Mostrar 3 segundos
        }
    }
    delay(100);
}
```

### Cómo se envía la señal

En `ladder/server.py`, la función `send_to_leds()` se encarga de esto:

```python
def send_to_leds(result):
    """Envía OK o X al microcontrolador vía Bridge."""
    try:
        from arduino.app_utils import Bridge
        Bridge.send(result)  # envía "OK" o "X"
        print(f"  → LED: {result} (via Bridge)")
    except ImportError:
        # No estamos en App Lab → standalone mode
        print(f"  → LED: {result} (standalone mode)")
```

- **En App Lab**: `Bridge.send()` transmite directamente al microcontrolador
- **En standalone (SSH)**: imprime por stdout para debug

## 📈 Estadísticas

La Web UI muestra en tiempo real:
- **Aciertos**: número de respuestas correctas
- **Total**: número de intentos totales
- **Precisión**: porcentaje `Aciertos / Total * 100`

Se actualizan automáticamente cada vez que el estudiante responde.

## 🐛 Troubleshooting

| Problema | Solución |
|---|---|
| **Los LEDs no muestran nada** | Verifica que el sketch.ino tiene `Bridge.begin()` y `Bridge.available()` en el loop |
| **No aparecen los botones A–E** | Verifica que el problema tiene opciones (`options: {A: ..., B: ..., ...}`) |
| **Error "unknown problem key"** | El problema no está en `data/val.jsonl` o la key está mal formada |
| **La Web UI no carga** | Asegúrate de que `ladder/index.html` esté en la misma carpeta que `server.py` |
| **Bridge.send() no funciona** | Estás en standalone mode (SSH), no en App Lab. Prueba con `ssh -L 8010:127.0.0.1:8010 arduino@<IP>` |

## 📦 Estructura de archivos

```
ladder/
├── server.py      ✏️ Modificado (añadido /api/submit + send_to_leds)
├── index.html     ✏️ Modificado (añadidos botones A–E, feedback, stats)
├── ladder.py      ✅ Intacto (lógica de los 4 rungs)
├── PROGRESS.md
├── tests/
│   ├── problems.jsonl
│   └── test_ladder.py
```

## 💡 Ideas para el futuro

1. **Animaciones en LEDs**: En vez de parpadear OK/X, hacer un efecto de onda o rotación
2. **Contador de intentos por problema**: Mostrar cuántos intentos llevó antes de acertar
3. **Exportar datos**: Guardar todas las respuestas en un archivo CSV para análisis posterior
4. **Modo examen**: Desactivar la escalera de pistas y solo dejar los botones A–E
5. **Timer**: Añadir un temporizador para ver cuánto tarda el estudiante en responder

## ✅ Checklist para la demo

- [ ] El ladder server corre sin errores: `python3 ladder/server.py`
- [ ] La Web UI carga en `http://127.0.0.1:8010`
- [ ] Aparecen los botones A–E debajo del problema
- [ ] Al hacer clic en una respuesta, la Web UI muestra ✅ o ❌
- [ ] Los LEDs de la placa muestran OK (verde) o X (rojo)
- [ ] La barra de estadísticas se actualiza con cada respuesta
- [ ] Los rungs 1–4 siguen funcionando como antes
- [ ] El teclado funciona (teclas 1–4, A–E, N para siguiente)

---

**Nota**: Todos los cambios son no destructivos. El código original de `ladder/ladder.py` y la lógica de los 4 rungs se mantienen intactos. Esta es una extensión sobre la funcionalidad existente, no un reemplazo.