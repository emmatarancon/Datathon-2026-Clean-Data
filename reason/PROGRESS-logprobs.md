# Progreso: corrección del sesgo de letra por log-probabilidades (camino Reason)

Última actualización: 14/11/2026 (sesión 1) — rama `reason` (creada desde main, con los cambios sin commitear de la sesión de la escalera; sin commit ni push, regla C1).

## Estado general
IMPLEMENTACIÓN Y VERIFICACIÓN LOCAL COMPLETAS en el portátil: 31 tests unitarios en verde y end-to-end contra un mock efímero de llama-server en el puerto 8091 (aquí no hay servidores reales: 8081/8082 connection refused). Los pasos 1–5 del plan (sonda real, calibración, dev, val) quedan para ejecutar EN LA PLACA.

## Datos descubiertos del código (leído completo: config.env, copilot/*.py, predict.py, eval.py, build_index.py, data/README.md, SKILL.md)
- `copilot/llm.py`: `answer_letter(problem, examples, timeout, image_b64, url=URL)` -> (letra, timings). URL = `http://127.0.0.1:{LLM_PORT}/v1/chat/completions`. Petición: `messages`, `"grammar": "root ::= [A-E]"`, `max_tokens: 1`, `temperature: 0`, `seed: int(CONFIG["SEED"])`, `cache_prompt: True`. `format_problem` escribe las opciones como `A) valor` y añade una pista si `options` es `{}`. `build_messages(problem, examples=(), image_b64=None)` -> lista de mensajes.
- `answer_letter_probs(...)` (misma firma) ya pide además `"logprobs": True, "top_logprobs": 5` y devuelve `(letra, timings, probs|None)`; probs = `copilot.confidence.extract_letter_probs(response)`. Es la configuración (a) de la sonda: el cliente existente SÍ admite los parámetros extra (`copilot.client.post` manda el payload tal cual).
- `copilot/confidence.py`: `LETTERS = "ABCDE"`; `extract_letter_probs` toma el PRIMER token (`logprobs.content[0].top_logprobs`), agrupa variantes por `token.strip().upper()`, ignora no-letras/NaN, exige >= 2 letras y devuelve probabilidades renormalizadas (0.0 en las no reportadas) o None.
- `copilot/pipeline.py`: `solve(problem, index, k, exclude_id, image_b64, llm_url, time_limit, doubt)` -> `{"answer","seconds","status","prompt_tokens","retrieved","examples"}` (+`probs`,`confidence` si doubt). MARGIN=3 s; en timeout/URLError devuelve FALLBACK (D) y no toca nada más.
- `copilot/client.py`: `post(url, payload, timeout)` genérico — la sonda/calibración lo reutilizan, no hace falta copia.
- `copilot/config.py`: `CONFIG` parsea `config.env` y las variables de entorno GANAN: `CALIBRATION_FILE=... python3 eval.py ...` cambia de calibración sin editar el fichero.
- `predict.py`/`eval.py`: solo llaman a `solve()`; `eval.py` baraja con `random.Random(42)` y corta `--limit N` (misma muestra siempre); imprime letras pred/truth y latencia, pero NO Wilson.
- Datos: portátil tiene data/train.jsonl (42 líneas), val.jsonl (8), sample_problems.jsonl (5, todas con opciones y label); .gitignore cubre data/* y results/*.
- Shape de logprobs esperado: `choices[0].logprobs.content[0] = {"token": "B", "logprob": -0.406, "top_logprobs": [{"token","logprob"}, ...]}`.
- Sesgo de los organizadores (40 eng con opciones): 22,5% (9/40), 21 D y 14 A predichos; respuestas verdaderas repartidas.
- Prior de la escalera (ladder/PROGRESS.md): hipótesis SIN verificar de que la gramática renormaliza los logprobs sobre A-E; la sonda de la placa la decidirá.
- Este portátil no tiene llama-server ni modelo: ni `command -v llama-server` ni `~/models`.

## Decisiones (y por qué)
- Gancho ÚNICO en `copilot/pipeline.py` (convergen predict.py, eval.py y el explorer). Activado por `CALIBRATION_FILE` en config.env, vacío por defecto = comportamiento actual EXACTO (sin import en caliente, sin campos extra en la petición, sin claves nuevas en el resultado ni en el CSV).
- Con CALIBRATION_FILE activo se usa `answer_letter_probs` (misma petición + campos logprobs; la letra muestreada no cambia) y la respuesta final = argmax de (log p − sesgo) sobre las letras válidas; el detector de duda sigue jugando con las probs crudas.
- Sesgo en espacio log: `argmax(log p_l − bias_l)`. (a) contextual: bias_l = log(promedio de p_l con entradas "N/A") — equivale a dividir cada probabilidad por su promedio (mismo argmax que dividir). (b) prior empírico: bias_l = media de las log-prob crudas de l en train con opciones ("restando esa media", tal cual el enunciado).
- `letter_logprobs(prompt_messages)` devuelve las logprobs CRUDAS agrupadas por letra; agrupa variantes "A"/" A" SUMANDO probabilidades (logaddexp); si con gramática no hay letras utilizables reintenta UNA vez sin gramática (la sonda decidirá en la placa).
- Calibración (b) determinista: mismo `random.Random(42).shuffle` que eval.py tomando las ÚLTIMAS n -> calibración y dev quedan disjuntos si n + N <= len(train). Solo problemas con `options`; las labels no se leen.
- Fallbacks de la corrección: sin probs, sin sesgo para ALGUNA letra presente, o sin candidatas -> letra cruda (argmax crudo == letra muestreada a temperatura 0); empates -> primera de A-E (determinista); aviso en stderr en cada caso. Sin problema previo, la reserva D queda intacta porque las excepciones de red saltan antes.

## Hecho (y verificado)
- Ficheros creados: `reason/__init__.py`, `reason/calibration.py`, `reason/probe_logprobs.py`, `reason/eval_summary.py`, `reason/tests/test_calibration.py`, este PROGRESS. Único gancho a ficheros existentes: `copilot/pipeline.py` (import sys, `calibration_file()`, `_corrected_answer()` y 5 líneas en solve()) + clave `CALIBRATION_FILE=` vacía en config.env.
- Tests: `python3 -m unittest discover -s reason/tests` -> Ran 31 tests, OK (agrupación de variantes, argmax con sesgo, letras ausentes, <5 opciones, opciones vacías, servidor sin logprobs, bias corrupto/ausente, solve() ON/OFF, servidor caído, prompt de calibración).
- `python3 -m py_compile` de todo reason/ + copilot/pipeline.py + predict.py + eval.py: OK.
- C3: `python3 predict.py --input data/sample_problems.jsonl` con CALIBRATION_FILE vacío -> respuestas y status IDÉNTICOS al results/smoke.csv previo (5x D, "error: [Errno 61] Connection refused"; solo varía la columna seconds, ruido del reloj).
- Servidor caído + CALIBRATION_FILE roto: misma reserva D, sin crash (la petición falla antes de la corrección).
- Sonda contra el mock (8091): las TRES configuraciones (a grammar, b sin grammar, c /completion nativa) devuelven las 5 letras y el veredicto lo imprime; sin servidor imprime `FAILED: <urlopen error...>` por configuración y un veredicto honesto al final.
- CLI `python3 -m reason.calibration --method contextual --n 3 --out /tmp/calib-contextual.json` contra el mock: JSON con claves [bias, date, mean_logprobs, mean_probs, method, n, requests, seed, url]; bias = log(mean_probs); 3 peticiones en 0,1 s. Sin servidor: `calibration contextual failed: <urlopen error [Errno 61]...> (is the model server on port 8081 up?)` sin traceback. Funciona también como script (`python3 reason/calibration.py`) gracias al bootstrap de sys.path.
- E2E del gancho (mock que muestrea D con A casi a la par): baseline -> D D D D D; con `LLM_PORT=8091 CALIBRATION_FILE=/tmp/calib-contextual.json` -> A A A A A (los scores corregidos quedan en empate 0 y gana la primera letra de A-E; determinista). CSV intacto, una letra por fila.
- eval.py con CALIBRATION_FILE contra el mock: CSV y ablations.csv normales; comparación con `reason/eval_summary.py`: Wilson 95% de 9/40 -> 12,3%–37,5% (verificado a mano), letras pred/truth y línea "Letters moved ... A +5, D -5". Los CSV de prueba del mock se borraron (results/ quedó como al principio).
- Coste de tiempo de la corrección: UNA sola petición más no hay; es la misma con campos logprobs (tiempo ≈ 0 extra; comprobable con la columna seconds del CSV).

## En curso
- nada.

## Pendiente (en la PLACA, en orden)
1. `python3 reason/probe_logprobs.py` -> registrar aquí qué configuración devuelve las 5 letras con el servidor REAL y si vienen renormalizadas sobre A-E (suman 1; mirar la línea "letters in top_logprobs").
2. `python3 -m reason.calibration --method contextual` y `python3 -m reason.calibration --method prior --n 100` (train completo en placa; el del portátil tiene 42 líneas -> --n 15).
3. Dev en train (N=10, UNA vez por variante; con N=10 la diferencia no es significativa salvo desplazamientos grandes): baseline / (a) / (b) mediante CALIBRATION_FILE; comparar con reason/eval_summary.py e informar N, aciertos, Wilson 95%, latencia media y peor, letras pred/true antes y después (C7).
4. Confirmar el mejor UNA vez en val completo: `CALIBRATION_FILE=results/calibration-<mejor>.json python3 eval.py --limit 0 --name val-<mejor>`; decidir si config.env conserva el fichero o queda vacío.
5. Actualizar este PROGRESS con los números y el veredicto de la sonda.

## Comandos para retomar (portátil y placa)
- Portátil: `python3 -m py_compile reason/__init__.py reason/calibration.py reason/probe_logprobs.py reason/eval_summary.py reason/tests/test_calibration.py copilot/pipeline.py && python3 -m unittest discover -s reason/tests`
- Portátil, cadenas de humo: `python3 predict.py --input data/sample_problems.jsonl --output /tmp/after.csv` (todo D como antes)
- En la PLACA (servidores systemd ya encendidos; NO arrancar servidores nuevos):
  - `python3 reason/probe_logprobs.py`
  - `python3 -m reason.calibration --method contextual`
  - `python3 -m reason.calibration --method prior --n 100`
  - dev (3 corridas): `python3 eval.py --input data/train.jsonl --limit 10 --name dev-baseline` ; `CALIBRATION_FILE=results/calibration-contextual.json python3 eval.py --input data/train.jsonl --limit 10 --name dev-contextual` ; `CALIBRATION_FILE=results/calibration-prior.json python3 eval.py --input data/train.jsonl --limit 10 --name dev-prior`
  - `python3 reason/eval_summary.py results/eval-*-dev-*.csv`
  - confirmación: `CALIBRATION_FILE=results/calibration-<mejor>.json python3 eval.py --limit 0 --name val-<mejor>`

PROGRESO GUARDADO
