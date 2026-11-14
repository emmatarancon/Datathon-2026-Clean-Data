# Progreso de la escalera de pistas
Última actualización: 14/11/2026 18:45 (sesión 2)

## Estado general
Sesión 2 completada: escalera de pistas implementada (core + tests + web) y probada offline: 31 tests con `python3 -m unittest discover -s ladder/tests` en verde y smoke test del servidor con curl. Queda: verificar en la placa con servidores reales y (si existe) conseguir el plan de las secciones 1–8.

## Hecho (verificado)
Estado de partida (no es código de la escalera; sin `git commit` ni `git push`):
- copilot/confidence.py (NUEVO): parsea logprobs a {A..E} renormalizados, veredicto sure/doubt, barras ASCII, tabla de calibración (ECE, Brier) | pruebas: pasan (ladder/tests/unit.py, 23 checks; el comando oficial unittest discover corría 0 tests -> arreglado con ladder/tests/test_doubt.py)
- calibration.py (NUEVO): reimprime la calibración desde un CSV de eval con --doubt | pruebas: pasan (contra CSV del mock)
- copilot/llm.py: función nueva answer_letter_probs() (misma petición + "logprobs": true, "top_logprobs": 5); answer_letter intacta | pruebas: pasan (mock)
- copilot/pipeline.py: solve(..., doubt=None) añade "probs"/"confidence" solo si el detector está encendido; timeout/error sin tocar | pruebas: pasan (respuestas idénticas ON/OFF, SAME_ANSWERS_OK)
- predict.py / eval.py: flag --doubt; columnas pA..pE, top_prob, verdict SOLO si se usa; CSV base con sus columnas exactas de siempre | pruebas: pasan (mock)
- explorer/server.py + explorer/index.html: /api/ask devuelve probs/confidence (null si OFF) y la web dibuja 5 barras A–E | pruebas: pasan (curl ON/OFF; index servido 200)
- config.env: DOUBT_DETECTOR=0 (apagado, así está ahora) y DOUBT_SURE_PROB=0.5 | pruebas: pasan (env override -> True, por defecto -> False)
- ladder/ladder.py (NUEVO): los 4 peldaños — neighbourhood() (vecino más fácil vía Index.search con exclude_id), topic_of() (tema = palabras compartidas por frecuencia de documento, sin modelo), leaks() (anti-fuga), hint_sentence() (1 frase del modelo, sin gramática, max_tokens 45), fallback_hint(), rung(n) (despachador, nunca devuelve label en 1–3) | pruebas: pasan (unittest, 18 tests offline con FakeIndex y servidor falso)
- ladder/__init__.py (NUEVO): paquete | pruebas: pasan (import)
- ladder/tests/test_ladder.py (NUEVO): unittest con FakeIndex (duck-typing de Index.search) y FakeHintServer (HTTP efímero) | pruebas: pasan (18)
- ladder/tests/test_doubt.py (NUEVO): unittest del detector de duda (parser, veredictos, buckets, calibración) | pruebas: pasan (13)
- ladder/tests/unit.py, mock_llm.py, problems.jsonl: scripts de la sesión 1, intactos | pruebas: unit.py sigue pasando (23 checks)
- .skills/m3kang-copilot/SKILL.md y READMEnuestro.txt: documentados (sesión 1) | pruebas: no ejecutadas (docs)

- ladder/server.py (NUEVO): servidor web propio de la escalera en el puerto 8010 (NO toca el explorer): GET /, /api/info, /api/problems, /api/problem (sin label) y POST /api/rung (valida clave y rango 1–4, 400 si no) | pruebas: pasan (smoke con curl: / -> 200, rung 1/2 con error controlado sin índice, rung 3 con fallback anti-fuga ante el mock, rung 4 revela la respuesta, rung 9 -> 400)
- ladder/index.html (NUEVO): página con selector de problemas, 4 botones de peldaños, confirm() antes del peldaño 4, teclas 1–4 y N | pruebas: pasan (servida 200; estructura HTML corregida tras detectar un </main> descolocado)

## Pendiente (en orden)
1. Conseguir el plan (secciones 1–8 y reglas R1–R10 completas) si existe; mientras tanto, diseñar desde "The hint ladder" del README.md: peldaño 1 problema similar resuelto y más fácil (embeddings), peldaño 2 tema/método de los vecinos, peldaño 3 pista de UNA frase del modelo sin filtrar la respuesta, peldaño 4 solución solo bajo petición.
2. Elegir ubicación del código nuevo: solo archivos nuevos (regla: no modificar existentes). SKILL.md sugiere separarlo de predict.py (p. ej. copilot/tutor.py).
3. Peldaño 1: función que dado un problema devuelva el vecino más similar con level inferior, reutilizando Index.search (con exclude_id).
4. Peldaño 2: tema/método deducido de los vecinos (keywords o clusters de embeddings), sin modelo.
5. Peldaño 3: prompt corto de 1 frase; comprobar EN CÓDIGO que la pista no contiene el valor de la opción correcta (regla de SKILL.md).
6. Peldaño 4: solución solo a petición explícita.
7. Medición: tiempo por peldaño; chequeo de fugas de respuesta en todo val; juicio manual de 20 vecinos (¿de verdad más fácil y mismo tema?).
8. (Según plan) UI en el explorer y/o LED matrix; usar la skill unoq-hardware-ui si toca hardware.

## Decisiones tomadas
- Detector de duda APAGADO por defecto (DOUBT_DETECTOR=0 en config.env): el run oficial y los CSV no cambian ni una columna; se enciende por config, por entorno o con --doubt. Motivo: el contrato de predict.py no se puede romper.
- Todo aditivo: answer_letter y el CSV oficial intactos; solve() solo añade claves cuando el detector está encendido; las funciones nuevas nunca reemplazan a las viejas.
- Probabilidades renormalizadas sobre las letras que reporta el servidor (válido si vienen crudas o ya enmascaradas por la gramática); con menos de 2 letras -> None, para no fingir seguridad.
- Umbral sure=0.5 configurable (DOUBT_SURE_PROB): con 5 letras la probabilidad mínima del máximo es 0.2 (uniforme), así que el tramo [0.0-0.2) es inalcanzable por construcción.
- La escalera irá en archivos nuevos porque las reglas prohíben modificar existentes; cuando llegue el plan de secciones, ese plan manda sobre esta lista.
- Tema de los vecinos (rung 2) por frecuencia de DOCUMENTO (en cuántos vecinos aparece cada palabra, máximo 5): una palabra repetida en un solo problema no domina.
- Anti-fuga (rung 3) más estricto que SKILL.md: se rechaza la pista si contiene el valor de CUALQUIER opción (no solo la correcta) o la letra correcta como palabra suelta; al rechazar o si el modelo falla, se cae al fallback del peldaño 2 (sin modelo).
- Rung 3 SIN gramática (es texto libre, lo contrario del copilot): max_tokens 45 lo mantiene corto; temperature 0 y seed fija como el resto.
- La web de la escalera es un servidor NUEVO (ladder/server.py, puerto 8010) porque el explorer no se puede tocar; reutiliza copilot.client/config/rag y ladder.ladder.

## Datos descubiertos del código existente
- data/index.db (SQLite, lo escribe copilot/rag.py build()): tabla problems(key TEXT PRIMARY KEY = "id|language", problem TEXT = JSON completo del problema, vec BLOB = float32, 384 dims de bge-small). Lo genera build_index.py desde data/train.jsonl. NUNCA commitear data/*.jsonl, data/images/ ni data/index.db.
- copilot/rag.py: Index(db_path).search(problem, k=1, exclude_id=None) -> [(score, problema_dict)], coseno, mejor primero; exclude_id salta el mismo problema en otras lenguas (mantenerlo SIEMPRE o la validación miente). build(db_path, problems, batch=16) es generador.
- copilot/llm.py: format_problem(problem, sees_image=False) -> str; build_messages(problem, examples=(), image_b64=None) -> lista de mensajes; answer_letter(problem, examples=(), timeout=60, image_b64=None, url=URL) -> (letra, timings); answer_letter_probs(...) -> (letra, timings, probs|None). Petición: grammar "root ::= [A-E]", max_tokens 1, temperature 0, seed=CONFIG["SEED"], cache_prompt True (SYSTEM estable porque se cachea).
- copilot/pipeline.py: solve(problem, index=None, k=0, exclude_id=None, image_b64=None, llm_url=URL, time_limit=TIME_LIMIT, doubt=None) -> {"answer","seconds","status","prompt_tokens","retrieved","examples"} (+ "probs","confidence" si doubt). MARGIN=3 s; FALLBACK_ANSWER=D en timeout/error.
- copilot/embed.py: embed(texts, query=False); las queries llevan el prefijo bge de config.env, los documentos no. copilot/config.py: CONFIG (parsea config.env; las variables de entorno ganan) y ROOT. copilot/client.py: post(url, payload, timeout), get(url, timeout=5). Solo stdlib: sin dependencias nuevas en la placa.
- Formato de problema: {"id","language","level"(int),"year","text_only","text","options"({A..E} u {}),"label"}; clave "id_language" en el explorer, "id|language" en el índice. options={} si el OCR no leyó (el modelo solo ve el enunciado).
- Puertos: LLM 8081, EMBED 8082, VISION 8083; explorer por defecto 8000. TIME_LIMIT_SECONDS=60, SEED=42, FALLBACK_ANSWER=D, DOUBT_DETECTOR=0, DOUBT_SURE_PROB=0.5, LLAMA_CPP_REF=2cbe0ec429b849a88664e0f62388fc92a5abd699.
- Rendimiento en placa (README): prefill ~11 tok/s, generación ~6 tok/s; 1 problema ~9 s; 2 ejemplos RAG ~30 s. Por eso los peldaños 1–2 deben ser solo embeddings + código, y el 3 una frase corta.
- eval.py escribe results/ RELATIVO al cwd y añade una línea a results/ablations.csv; comparar con --limit 50 (seed 42, misma muestra siempre).
- .gitignore: cubre data/* (excepto sample_problems.jsonl y README.md), results/* (salvo .gitkeep), *.gguf, *.db, __pycache__, .venv, .DS_Store. NO cubre data_eng/ ni *.jsonl fuera de data/. En este Mac data/ contiene train.jsonl y val.jsonl reales (no salen en git status: gitignore funciona ✓).
- unittest discover -s ladder/tests usa el patrón test*.py: los scripts tipo unit.py no se ejecutan con él (por eso corría 0 tests).
- Editor/insert: insert_line inserta ANTES de la línea indicada (importante al añadir bloques al final de un fichero sin \n final).

## Problemas abiertos y cosas no probadas
- El formato REAL de logprobs del llama-server fijado NO está verificado contra el servidor de verdad (solo contra un mock). El parser es tolerante: si la respuesta no encaja devuelve None y todo sigue funcionando. Primera prueba en placa: eval.py --doubt --limit 5.
- Sin probar en la Arduino UNO Q, ni con RAG (index+k con probs encendidas), ni con el servidor de visión.
- Calibración con datos reales pendiente: comprobar si el umbral 0.5 separa de verdad "seguro y acierta" de "duda".
- El chequeo anti-fuga es por SUBCADENA: una pista parafraseada ("twenty" en vez de 24, o "la mitad de 48") puede colarse. Pendiente: lista de sinónimos numéricos o comparación semántica ligera.
- Rungs 1–2 con un index REAL no probados aquí (necesitan el servidor de embeddings 8082): en esta máquina solo probados con FakeIndex. En placa: python3 ladder/server.py --input data/val.jsonl --index data/index.db
- Rung 3 con índice presente pero embeddings caídos: devolvería error en vez del fallback genérico (la búsqueda lanza y rung() la captura como error). Aceptable; revisar si molesta en la demo.
- ladder/tests/ copia lo que estaba en /tmp/doubt-test (efímero). unit.py tiene la ruta absoluta de este repo en sys.path (solo vale en este ordenador). problems.jsonl son 4 problemas sintéticos inventados por nosotros (p1..p4), sin datos reales de M3Kang.
- Git: 9 ficheros modificados + calibration.py, copilot/confidence.py y ladder/* sin commit (prohibido por las reglas; el equipo decide cuándo).
- Las secciones 1–8 del plan y el texto íntegro de R1–R10 no están en esta conversación: pedirlos antes de codificar si existen.
- ATENCIÓN (sesión 2): data_eng/.ocr_cache.jsonl (116 KB) está sin trackear y .gitignore NO lo cubre (solo data/*). Es la caché OCR de prepare_m3kang = datos derivados de M3Kang. NO lo he tocado (regla: no modificar existentes). El equipo debe añadir data_eng/ (o *_eng/, .ocr_cache.jsonl) a .gitignore ANTES de cualquier git add ., o se subiría al repo.

## Comandos para retomar
(Descubiertos y funcionando; desde la raíz del repo /Users/emmataranconestrada/Documents/UPCdades/2n/datathon)
- Compilar todo: python3 -m py_compile copilot/confidence.py copilot/llm.py copilot/pipeline.py predict.py eval.py calibration.py explorer/server.py ladder/__init__.py ladder/ladder.py ladder/tests/test_ladder.py ladder/tests/test_doubt.py
- TESTS OFICIALES de la escalera y del detector (31 tests, pasan): python3 -m unittest discover -s ladder/tests
- Unit tests del detector (23 checks, ahora pasan): python3 ladder/tests/unit.py
- Mock de llama-server (ocupa el puerto 8081; parar con: pkill -f mock_llm.py) + predict ON/OFF:
    python3 ladder/tests/mock_llm.py &
    python3 predict.py --input ladder/tests/problems.jsonl --output /tmp/base.csv
    python3 predict.py --input ladder/tests/problems.jsonl --output /tmp/doubt.csv --doubt
    (las letras de base.csv y doubt.csv deben coincidir; doubt.csv añade pA..pE, top_prob, verdict)
- eval con calibración (escribe en results/ del repo; borrar el CSV de prueba después si no interesa):
    python3 eval.py --input ladder/tests/problems.jsonl --limit 4 --doubt --name prueba
    python3 calibration.py results/eval-*-prueba.csv
- Explorer con barras (con el mock arriba; ON por entorno, OFF sin él):
    DOUBT_DETECTOR=1 python3 explorer/server.py --input ladder/tests/problems.jsonl --port 8000
    curl -s -X POST -H 'Content-Type: application/json' -d '{"key":"p1_eng","rag":0}' http://127.0.0.1:8000/api/ask
    (ON devuelve "probs" y "confidence"; OFF devuelve null)
- Servidor de la escalera (smoke completo, con mock arriba en 8081):
    python3 ladder/tests/mock_llm.py &
    python3 ladder/server.py --input ladder/tests/problems.jsonl --port 8010
    curl -s http://127.0.0.1:8010/api/info
    curl -s -X POST -H 'Content-Type: application/json' -d '{"key":"p1_eng","rung":3}' http://127.0.0.1:8010/api/rung
    curl -s -X POST -H 'Content-Type: application/json' -d '{"key":"p1_eng","rung":4}' http://127.0.0.1:8010/api/rung
    pkill -f 'ladder/server.py'; pkill -f mock_llm.py
    (con el mock, rung 3 SIEMPRE cae al fallback: el mock contesta una letra y leaks() la rechaza — así se prueba el anti-fuga)
- En la PLACA (escalera): systemctl is-active math-llm math-embed && python3 ladder/server.py --input data/val.jsonl --index data/index.db
- En la PLACA (detector de duda): systemctl is-active math-llm math-embed && python3 eval.py --doubt --limit 5 --name placa-doubt