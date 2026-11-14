# PROGRESS — rag2 (camino Reason)

## Estado
COMPLETADO EN EL PORTATIL (codigo + tests + mock). Nada commiteado (R1).
Pendiente en la placa: verificar /tokenize, ejecutar tune (dev) y final (val),
decidir la activacion con la regla de activacion y fijar los parametros.

## Hecho
- Lectura completa (sección 2): config.env, copilot/{config,client,llm,embed,rag,pipeline,confidence}.py,
  build_index.py, eval.py, predict.py, data/README.md, .skills/m3kang-copilot/SKILL.md.
- Verificaciones reales en el portátil (macOS, sin placa):
  - NO hay servidores en 127.0.0.1:8081/8082 (curl exit 7, conexión rechazada).
  - NO existe data/index.db en el portátil (solo en la placa) → tuning/val en la placa; aquí --mock y tests.
  - /tokenize (8081): NO comprobable aquí. El código sondea POST /tokenize al arrancar y, si falla,
    estima con caracteres/3,5. Verificar en la placa antes de confiar en el presupuesto.
  - git: en rama reason, diff vacío en código, untracked ajenos: laptop/ocr_v2.py, reason/PROGRESS-ocr.md.

## Datos descubiertos del código y de los ficheros (sin supuestos)
- train.jsonl: 1466 filas, TODAS language=eng, 1466 ids únicos (cada id aparece 1 vez; no hay
  traducciones dentro de train). val.jsonl: 281 filas, todas eng, 0 ids compartidos con train.
- Partición dev por la regla sha1: dev = 277 ids, train_fit = 1189 problemas.
- Opciones vacías (OCR no leyó): 555/1466 en train (37.9%), 114/281 en val (40.6%).
- Esquema EXACTO de index.db (copilot/rag.py): tabla `problems` con
  key TEXT PRIMARY KEY ("id|language"), problem TEXT (JSON completo del problema:
  id, language, level, year, text_only, text, options, label), vec BLOB (array float32, 384 dims).
- Similitud (copilot/rag.py Index.search): embebe format_problem(problem) CON prefijo de query bge,
  trunca a 1400 chars (embed.py MAX_CHARS), producto escalar contra los vectores guardados
  (unitarios => coseno), salta filas cuyo problem["id"] == exclude_id, ordena desc, devuelve top-k
  (score, problem).
- Prompt con ejemplos (copilot/llm.py build_messages): system SYSTEM estable; por cada ejemplo:
  user = format_problem(ex) + "\nAnswer:", assistant = ex["label"]; luego el problema actual como
  user. => El formato compacto de la Técnica 1 se consigue pasando ejemplos con "text" recortado
  (mismo renderizado A) ... / Answer:).
- Recuperación se llama DENTRO de solve() (copilot/pipeline.py) al principio del try, si
  index is not None and k > 0.
- --rag: predict.py y eval.py parsean --rag K (int); si K != 0 cargan copilot.rag.Index(args.index)
  y llaman solve(problem, index=index, k=K). eval.py pasa exclude_id=problem["id"]; predict.py no
  (innecesario: el índice solo contiene train). Ambos llaman a solve => el gancho en solve cubre
  los dos.
- solve() ya devuelve {"answer","seconds","status","prompt_tokens","retrieved","examples"} y captura
  timeout/URLError. FALLBACK=D, MARGIN=3 s, TIME_LIMIT=60.
- config.py: las variables de entorno SOBREESCRIBEN config.env (útil para final.py sin editar el
  fichero). CONFIG es un dict cargado una vez al importar.
- SKILL.md: no hay campo de tema en los datos ("There's no topic field") => la métrica de calidad
  de recuperación de 3.2 usa level+year y solapamiento de números (documentado abajo).
- eval.py baraja con random.Random(42) y corta en --limit: la misma muestra de val en cada run.

## Decisiones
- La puerta (gate) reemplaza a la búsqueda original cuando RAG_MODE != baseline: si nadie pasa tau,
  examples=[] => prompt IDÉNTICO a línea base (mismo build_messages).
- Máximo 2 ejemplos (gate e híbrido); el híbrido no aplica tau (la fusión sustituye a la puerta),
  aplica el mismo presupuesto de tiempo. Documentado en el informe.
- Formato compacto = mismo renderizado que build_messages pero con el enunciado recortado a
  120 tokens (estimados con chars/3,5 => 420 chars, corte en palabra). /tokenize solo se usa para
- dedup SOLO se activa si el límite inferior de Wilson de su precisión en dev supera el acierto
  del modelo en esos mismos problemas; si no, queda desactivada.
- Sin servidores aquí: mock determinista (hash) para embeddings/respuestas. Las medidas reales
  SOLO en la placa. El resultado en val de dedup mide cuántos casi-duplicados de train hay en
  val; el test oculto puede tener más, menos o ninguno.

## Hecho (bloque 3.0) — infraestructura
- rag2/__init__.py, rag2/data.py: load_problems/load_train/load_val (filtro eng), is_dev/dev_ids
  (int(sha1(id)[:8],16)%5==0), split_train, MemIndex (búsqueda línea a línea equivalente a
  copilot.rag.Index.search; from_db filtrando ids de dev SIN recalcular embeddings;
  from_problems para mock/tests), query_embed con gancho _MOCK_EMBED para mock.
- rag2/protocol.py: wilson 95%, paired, describe() (N, aciertos, Wilson, latencia
  media/mediana/peor, letras pred vs true, "diferencia no significativa" si N<100),
  count_tokens (/tokenize sondeado 1 vez por proceso + reserva chars/3,5), truncate_tokens,
  messages_tokens, estimate_latency_seconds (tokens/11 + 1 s), results_path.
- rag2/mock.py: embeddings deterministas (hash de tokens a 384 dims), respuestas deterministas,
  install() parchea copilot.pipeline.answer_letter(_probs) y rag2.data._MOCK_EMBED.
- Tests: rag2/tests/test_data.py, test_protocol.py — PASAN.

## Hecho (bloques 3.1-3.5) — COMPLETO
- rag2/gate.py (3.1): puerta tau (top-5, max 2, solo con opciones leidas), formato compacto
  (enunciado a 120 tokens, chars/3,5, corte en palabra), presupuesto (tokens/11 + 1 s <= 35 s,
  quita el menos parecido primero), orden mas-parecido-ultimo; sin vecinos => prompt identico a
  baseline. Tests: test_gate.py (12).
- rag2/hybrid.py (3.2): BM25 puro (regex del enunciado, k1=1.5, b=0.75, stopwords cortas,
  numeros conservados; "3.5" un token, "1/2" se parte por la especificacion), indice invertido en
  memoria sobre train_fit, cache pickle en data/rag2/bm25.pkl SOLO si construir tarda > 3 s
  (huella ids+longitudes; no guarda texto); fusion min-max del top-20 de cada metodo con
  w*cos+(1-w)*bm25 y RRF k=60; metrica de calidad (a) sin modelo: mismo nivel y ano, o
  solapamiento de numeros del enunciado (Jaccard >= 0.5, no hay clasificador de temas en los
  datos). Tests: test_hybrid.py (13).
- rag2/values.py + rag2/dedup.py (3.3): valores canonicos (1/2 == 0,5 == 0.5; 50% NO; miles
  fuera; coma decimal a punto; Fraction; unidades conservadas) y multiconjunto de numeros del
  enunciado; transferencia del VALOR de la opcion correcta del vecino (label del vecino, nunca el
  del problema actual) con condiciones a-d; el numero de examen inicial ("13.") se quita antes de
  comparar (es la posicion del examen, no del problema; guarda anti-decimales "13.5"). Tests:
  test_values.py (13), test_dedup.py (12).
- Gancho minimo (3.4): copilot/pipeline.py +31/-1 (funcion _rag2 + 6 lineas en solve) y
  config.env +10 (RAG_MODE=baseline, RAG2_TAU, RAG2_W, RAG2_TAU_DUP). Con RAG_MODE=baseline el
  camino es el original byte a byte (test con messages capturados); dedup devuelve la letra sin
  llamar al modelo; cualquier excepcion de rag2 cae al camino original; CSV de caminos
  results/rag2-paths-<hora>.csv (id y camino, sin texto; RAG2_PATHS_FILE lo redirige). Tests:
  test_hook.py (11).
- rag2/tune.py (3.5): barrido SOLO en dev (indice en memoria con vectores de index.db filtrando
  los 277 ids de dev; no recalcula embeddings), --technique gate|hybrid|dedup, --limit 60 por
  defecto, aviso de tiempo antes de ejecutar, CSV en results/rag2-tune-<tecnica>-<hora>.csv.
- rag2/final.py (3.5): UNA pasada por config sobre val COMPLETO por la via oficial
  (copilot.pipeline.solve, igual que predict.py/eval.py), parametros fijados por CLI (nunca
  ajustados mirando val), CSV results/rag2-final-<hora>.csv y resumen results/rag2-resumen.md
  (Wilson, pareadas, latencias, camino dedup).
- Smoke tests en mock OK (dedup/gate/hybrid/final). predict.py con RAG_MODE=baseline sobre
  data/sample_problems.jsonl: CSV identico entre dos corridas (sin servidores => fallback D por
  conexion rechazada, el comportamiento de siempre). 82 tests OK.

## Comandos para retomar
- Portatil: cd /Users/emmataranconestrada/Documents/UPCdades/2n/datathon
  python3 -m unittest discover -s rag2/tests     (82 tests, sin placa)
  python3 rag2/tune.py --technique gate --limit 5 --mock    (humo)
- Placa (copiar rag2/ + el diff de copilot/pipeline.py y config.env):
  python3 -c "import urllib.request,json;print(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8081/tokenize',data=json.dumps({'content':'hello world'}).encode(),headers={'Content-Type':'application/json'})).read())"   # verificar /tokenize
  python3 rag2/tune.py --technique gate   --limit 60     # ~2 h
  python3 rag2/tune.py --technique hybrid --limit 60     # ~1.5 h
  python3 rag2/tune.py --technique dedup  --limit 60     # ~15 min (1 pasada baseline + transferencias gratis)
  python3 rag2/final.py --gate-tau <t> --hybrid-w <w> --tau-dup <td> --limit 0   # ~2.5 h con las 5 configs
  (con RAG_MODE en config.env o por entorno; predict.py oficial: RAG_MODE=gate+dedup python3 predict.py --input X --output Y --rag 1)

## Pendiente (solo placa)
- Verificar POST /tokenize en 8081 (aqui imposible); si falla, el codigo ya usa chars/3,5.
- Ejecutar los tres tune en dev, elegir parametros, y final en val UNA vez.
- Decidir activacion: gate/hybrid solo si superan al baseline con claridad (pareada + Wilson);
  dedup solo si el limite inferior de Wilson de su precision en dev supera al modelo en esos
  mismos problemas. Si nada supera al baseline, se deja RAG_MODE=baseline y se reporta tal cual.
- El peor caso medido de cualquier configuracion debe quedar <= 50 s (el presupuesto estima
  <= 35 s de lectura).

PROGRESO GUARDADO (trabajo de portatil completo)

  el presupuesto del prompt completo si el sondeo tiene éxito.
- Presupuesto: tokens_estimados/11 + 1 s <= 35 s; se quita primero el ejemplo menos parecido.
- Normalización de valores: SOLO 1/2, 0,5 y 0.5 se consideran iguales (50% NO); separadores de
  miles fuera ("1,000"->1000); coma decimal a punto ("0,5"->0.5); números a Fraction canónica;
  unidades se conservan en la comparación ("81 cm²" != "81"). Documentado con tests.
- El CSV de caminos (results/rag2-paths-<hora>.csv) solo contiene id y camino (sin texto).
- dedup SOLO se activa si el límite inferior de Wilson de su precisión en dev supera el acierto
  del modelo en esos mismos problemas; si no, queda desactivada.
- Sin servidores aquí: mock determinista (hash) para embeddings/respuestas. Las medidas reales
  SOLO en la placa.
