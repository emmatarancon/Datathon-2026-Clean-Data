# PROGRESS — bias (camino Reason): diagnóstico y corrección del sesgo de letra A/D

## Estado
EN CURSO — bloque inicial (infra + diagnóstico). Rama reason ya activa; nada commiteado.
AVISO: el árbol de trabajo contiene TAMBIÉN la entrega anterior sin commitear (rag2/ + gancho
RAG_MODE en copilot/pipeline.py y config.env). Ambos ganchos conviven en pipeline.py.

## Datos descubiertos del código (leído completo; sin supuestos)
- Prompt y grammar: copilot/llm.py build_messages: system estable + ejemplos (user
  format_problem(ex)+"\nAnswer:" / assistant ex["label"]) + problema. Petición: grammar
  "root ::= [A-E]", max_tokens 1, temperature 0, seed=CONFIG["SEED"], cache_prompt true.
- answer_letter_probs = misma petición + "logprobs": true, "top_logprobs": 5; probs de
  copilot.confidence.extract_letter_probs (primer token, agrupa variantes por strip().upper(),
  exige >=2 letras, renormaliza; None si no hay logprobs usables).
- FALLBACK_ANSWER=D (config.env): solve() la escribe al inicializar result y la mantiene en
  timeout/URLError (status="timeout" o "error: ..."); con éxito status="ok".
- eval.py imprime "Letters pred/true" contando result["answer"] vs problem["label"]; baraja con
  random.Random(42) y corta --limit. CSV de predict.py: id,language,answer,seconds,status.
- CONTRADICCIÓN ANOTADA (R2.5): CALIBRATION_FILE= YA EXISTE en config.env y el pipeline YA TIENE
  el gancho _corrected_answer (argmax log p - sesgo) vía reason/calibration.py (comprometido,
  con 31 tests del equipo según reason/PROGRESS-logprobs.md). => No se duplica la clave; solo se
  añade DEBIAS_MODE=off. bias/ solo importa stdlib+copilot (comprobación d), así que el núcleo
  probado se COPIA mínimo a bias/calibration.py (documentado) en vez de importar reason.
- reason/PROGRESS-logprobs.md: trabajo previo del equipo = exactamente la Corrección B, verificado
  en portátil con mock efímero (puerto 8091): las 3 configuraciones de sonda devuelven 5 letras
  contra el mock; pendiente en placa la sonda real y las corridas dev/val.
- Datos medidos hoy: train.jsonl 1466 filas todas eng; dev (sha1%5==0) = 277 ids, train_fit 1189;
  con opciones leídas 911/1466 (62.3%); val 281 eng (114 sin opciones). FALLBACK=D; baseline
  organizadores 22.5% (9/40), 21 D y 14 A predichos.

## Decisiones
- Diagnóstico (fase 1) con pipeline.solve(problem, doubt=True) para tener status/latencia/probs
  de la vía oficial, y peticiones directas (bias.calibration.letter_logprobs) para D4/D6.
- Chi-cuadrado df=4 contra uniforme y contra la distribución verdadera; crítico 9.49 al 5%.
- D4: desplazamientos cíclicos k=1,2,3 sobre problemas con 5 opciones y contenido distinto.
- Gancho: DEBIAS_MODE (off/calibrate/permute/calibrate+permute) en solve(), después del bloque
  de calibración existente; usa answer_letter_probs (misma petición + campos logprobs) para no
  duplicar la lectura; permute pasa presupuesto de tiempo y usa las permutaciones ya hechas.
- bias/ NO importa reason (comprobación d): núcleo copiado y anotado aquí.

## Hecho — COMPLETO EN EL PORTATIL (sin placa)
- bias/stats.py: wilson, chi-cuadrado df=4 (critico 9.49), conteo de letras (agrupa variantes
  strip().upper()), matriz 5x5 con precision/recall, pareada, describe con aviso N<100.
- bias/calibration.py: agrupacion "A"/" A" con logaddexp (nucleo copiado de reason/calibration.py,
  probado por el equipo con 31 tests; bias/ solo importa stdlib+copilot), letter_logprobs que
  devuelve tambien timings (D6), debiased_argmax (letras ausentes excluidas; sesgo incompleto ->
  argmax crudo), calibracion contextual (enunciado/opciones "N/A" en el MISMO formato del pipeline)
  y prior empirico (100 problemas train_fit deterministas), save/load JSON con N, seed y fecha.
- bias/permute.py: desplazamientos ciclicos k=1,2,3, remapeo letra<->contenido correcto con
  opciones en cualquier orden, promedio por contenido, respuesta = letra ORIGINAL del mejor
  contenido, presupuesto de tiempo (si se agota, usa las permutaciones ya hechas).
- bias/hook.py: debias_letter para DEBIAS_MODE calibrate/permute/calibrate+permute; sin logprobs
  o excepcion -> letra original con aviso; permute recibe el presupuesto de tiempo restante.
- Gancho minimo (4.4): copilot/pipeline.py +41/-2 (debias_mode() + _bias_answer() + 8 lineas en
  solve, tras el bloque de calibracion existente) y config.env +6 (DEBIAS_MODE=off;
  CALIBRATION_FILE YA EXISTIA - contradiccion anotada, no duplicada). Con off: byte a byte igual.
- bias/diagnose.py (fase 1): D1 separacion de artefactos (status ok vs reserva; con/sin opciones),
  D2 distribuciones + chi2 contra uniforme y contra la verdad, D3 confusion 5x5 + precision/recall,
  D4 pegajosidad de letra vs consistencia de contenido con k desplazamientos, D5 entropia vs ln 5,
  % top>50%, prob media por letra, D6 latencia y prompt_n original vs desplazado (reuso de
  prefijo), mas Correccion A (letra mas frecuente de train_fit vs FALLBACK=D). CSV + MD en results/.
- bias/final.py (fase 2): una pasada por config sobre val via oficial solve(); Wilson, chi2,
  pareada, letras movidas, reservas; results/bias-final-<hora>.md.
- 28 tests OK en bias/tests (chi2/wilson, agrupacion, argmax con sesgo y letras ausentes,
  remapeo de permutacion, DEBIAS_MODE=off identico, sin logprobs -> camino original). Smoke mock
  de diagnose y final OK. predict.py con DEBIAS_MODE=off y con calibrate-sin-fichero: CSV
  identico entre configs y determinista (sin servidores -> fallback D, como siempre).

## VEREDICTO H1/H2/H3 (fase 1)
PENDIENTE DE PLACA: el diagnostico necesita el servidor real (logprobs y latencias). No se puede
escribir un veredicto con numeros desde el portatil, y NO se activa ninguna correccion sin el.
Antecedentes del equipo (reason/PROGRESS-logprobs.md): la sonda contra mock devuelve las 5 letras
en las 3 configuraciones; queda la sonda real en placa. Lo que diagnose.py imprimira al ejecutar:
- D1 cuantifica H3 (reservas D por status!=ok y letras sin opciones);
- D4 separa H1 (letra pegajosa, contenido inconsistente) de H2 (sigue el contenido);
- D5 decide si el prior es corregible (entropia ~ln 5) o el modelo esta seguro y equivocado.

## Comandos para retomar
- Portatil: python3 -m unittest discover -s bias/tests   (28 tests)
- Placa fase 1: python3 bias/diagnose.py --limit 60       (~15-25 min; avisa antes de lanzar)
- Placa calibracion: python3 -m bias.calibration --method contextual
                    python3 -m bias.calibration --method prior --n 100
- Placa fase 2 (solo con las correcciones que la fase 1 justifique, parametros ya fijados):
  python3 bias/final.py --configs baseline,calibrate,permute \
      --calibration-file results/bias-calibration-contextual.json --limit 0
- Oficial: DEBIAS_MODE=calibrate CALIBRATION_FILE=results/bias-calibration-contextual.json \
    python3 predict.py --input X --output Y
- Correccion A: NO editar config.env; proponer FALLBACK_ANSWER=<letra mas frecuente de train_fit>
  en el informe si H3 pesa (diagnose.py imprime el dato).

## Pendiente (solo placa)
1. Ejecutar diagnose.py y escribir AQUI el veredicto H1/H2/H3 con numeros.
2. Calcular el sesgo (contextual y prior) y medir cada correccion sola en dev frente a baseline.
3. Confirmar las ganadoras UNA vez en val con final.py y decidir si config.env activa algo.
4. Aviso de honestidad: si el acierto sigue en 20-25%, es resultado valido, no fallo.

PROGRESO GUARDADO (trabajo de portatil completo)
