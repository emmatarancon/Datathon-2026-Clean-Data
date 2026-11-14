# Progreso: mejorar la lectura OCR de las opciones A-E (camino Reason, portátil)

Última actualización: 14/11/2026 (sesión 1) — rama `reason`, sin commit ni push (C1).

## Estado general
PASOS 1–5 COMPLETADOS sobre la muestra fija (60 filas eng del shard 0) en el portátil. Variante elegida: **crop+gray+postproc (cf0.4)**: 70,0% de lecturas con 5 opciones (vs 65,0% baseline), 0 lecturas perdidas, ~3× más rápida. Paso 6 (comparación en placa) documentado; el run del dataset COMPLETO (~31 min estimados) espera aprobación del usuario. Ficheros protegidos intactos por mí (C4).

## Datos descubiertos (lecturas completas: laptop/prepare_m3kang.py, laptop/requirements.txt, data/README.md)
- Pipeline existente (prepare_m3kang.py): 1) descarga shards parquet de qualcomm/M3Kang (hf_hub_download, caché en ~/.cache/huggingface), 2) filtra lenguajes, 3) OCR con RapidOCR sobre los BYTES de la imagen (columna `image` del parquet), 4) reparto por id: `int(sha256(id).hexdigest(),16) % 1000 < val_fraction*1000 -> val` (o años completos con --val-years), 5) escribe data/train.jsonl y data/val.jsonl con {id, language, level, year, text_only, text, options, label} y SIN imágenes.
- Parseo de opciones (funciones reutilizables por import): `reading_order(boxes)` ordena filas (por y) y luego x; `is_marker(text, letter, strict)` usa la regex `MARKER = ^\s*([A-E])\s*([\)）\]:.]?)\s*(.*)$` (strict solo para A: sin paréntesis solo acepta letra sola o valor que empieza por número); `parse_options(boxes)` toma la ÚLTIMA secuencia A..E en orden y devuelve {} si no son 5 valores no vacíos ("opciones dibujo"). Ruido que ya maneja: "A)21", "E16", "C"+"13" en cajas separadas, ")768351".
- La caché es append-only `data/.ocr_cache.jsonl` con {"key": "id|language", "options"}; el proceso en marcha la tiene abierta en modo append (por eso PROHIBIDO escribirla).
- `--save-images` guarda PNG a out/images/key.png; `--limit N` toma las PRIMERAS N filas del shard (orden del fichero) tras filtrar lenguaje -> mi muestra fija = primeras N filas eng del shard 0 (determinista, mismas filas que el --limit del script).
- Shard 0 high_quality: 10155 filas, **1747 eng**. requirements.txt: huggingface_hub, pyarrow, rapidocr_onnxruntime (sin Pillow explícita, pero instalada).
- Entorno del portátil: Python 3.11.5 **arm64** Darwin (NO es la causa de la lentitud); rapidocr_onnxruntime 1.4.4; onnxruntime 1.32.0 con providers disponibles [CoreML, Azure, CPU]; opencv_python 5.0.0.93; numpy 2.4.6; Pillow 13.0.0 (todas YA instaladas -> no hace falta requirements-ocr.txt ni permiso de pip).
- **Hallazgo del backend**: `rapidocr_onnxruntime/utils/infer_engine.py` SOLO usa CPU EP (inserta CUDA si use_cuda y hay provider, y DML en Windows). **CoreML está disponible y NO se usa**: en macOS todo el OCR corre en CPU. Candidato a palanca de velocidad en ocr_v2 (sin dependencias nuevas).
- Config por defecto de RapidOCR (config.yaml): det limit_side_len 736 con limit_type **min** (las imágenes GRANDES no se reducen -> det caro), use_cls true (classificador de giro en cada caja), text_score 0.5, intra_op_num_threads -1.
- Procesos: solo UN prepare_m3kang en marcha (no hay duplicados).
- Parquet local: ~/.cache/huggingface/hub/datasets--qualcomm--M3Kang/snapshots/e772e611d45679cf48f1c92618b2963a40d1d584/data/high_quality-00000-of-00004.parquet.

## Decisiones
- Todo el código nuevo en UN fichero: `laptop/ocr_v2.py` con subcomandos probe/baseline/run/revise/split (C2). Sin requirements-ocr.txt (numpy/Pillow ya están).
- Importo la lógica existente (`sys.path` + `import prepare_m3kang`) para parse_options/reading_order/is_marker: cero replicación del parser en la línea base. El REPARTO está inline en main() (no importable): lo replico VERBATIM (sha256 % 1000) y lo valido contra los train/val actuales para las claves que solapan.
- Muestra fija: primeras N filas eng del shard 0 (N=20 para el cronómetro, 60 para lo demás; --limit cambia N, nunca el orden).
- Mi caché vive SOLO en data/ocr_v2/ (.cache-<variante>.jsonl, guarda CAJAS OCR para poder re-probar posprocesados sin re-OCR). NUNCA escribo en data/.ocr_cache.jsonl.
- Corridas de OCR anunciadas aquí y con `nice -n 20` (cortas, <= 60 filas). No ejecutar dataset completo sin aprobación.
- "Dudosas" (vacía o duplicada): se marcan en los informes y en revision.md; el JSONL oficial mantiene el formato exacto (options {} o {A..E}) + ocr_version, para que predict/eval/build_index no cambien.

## Hecho
- `laptop/ocr_v2.py` (NUEVO, ~580 líneas, 1 solo fichero nuevo): subcomandos `probe` / `baseline` / `run --improve crop,scale,gray,contrast,postproc,secondpass` / `revise` / `split`. Importa `prepare_m3kang` (parser existente) y `reason.calibration.wilson_interval`. Caché propia de CAJAS en `data/ocr_v2/.cache-<variante>.jsonl` (el posprocesado se re-mide sin re-OCR). Compila OK.
- **Paso 1 (cronometraje oficial, probe --limit 20, SIN contención: el proceso largo del usuario terminó durante la sesión)**: **0,36 filas/s** (2,79 s/fila media y mediana, peor 4,87 s) -> el ~0,3 del usuario estaba bien. Diagnóstico: det 1,35 s (**49%**), rec 1,40 s (**50%**), cls 0,02 s (1%); imágenes VERTICALES estrechas (media 323 x 1653 px); det usa limit_side_len 736 con limit_type **min** -> AMPLÍA el lado corto (323<736) a 736 => entrada de det ~736 x ~3800 px. Arquitectura arm64 OK; 0 procesos duplicados al medir; backend: CPU (CoreML disponible y sin usar por rapidocr 1.4.4).
- **Paso 2 (baseline, 60 filas fijas, lógica EXISTENTE importada)**: N 60 en 185,6 s (**0,32 filas/s**); **5 opciones completas 39/60 = 65,0%** (Wilson 95%: 52,4%–75,8%); dudosas (duplicadas) entre leídas: 0/39; fallidas 21: lvl-0_2015_1, lvl-0_2015_12, lvl-0_2015_16, lvl-0_2015_3, lvl-0_2015_8, lvl-0_2016_20, lvl-0_2017_10, lvl-0_2017_11, lvl-0_2017_16, lvl-0_2017_19, lvl-0_2017_4, lvl-0_2017_5, lvl-0_2017_8, lvl-0_2018_14, lvl-0_2018_17, lvl-0_2018_3, lvl-0_2018_8, lvl-0_2018_9, lvl-0_2019_13, lvl-0_2019_14, lvl-0_2019_16. **Validación: 60/60 claves ya estaban en la caché compartida y 60/60 opciones idénticas** (mi réplica == lógica del usuario). Filas en data/ocr_v2/rows-baseline.jsonl.

## Hecho (continuación): pasos 3-5, misma muestra fija de 60 filas
- **Paso 3 (variantes, una a una)**:
  - baseline: 0,32 filas/s; 39/60 = 65,0% (Wilson 52,4–75,8); 21 fallidas.
  - **crop (a), cf0.4**: 0,83 filas/s (**2,6x**); 42/60 = 70,0% (57,5–80,1); +3 ganadas, **0 perdidas**.
  - crop+contrast (b): 0,77; 40/60 = 66,7%; **2 perdidas** -> DESCARTADA (imágenes ya casi binarias).
  - **crop+gray (b): 0,93 filas/s (la más rápida); 42/60 = 70,0%; 0 perdidas**.
  - crop+scale (b): 0,89; 41/60 = 68,3%; 1 perdida -> DESCARTADA.
  - crop+gray-cf0.3: 1,11; 33/60 = 55,0%; **9 perdidas** -> DESCARTADA (0.4 se queda).
  - crop+gray+postproc (c), cf0.4: 42/60 = 70,0%; 0 perdidas; 13 valores cambiados (mejores: ver paso 4).
  - + secondpass (d): **0 recuperadas de 18** -> no aporta, fuera de la variante final.
  - Con N=60 los intervalos se solapan: la mejora de CALIDAD no es significativa; la de VELOCIDAD sí (~3x, mismas condiciones).
- **Paso 4 (revisión visual; vi 7 imágenes yo mismo)**: ganancias verificadas correctas: 2016_20 ("10 ml..50 ml"), 2017_10 ("2..6"), 2018_14 (opciones-letras A..E). Lecturas v2 mejores: 2017_2 (D "5"->"55"), 2016_9 (E "735186 = 一"->"735186"). Fix del posprocesado: 2015_10 ("5 白"->"5"). Fallo correcto irrecuperable: 2015_3 (opciones dibujo). El posprocesado (c) añadió: glifos CJK/símbolos, tokens solo-puntuacion (<=2 chars), secuencia "ABCDE", fracciones unicode — nunca vacía el valor. Quedan 2 colas ASCII no eliminables de forma segura ("9 ooo", "5 P"). `data/ocr_v2/revision.md` (30 filas + mis veredictos) y 30 PNG en `data/ocr_v2/images/`.
- **Paso 5 (split)**: data/ocr_v2/train.jsonl (52 filas) + val.jsonl (8), mismas claves de campo + `ocr_version: "v2-crop+gray-cf0.4+postproc"`. **Reparto replicado verbatim (sha256 % 1000): 0 mismatches sobre 60 claves solapadas** contra los train/val actuales. OJO: el proceso del usuario, al terminar (20:41), REESCRIBIÓ data/train.jsonl (1466 filas) y data/val.jsonl (281) con el shard-0 eng completo — yo no los toqué (C4 cumplido; su caché .ocr_cache.jsonl también creció por su proceso).
- Formato: json.loads + set de claves exacto en las 60 filas: OK (lector mínimo, mismo contrato que copilot).

## En curso
- nada.

## Pendiente / decisiones abiertas
1. **Aprobación del usuario** para el run completo: `python3 laptop/ocr_v2.py run --improve crop,gray,postproc --limit 0` + `python3 laptop/ocr_v2.py split --variant 'crop+gray-cf0.4+postproc'` (~1747 filas eng, ~31 min estimados a 0,93 filas/s; escribiría data/ocr_v2/train.jsonl+val.jsonl COMPLETOS sin tocar los protegidos).
2. Revisión humana de las filas sin veredicto en revision.md (¿las 18 fallidas son opciones dibujo?).
3. Comparación en placa (paso 6, comandos abajo) con el MISMO --limit para v1 y v2.
4. Opcional, sin probar: montar CoreML en onnxruntime (dependencia ya instalada) para acelerar det/rec — palanca de velocidad no evaluada.

## Comandos para retomar (portátil)
- Flujo de medidas: `.venv/bin/python3 laptop/ocr_v2.py probe --limit 20` ; `baseline` ; `run --improve crop,gray,postproc` ; `revise --n 30 --variant 'crop+gray-cf0.4+postproc'` ; `split --variant 'crop+gray-cf0.4+postproc'`.
- **Paso 6 en la placa**: portátil -> `scp data/ocr_v2/train.jsonl data/ocr_v2/val.jsonl usuario@placa:~/data/ocr_v2/`; placa (servidores systemd ya encendidos) ->
  `python3 eval.py --input data/val.jsonl --limit 50 --name ocr-v1`
  `python3 eval.py --input data/ocr_v2/val.jsonl --limit 50 --name ocr-v2`
  `python3 reason/eval_summary.py results/eval-*-ocr-v1.csv results/eval-*-ocr-v2.csv`
  (mismo --limit, misma semilla de shuffle; comparar los dos números y las letras pred/truth). Con los ficheros de la MUESTRA la comparación es pequeña (8 filas de val en mi muestra): para una comparación real, aprobar el run completo (pendiente 1) y re-scp.

PROGRESO GUARDADO
