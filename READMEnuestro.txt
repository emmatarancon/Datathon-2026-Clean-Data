datathon 2026

Repte Qualcomm

Manual Github:

    git add .  (afegeix els canvis)
    git commit -m 'missatge' (missatge)
    git push (puja els canvis per els altres)

    git pull (descarregar els canvis dels altres)

https://github.com/mpous/datathon-UPC-maths-qualcomm-arduino-unoq
## Setup guide for Arduino UNO Q
https://www.youtu.be/l1gk_Qj23cY



172.17.0.1 10.198.33.193 

hugngingface
rag amb les diferens tipus de problemes per preparar les dades duvidir per com funcionen
procediments i tipus de resposta
qwen dos modelos y comparar con un benshmark
es mejor un slm



hostname -I (para ver la IP)
ssh arduino@10.198.33.193           (con el segundo numero para conectar)
 sudo poweroff (para apagar)






systemctl status math-llm               # is it running?
journalctl -u math-llm -f               # follow its log (Ctrl+C to stop)
sudo systemctl restart math-llm         # restart it, for example after changing the model
sudo systemctl stop math-llm math-embed # free the RAM for an experiment



 systemctl is-active math-llm math-embed
active
active
arduino@CleanData:~/models$ cd ~/Datathon-2026-Clean-Data
ls
-bash: cd: /home/arduino/Datathon-2026-Clean-Data: No such file or directory
bge-small-en-v1.5-q8_0.gguf  Qwen_Qwen3.5-0.8B-Q8_0.gguf  results
arduino@CleanData:~/models$ mkdir -p results
python3 predict.py --input data/sample_problems.jsonl --output results/smoke.csv 
cat results/smoke.csv
python3: can't open file '/home/arduino/models/predict.py': [Errno 2] No such file or directory
cat: results/smoke.csv: No such file or directory




2. El “detector de duda” de la IA

Pides al modelo las probabilidades de cada letra (el README menciona logprobs) y dibujas 5 barras A–E. Si una barra domina, la IA está segura. Si están todas parecidas, duda. Lo original: el estudiante ve que la IA a veces adivina, y eso enseña pensamiento crítico. Lo medible: cuando está muy segura, ¿acierta de verdad? Es una calibración, que queda muy bien en el análisis de errores.

    ============================================================
    RESUMEN DE LO IMPLEMENTADO: el "detector de duda" (doubt detector)
    ============================================================

    QUÉ HACE
    --------
    - Es la MISMA petición al modelo de siempre (misma gramática "root ::= [A-E]",
      1 token, temperature 0, seed), pero pidiendo además las probabilidades de
      cada letra: "logprobs": true, "top_logprobs": 5 (lo que sugiere el README).
      Como la gramática enmascara todo menos A-E, la respuesta trae la
      distribución completa de las 5 letras.
    - En el explorer (la web) se dibujan 5 barras A–E al pedir la respuesta del
      modelo: si una barra domina, la IA está segura; si están todas parecidas,
      duda. El estudiante VE cuándo la IA está adivinando -> pensamiento crítico.
    - En los CSV de predict.py y eval.py se guardan columnas nuevas: pA..pE,
      top_prob y verdict (sure/doubt).
    - La parte medible es la CALIBRACIÓN: una tabla que parte las respuestas por
      nivel de confianza y muestra el acierto de cada tramo. Responde a "cuando
      está muy segura, ¿acierta de verdad?", y da dos números resumen (ECE y
      Brier; 0 = perfecto). Queda muy bien en el análisis de errores.

    CÓMO SE ENCENDE Y SE APAGA (por defecto está APAGADO y no cambia nada)
    ----------------------------------------------------------------------
    - Interruptor único: DOUBT_DETECTOR=0 en config.env (así está ahora).
      Apagado, todos los ficheros de salida son idénticos a la línea base
      (mismas columnas, mismas respuestas) y la petición al servidor no cambia.
    - Encenderlo para todo (predict + eval + explorer):
          DOUBT_DETECTOR=1 en config.env   (o la variable de entorno al lanzar)
    - Encenderlo solo para un experimento, con el flag --doubt:
          python3 eval.py --doubt --limit 50 --name doubt
          python3 predict.py --input data/val.jsonl --output results/p.csv --doubt
    - Ajuste fino: DOUBT_SURE_PROB=0.5 -> una letra con probabilidad >= 0.5
      cuenta como "segura". Se puede subir/bajar para probar.

    COMANDOS ÚTILES
    ---------------
    - python3 eval.py --doubt --limit 50 --name doubt
        (al final del resumen imprime la tabla de calibración)
    - python3 calibration.py results/eval-<fecha>-doubt.csv
        (la misma tabla, a partir de cualquier CSV escrito con --doubt)
    - python3 explorer/server.py
        (las barras salten en la web al pulsar "Ask the model")

    FICHEROS QUE TOCAN (todo aditivo, nada reescrito)
    -------------------------------------------------
    - copilot/confidence.py   NUEVO: parsea los logprobs, decide sure/doubt,
                              dibuja las barras en texto y calcula la calibración
    - calibration.py          NUEVO: reimprime la tabla desde un CSV de eval
    - copilot/llm.py          función nueva answer_letter_probs()
                              (answer_letter se queda exactamente igual)
    - copilot/pipeline.py     solve(..., doubt=None): si está encendido añade
                              "probs" y "confidence" al resultado
    - predict.py / eval.py    flag --doubt y columnas extra SOLO si se usa
    - explorer/server.py,
      explorer/index.html     las 5 barras en la web
    - config.env              DOUBT_DETECTOR y DOUBT_SURE_PROB
    - .skills/m3kang-copilot/SKILL.md   documentación para el equipo

    POR QUÉ NO CAMBIA EL RESULTADO OFICIAL (predict.py)
    ---------------------------------------------------
    - La letra no cambia: es la misma generación de 1 token; solo se pide además
      el informe de logprobs. Probado: respuestas idénticas con el detector
      encendido y apagado (SAME_ANSWERS_OK).
    - Sin coste de tiempo en la placa: la generación es igual; solo la respuesta
      es un poco más grande.
    - Si el servidor no devolviera logprobs, el detector degrada con elegancia:
      celdas vacías en el CSV y sin barras en la web, y la respuesta sigue igual.
    - El CSV oficial sin --doubt conserva exactamente sus columnas:
      id,language,answer,seconds,status

    IDEA PARA EL PITCH
    ------------------
    En la demo: el alumno responde, pide la respuesta al modelo y ve las 5 barras.
    Cuando la IA duda, se abre la conversación ("¿por qué crees que duda?").
    Y con la tabla de calibración se demuestra con números si la IA "segura"
    acierta de verdad: es análisis de errores medible, no una anécdota.