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