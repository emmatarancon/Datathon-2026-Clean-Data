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