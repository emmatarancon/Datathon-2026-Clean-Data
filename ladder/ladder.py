"""The hint ladder: guide with solved problems first, the model last.

Four rungs, from cheapest to most expensive (README, "The hint ladder"):

  1. a similar SOLVED problem, one level easier, straight from the embeddings
  2. the topic: the words the nearest neighbours share (no model at all)
  3. a ONE-sentence hint from the model, checked in code so it can't leak the
     answer; if it leaks or the model fails, it falls back to rung 2
  4. the solution, only when the student asks for it

Rungs 1 and 2 cost well under a second. Only rung 3 uses the language model,
and one sentence at max_tokens 45 is short to write even at 6 tokens per
second. Every helper here takes the problems as plain dicts, so the tests run
offline with a fake index and a fake server: no M3Kang data, no network.
"""
import re
import time
import urllib.error

from copilot.client import post
from copilot.config import CONFIG
from copilot.llm import URL, format_problem

# Words that never name a topic. The neighbours' own text does the real work;
# this list just removes the grammar and the boilerplate of problem statements
# in the three languages the team practises in (English, Spanish, Catalan).
STOPWORDS = set("""the a an and or of to in on at for with by from into is are was
were be been being it its this that these those you your they their we our he she
his her as if then than so what which who whom how when where why not no nor all
any some each every one two three four five six seven eight nine ten first second
third next last other another such very more most less least only same too also
just because while during before after above below over under again here there
both few many much figure picture diagram shown shows show given following
problem question answer options option value values letter letters
el la los las un una unos unas al del lo y e o u de a en por para con sin sobre
entre desde hasta hacia tras que qué como cómo cuando cuándo donde dónde
cual cuales cuál cuáles cuanto cuánto cuantos cuántos cuanta cuántas cuántas es
son era eran fue ser sido siendo está están estaba este esta estos estas ese esa
esos esas aquel aquella su sus tu tus mi mis nuestro nuestra si sí no más menos
cada todo toda todos todas hay había haber luego pues aunque pero porque muy ya
también tambien tan tanto figura figuras muestra muestran dibujo imagen siguiente
pregunta respuesta opción opcions opción valor valores letra letras primera
el la els les un una uns unes del d l a en per amb sense des fins després
llavors però perquè perque què com quant quanta quants quantes on quan qui quin
quina quines quins es són era eren ser sent estat aquest aquesta aquests
aquestes això allò seu seva seus seves meu meva teu teva nostra nostres hi ha
havia haver encara exemple exemples mostra mostren dibuix imatge següent seguent
molt molts moltes ja també tota totes""".split())

WORD_RE = re.compile(r"[a-zà-öø-ÿ]{3,}")

# Stable system prompt for rung 3: cached by the server like the copilot's, and
# strict about not giving the answer away (the code checks it anyway).
HINT_SYSTEM = ("You help a student start a math competition problem. "
               "Reply with exactly one short sentence (25 words max) about how to "
               "begin or what to try first. Never give or imply the final answer, "
               "and never mention the option letters A to E.")


def public(problem):
    """The problem as the student may see it: every field except the label."""
    return {k: v for k, v in problem.items() if k != "label"}


def neighbourhood(problem, index, k=8):
    """The nearest solved problems, plus whether an easier one exists.

    Returns (hits, relation). hits is the list of (score, problem) pairs the
    index returned (best first); when any of them is at a lower level, hits is
    narrowed to just the easier ones and relation is 'easier', otherwise it is
    'same'. Without an index or without neighbours: ([], None). exclude_id is
    always the problem's own id, so its translations can't come back.
    """
    if index is None:
        return [], None
    level = problem.get("level", 0)
    hits = index.search(problem, k=k, exclude_id=problem.get("id"))
    easier = [hit for hit in hits if hit[1].get("level", level) < level]
    if easier:
        return easier, "easier"
    return hits, ("same" if hits else None)


def topic_of(hits, limit=3):
    """The words the neighbours share, most shared first: the topic.

    Counted as document frequency (in how many of the neighbours the word
    appears, out of at most 5), so one repetitive problem can't dominate.
    Returns [(word, neighbours_with_it, neighbours_looked_at), ...]. Pure code,
    no model: this is the rung-2 answer and the rung-3 fallback.
    """
    total = min(len(hits), 5)
    counts = {}
    for _score, neighbour in hits[:5]:
        words = {word for word in WORD_RE.findall(neighbour.get("text", "").lower())
                 if word not in STOPWORDS}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(word, n, total) for word, n in ranked[:limit]]


def first_sentence(text):
    """The first sentence of the model's text, without quotes or line breaks."""
    text = " ".join(str(text).replace("\n", " ").split()).strip("\"' ")
    match = re.search(r"^(.+?[.!?…])", text)
    return (match.group(1) if match else text).strip()


def leaks(hint, problem):
    """Does the hint give the answer away?

    Strict, because in M3Kang the option value IS the answer: any option value
    appearing in the hint is a leak even inside a sentence, and so is the
    correct letter as a standalone word (the letters are arbitrary, so naming
    the right one is luck, not guidance). An empty hint is unusable too.
    """
    if not hint or not str(hint).strip():
        return True
    text = " ".join(str(hint).lower().split())
    for value in (problem.get("options") or {}).values():
        value = " ".join(str(value).lower().split())
        if value and value in text:
            return True
    label = str(problem.get("label", "")).lower()
    if label and re.search(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])", text):
        return True
    return False


def hint_sentence(problem, url=URL, timeout=25.0):
    """Rung 3's model call: one sentence about how to start.

    No grammar here — a hint is free text, so this is the one request that can
    ramble, and max_tokens keeps it short (45 tokens is about 8 s on the board).
    Returns (hint, meta); meta["error"] says why there is no hint.
    """
    payload = {
        "messages": [
            {"role": "system", "content": HINT_SYSTEM},
            {"role": "user", "content": format_problem(problem) + "\nHint:"},
        ],
        "max_tokens": 45,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }
    try:
        response = post(url, payload, timeout=timeout)
        text = response["choices"][0]["message"]["content"]
    except (OSError, urllib.error.URLError, KeyError, IndexError, ValueError) as error:
        return "", {"error": f"model server: {error}"}
    hint = first_sentence(text)
    if not hint:
        return "", {"error": "the model returned an empty hint"}
    return hint, {}


def fallback_hint(problem, index, reason=""):
    """Rung 3 without the model: what the similar solved problems share."""
    out = {"source": "fallback"}
    if reason:
        out["reason"] = reason
    hits, _relation = neighbourhood(problem, index)
    topic = topic_of(hits) if hits else []
    if topic:
        words = ", ".join(word for word, _n, _total in topic)
        out["hint"] = (f"The similar solved problems share: {words}. "
                       "Start by listing what changes and what stays the same.")
    else:
        out["hint"] = "Read the problem twice and write down every number and what it counts."
        out["source"] = "generic"
    return out


def rung(n, problem, index=None, url=URL, hint_timeout=25.0):
    """One rung of the ladder, ready for a web page: {"rung", "seconds", ...}.

    Rungs 1 to 3 never carry the label. Rung 4 is the solution and it does,
    which is why the server only serves it on an explicit request. An "error"
    key means the rung can't be served right now (no index, server down).
    """
    start = time.perf_counter()
    out = {"rung": n}
    try:
        if n == 1:
            hits, relation = neighbourhood(problem, index)
            if not hits:
                out["error"] = "no retrieval index or no neighbours yet"
            else:
                score, neighbour = hits[0]
                out["score"] = round(score, 3)
                out["relation"] = relation
                out["neighbour"] = public(neighbour)
        elif n == 2:
            hits, relation = neighbourhood(problem, index)
            if not hits:
                out["error"] = "no retrieval index or no neighbours yet"
            else:
                out["topic"] = topic_of(hits)
                out["relation"] = relation
        elif n == 3:
            hint, meta = hint_sentence(problem, url=url, timeout=hint_timeout)
            if meta.get("error"):
                out.update(fallback_hint(problem, index, reason=meta["error"]))
            elif leaks(hint, problem):
                out.update(fallback_hint(problem, index,
                                         reason="the model's hint leaked the answer"))
            else:
                out["hint"] = hint
                out["source"] = "model"
        elif n == 4:
            out["answer"] = problem.get("label", "")
            hits, relation = neighbourhood(problem, index)
            if hits:
                score, neighbour = hits[0]
                out["relation"] = relation
                worked = dict(neighbour)  # the worked example may carry its label
                worked["score"] = round(score, 3)
                out["worked"] = worked
        else:
            out["error"] = "the rungs are 1 to 4"
    except (OSError, urllib.error.URLError, KeyError, ValueError) as error:
        out["error"] = f"model server: {error}"
    out["seconds"] = round(time.perf_counter() - start, 2)
    return out