"""Ask the small language model for one letter, A to E."""
from .config import CONFIG
from .client import post
from .confidence import extract_letter_probs

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# Keep the system prompt stable: llama-server caches the start of the prompt,
# so text that never changes is only read once.
SYSTEM = (
    "You are a math problem solver. You solve multiple-choice math competition "
    "problems step by step and output the letter of the correct option. "
    "Read each example carefully. Think through the logic before answering. "
    "Your final answer must be exactly one letter: A, B, C, D, or E."
)

# ------------------------------------------------------------------- structured prompt helpers

STOPWORDS = set("""the a an and or of to in on at for with by from into is are was
were be been being it its this that these those you your they their we our he she
his her as if then than so what which who whom how when where why not no nor all
any some each every one two three four five six seven eight nine ten first second
third next last other another such very more most less least only same too also
just because while during before after above below over under again here there
both few many much figure picture diagram shown shows show given following
problem question answer options option value values letter letters
el la los las un una unos unas al del lo y e o u de a en por para con sin sobre
entre desde hasta hacia tras que qué como cómo cuando dónde como dónde
cual cuales cuál cuántos cuantos cuántas cuántas es son era eran fue ser sido
siendo está estaba este esta esos esas aquel aquella su sus tu tus mi mis
nuestro nuestra si no más menos cada todo toda todos todas hay había luego
pues aunque pero porque muy ya también tanto figura muestra dibujo imagen
següent pregunta respuesta opción valor letra""".split())

WORD_RE_PATTERN = None  # lazy-init below


def _get_word_re():
    global WORD_RE_PATTERN
    if WORD_RE_PATTERN is None:
        import re
        WORD_RE_PATTERN = re.compile(r"[a-zà-öø-ÿ]{3,}")
    return WORD_RE_PATTERN


def _format_example(ex):
    """Render one solved example for the structured prompt."""
    letter = ex["label"]
    opts = "  ".join(f"{k}) {v}" for k, v in sorted(ex.get("options", {}).items()))
    return (f"Example: {ex['text']}\n"
            f"{opts}\n"
            f"Reasoning: This is a {ex.get('level', '?')} level problem. "
            f"Working through the logic gives the answer {letter}.\n"
            f"Answer: {letter}")


def _format_problem_with_reasoning(problem, has_image=False):
    """Render the target problem with a step-by-step reasoning cue."""
    lines = [problem["text"].strip()]
    opts = problem.get("options") or {}
    for letter in "ABCDE":
        if letter in opts:
            lines.append(f"{letter}) {opts[letter]}")
    if not opts:
        lines.append("(The options are in the picture.)" if has_image
                     else "(The options are in a picture that you can't see.)")
    lines.append("")
    lines.append("Think step by step about how to solve this problem.")
    lines.append("Then output your answer as: Answer: X  where X is the letter.")
    lines.append("Only output the letter after 'Answer:', nothing else.")
    return "\n".join(lines)


def build_structured_prompt(problem, examples, has_image=False):
    """Build messages for structured few-shot + reasoning prompting.
    
    This is the prompt used by the voting strategies.
    """
    sys_msg = SYSTEM
    messages = [{"role": "system", "content": sys_msg}]
    for ex in examples:
        messages.append({"role": "user", "content": _format_example(ex)})
        messages.append({"role": "assistant", "content": f"Answer: {ex['label']}"})
    messages.append({"role": "user", "content": _format_problem_with_reasoning(problem, has_image)})
    return messages


# ------------------------------------------------------------------- format functions

def format_problem(problem, sees_image=False):
    lines = [problem["text"].strip()]
    options = problem.get("options") or {}
    for letter in "ABCDE":
        if letter in options:
            lines.append(f"{letter}) {options[letter]}")
    if not options:
        lines.append("(The options are in the picture.)" if sees_image else
                     "(The options are in a picture that you can't see. Pick the most likely letter.)")
    return "\n".join(lines)


def build_messages(problem, examples=(), image_b64=None, prompt_func=None):
    """image_b64 is a PNG as base64 text. Only a server started with --mmproj can see it.
    
    If prompt_func is provided, use it to build the messages.
    Otherwise use the default few-shot format.
    """
    if prompt_func:
        return prompt_func(problem, examples)
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in examples:  # solved problems retrieved with RAG, if any
        messages.append({"role": "user", "content": format_problem(ex) + "\nAnswer:"})
        messages.append({"role": "assistant", "content": ex["label"]})
    text = format_problem(problem, sees_image=bool(image_b64)) + "\nAnswer:"
    if image_b64:
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
                    {"type": "text", "text": text}]
    else:
        content = text
    messages.append({"role": "user", "content": content})
    return messages


def build_voting_prompt(problem, examples=()):
    """Alias for build_structured_prompt - used by strategist."""
    return build_structured_prompt(problem, examples)


# ------------------------------------------------------------------- API functions

def answer_letter(problem, examples=(), timeout=60, image_b64=None, url=URL, seed=None, prompt_func=None):
    """Return (letter, timings). The grammar makes any other output impossible.
    
    If seed is provided, use it instead of CONFIG["SEED"].
    If prompt_func is provided, use it to build the messages.
    """
    effective_seed = seed if seed is not None else int(CONFIG["SEED"])
    response = post(url, {
        "messages": build_messages(problem, examples, image_b64, prompt_func),
        "grammar": "root ::= [A-E]",
        "max_tokens": 1,
        "temperature": 0,
        "seed": effective_seed,
        "cache_prompt": True,
    }, timeout=timeout)
    letter = response["choices"][0]["message"]["content"].strip()
    return letter, response.get("timings", {})


def answer_letter_probs(problem, examples=(), timeout=60, image_b64=None, url=URL, seed=None, prompt_func=None):
    """answer_letter plus the probability of every letter, for the doubt detector.

    The request is identical apart from the logprobs fields, and the grammar
    still forces one letter, so the answer itself never changes. Returns
    (letter, timings, probs), where probs is {letter: probability} or None
    when the server sent no usable logprobs.
    
    If seed is provided, use it instead of CONFIG["SEED"].
    If prompt_func is provided, use it to build the messages.
    """
    effective_seed = seed if seed is not None else int(CONFIG["SEED"])
    response = post(url, {
        "messages": build_messages(problem, examples, image_b64, prompt_func),
        "grammar": "root ::= [A-E]",
        "max_tokens": 1,
        "temperature": 0,
        "seed": effective_seed,
        "cache_prompt": True,
        "logprobs": True,
        "top_logprobs": 5,
    }, timeout=timeout)
    letter = response["choices"][0]["message"]["content"].strip()
    return letter, response.get("timings", {}), extract_letter_probs(response)