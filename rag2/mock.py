"""--mock: simulated model and embeddings, so tune.py/final.py run on the
laptop without the board or its servers. Mock numbers are NOT measurements:
they only prove the plumbing works. Real measurements happen only on the board.
"""
import hashlib
import re

DIM = 384  # same width as bge-small-en-v1.5 so stored code paths are identical
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _bucket(token):
    return int.from_bytes(hashlib.sha1(token.encode()).digest()[:8], "big")


def mock_embed(texts):
    """Deterministic hashed bag-of-tokens vectors, unit length."""
    out = []
    for text in texts:
        vec = [0.0] * DIM
        for token in TOKEN_RE.findall(str(text).lower()):
            vec[_bucket(token) % DIM] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        out.append([x / norm for x in vec])
    return out


def _prompt_seed(problem, examples):
    from copilot.llm import format_problem
    parts = [format_problem(problem)]
    parts += [f"{e.get('id', '')}|{format_problem(e)}" for e in examples]
    return hashlib.sha1("||".join(parts).encode()).hexdigest()[:8]


def mock_answer(problem, examples=(), timeout=60, image_b64=None, url=None):
    """(letter, timings) like answer_letter, deterministic on the prompt."""
    letter = "ABCDE"[int(_prompt_seed(problem, examples), 16) % 5]
    return letter, {}


def mock_answer_probs(problem, examples=(), timeout=60, image_b64=None, url=None):
    """(letter, timings, probs) like answer_letter_probs, all deterministic."""
    letter, timings = mock_answer(problem, examples)
    probs = {l: 0.2 for l in "ABCDE"}
    probs[letter] = 0.6
    return letter, timings, probs


def install():
    """Patch the pipeline's model calls and rag2's query embeddings."""
    import copilot.pipeline as pipeline
    pipeline.answer_letter = mock_answer
    pipeline.answer_letter_probs = mock_answer_probs
    from rag2 import data
    data._MOCK_EMBED = mock_embed
