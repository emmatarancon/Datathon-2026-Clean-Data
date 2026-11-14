"""Probe: which llama.cpp configuration returns the probability of every letter?

The README suggests asking the server for the log probabilities
(`"logprobs": true, "top_logprobs": 5`). Whether they show all five letters
depends on the grammar, the endpoint, and the llama.cpp version, so this
script asks the same sample problem three ways and prints the raw answers:

  (a) grammar `root ::= [A-E]`, max_tokens 1, logprobs   (what the pipeline sends)
  (b) no grammar,      max_tokens 1, logprobs
  (c) the llama.cpp native /completion endpoint          (informative only:
       the existing client uses the OpenAI-compatible /v1/chat/completions)

Run it on the board, where the model server lives:
    python3 reason/probe_logprobs.py
    python3 reason/probe_logprobs.py --base-url http://127.0.0.1:8081 --input data/sample_problems.jsonl
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from copilot.client import post                     # noqa: E402
from copilot.config import CONFIG                   # noqa: E402
from copilot.llm import URL, build_messages         # noqa: E402
from reason import calibration                      # noqa: E402


def setup_report(response, note=""):
    """One line with what this response is useful for."""
    letter = response["choices"][0]["message"]["content"].strip()
    grouped = calibration.group_letter_logprobs(
        response["choices"][0].get("logprobs", {}).get("content", [{}])[0].get("top_logprobs"))
    probs = calibration.renormalize(grouped)
    seen = ", ".join(f"{l}:{probs[l]:.3f}" for l in sorted(probs)) or "(no letters)"
    complete = all(l in probs for l in "ABCDE")
    print(f"    sampled letter: {letter!r} | letters in top_logprobs: {seen} | "
          f"all five: {'YES' if complete else 'no'}{note}")
    return complete


def chat_completion(base_url, messages, grammar):
    payload = {"messages": messages, "max_tokens": 1, "temperature": 0,
               "seed": int(CONFIG["SEED"]), "cache_prompt": True,
               "logprobs": True, "top_logprobs": 5}
    if grammar:
        payload["grammar"] = grammar
    try:
        return post(base_url + "/v1/chat/completions", payload, timeout=120), None
    except (OSError, ValueError) as error:  # timeout, refused, malformed JSON
        return None, error


def native_completion(base_url, messages):
    """The native llama.cpp /completion endpoint, for completeness only."""
    text = "\n".join(str(m.get("content", "")) for m in messages if m["role"] != "system")
    prompt = "You solve multiple-choice math competition problems for school students.\n" + text
    payload = {"prompt": prompt, "n_predict": 1, "temperature": 0,
               "seed": int(CONFIG["SEED"]), "grammar": calibration.GRAMMAR,
               # every spelling the server might accept, so one raw answer settles it
               "n_probs": 20, "logprobs": True, "top_logprobs": 5}
    try:
        return post(base_url + "/completion", payload, timeout=120), None
    except (OSError, ValueError) as error:
        return None, error


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=URL.rsplit("/", 1)[0],
                        help="llama-server base URL (default: the pipeline's)")
    parser.add_argument("--input", default="data/sample_problems.jsonl")
    parser.add_argument("--index", type=int, default=0, help="which problem of the input to use")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    with open(args.input) as f:
        problems = [json.loads(line) for line in f if line.strip() and json.loads(line).get("options")]
    if args.index >= len(problems):
        sys.exit(f"only {len(problems)} problems with options in {args.input}")
    problem = problems[args.index]
    messages = build_messages(problem, examples=(), image_b64=None)
    print(f"Problem {problem['id']} ({problem.get('language', '')}).\n")

    verdicts = []
    print("(a) grammar + logprobs on /v1/chat/completions (the pipeline's request)")
    response, error = chat_completion(args.base_url, messages, calibration.GRAMMAR)
    if response:
        print("    raw response:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        verdicts.append(("grammar", setup_report(response)))
    else:
        print(f"    FAILED: {error}")

    print("\n(b) no grammar, logprobs on /v1/chat/completions")
    response, error = chat_completion(args.base_url, messages, None)
    if response:
        print("    raw response:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        verdicts.append(("nogrammar", setup_report(response)))
    else:
        print(f"    FAILED: {error}")

    print("\n(c) native /completion endpoint (the existing client does NOT use it)")
    response, error = native_completion(args.base_url, messages)
    if response:
        print("    raw response:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        entries = response.get("probs") or response.get("completion_probabilities") or []
        probs = calibration.renormalize(calibration.group_letter_logprobs(entries))
        seen = ", ".join(f"{l}:{probs[l]:.3f}" for l in sorted(probs)) or "(no letters)"
        print(f"    content: {str(response.get('content', '')).strip()!r} | letters: {seen}")
        verdicts.append(("native", all(l in probs for l in "ABCDE")))
    else:
        print(f"    FAILED: {error}")

    print("\n=== VERDICT ===")
    for name, complete in verdicts:
        print(f"    {name:11} returns all five letters: {'YES' if complete else 'no'}")
    good = [name for name, complete in verdicts if complete]
    if good:
        print(f"Use {' or '.join(good)} in letter_logprobs(); the raw response above shows the shape.")
    else:
        print("No configuration returned all five letters; letter_logprobs() renormalizes over "
              "whichever letters come, and the hook only corrects those.")


if __name__ == "__main__":
    main()
