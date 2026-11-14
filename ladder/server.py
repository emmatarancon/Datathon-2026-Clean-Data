#!/usr/bin/env python3
"""The hint ladder page: four rungs from a similar problem to the solution.

    python3 ladder/server.py --input data/val.jsonl --index data/index.db
    python3 ladder/server.py --input ladder/tests/problems.jsonl --port 8010

Then open http://127.0.0.1:8010 in a browser on the board. A separate server
from the explorer's, so explorer/ stays untouched. Standard library only,
nothing leaves the device. The label is only ever shown by rung 4, on purpose:
the ladder guides first and answers last.
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.client import get  # noqa: E402
from copilot.config import CONFIG, ROOT  # noqa: E402
from copilot.rag import Index  # noqa: E402
from ladder.ladder import public, rung  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LLM_BASE = f"http://127.0.0.1:{CONFIG['LLM_PORT']}"
HINT_URL = LLM_BASE + "/v1/chat/completions"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/val.jsonl", help="problems to practise with")
    p.add_argument("--index", default="data/index.db", help="retrieval index from build_index.py")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open it to your network")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--hint-url", default=HINT_URL, help="override the chat completions URL (for tests)")
    return p.parse_args()


def key_of(problem):
    return f"{problem['id']}_{problem['language']}"


def is_up(base):
    try:
        return get(base + "/health", timeout=2).get("status") == "ok"
    except (OSError, ValueError):
        return False


class App:
    def __init__(self, args):
        os.chdir(ROOT)
        with open(args.input) as f:
            self.problems = [json.loads(line) for line in f if line.strip()]
        self.by_key = {key_of(p): p for p in self.problems}
        self.index = Index(args.index) if os.path.exists(args.index) else None
        self.hint_url = args.hint_url

    def info(self):
        return {"count": len(self.problems),
                "languages": sorted({p["language"] for p in self.problems}),
                "levels": sorted({p["level"] for p in self.problems}),
                "index": len(self.index.rows) if self.index else 0,
                "llm": is_up(LLM_BASE),
                "time_limit": float(CONFIG["TIME_LIMIT_SECONDS"])}

    def listing(self):
        return [{"key": key_of(p), "id": p["id"], "language": p["language"],
                 "level": p["level"], "options": bool(p.get("options"))}
                for p in self.problems]

    def climb(self, key, n):
        return rung(n, self.by_key[key], index=self.index, url=self.hint_url)


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("  " + fmt % args + "\n")

        def send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            params = dict(pair.split("=", 1) for pair in query.split("&") if "=" in pair)
            if path == "/":
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    return self.send(200, f.read(), "text/html; charset=utf-8")
            if path == "/api/info":
                return self.send(200, app.info())
            if path == "/api/problems":
                return self.send(200, app.listing())
            if path == "/api/problem" and params.get("key") in app.by_key:
                shown = public(app.by_key[params["key"]])
                shown["key"] = params["key"]
                return self.send(200, shown)
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/rung":
                return self.send(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            key, n = body.get("key"), body.get("rung")
            if key not in app.by_key or not isinstance(n, int) or not 1 <= n <= 4:
                return self.send(400, {"error": "need a known key and a rung from 1 to 4"})
            return self.send(200, app.climb(key, n))

    return Handler


def main():
    args = parse_args()
    app = App(args)
    info = app.info()
    print(f"{info['count']} problems from {args.input}, {info['index']} in the retrieval index")
    print(f"Language model {'up' if info['llm'] else 'DOWN'} (rung 3 needs it); "
          f"rungs 1-2 need the embeddings server and the index")
    print(f"Open http://{args.host}:{args.port} (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(app)).serve_forever()


if __name__ == "__main__":
    main()