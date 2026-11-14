"""A fake llama-server for testing the doubt detector end to end.

Returns a "sure" distribution (top letter B, ~58%) for problems whose text
mentions "bee", and a "doubting" one (nearly uniform, letter A) otherwise.
Deterministic per problem, so runs compare.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SURE = {"token": "B", "logprob": -0.406, "top_logprobs": [
    {"token": "B", "logprob": -0.406}, {"token": "A", "logprob": -1.2},
    {"token": "E", "logprob": -2.0}, {"token": "C", "logprob": -3.3}, {"token": "D", "logprob": -4.5}]}
DOUBT = {"token": "A", "logprob": -1.5, "top_logprobs": [
    {"token": "A", "logprob": -1.5}, {"token": "D", "logprob": -1.55}, {"token": "B", "logprob": -1.6},
    {"token": "E", "logprob": -1.65}, {"token": "C", "logprob": -1.7}]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"status": "ok"})
        if self.path == "/props":
            return self.reply(200, {"modalities": {"vision": False}})
        return self.reply(404, {"error": "not found"})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        entry = SURE if "bee" in body else DOUBT
        self.reply(200, {
            "choices": [{"message": {"role": "assistant", "content": entry["token"]},
                         "logprobs": {"content": [entry]}}],
            "timings": {"prompt_n": 42},
        })


ThreadingHTTPServer(("127.0.0.1", 8081), Handler).serve_forever()