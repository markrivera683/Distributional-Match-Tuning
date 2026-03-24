#!/usr/bin/env python3
"""Minimal mock teacher server for smoke-testing the remote teacher path.

Implements the /v1/completions endpoint with OpenAI-compatible responses.
Returns deterministic pseudo-random text so tests are reproducible.

Usage:
    python scripts/mock_teacher_server.py --port 8192
"""

import argparse
import json
import random
import string
import time
from http.server import HTTPServer, BaseHTTPRequestHandler


VOCAB = list(string.ascii_lowercase + " " * 5)


def _generate_text(n_tokens: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    words = []
    for _ in range(n_tokens):
        word_len = rng.randint(1, 8)
        word = "".join(rng.choices(VOCAB, k=word_len))
        words.append(word.strip() or "x")
    return " ".join(words)


class CompletionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path not in ("/v1/completions", "/completions"):
            self.send_error(404, f"Unknown endpoint: {self.path}")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        prompt = body.get("prompt", "")
        n = int(body.get("n", 1))
        max_tokens = int(body.get("max_tokens", 64))
        model = body.get("model", "mock-teacher")

        base_seed = hash(prompt) & 0xFFFFFFFF
        choices = []
        for i in range(n):
            text = _generate_text(max_tokens, seed=base_seed + i)
            choices.append(
                {
                    "index": i,
                    "text": text,
                    "finish_reason": "length",
                }
            )

        response = {
            "id": f"mock-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": max_tokens * n,
                "total_tokens": len(prompt.split()) + max_tokens * n,
            },
        }

        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"[MockTeacher] {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="Mock teacher completion server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8192)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), CompletionHandler)
    print(f"[MockTeacher] Listening on {args.host}:{args.port}")
    print(f"[MockTeacher] Endpoint: http://localhost:{args.port}/v1/completions")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MockTeacher] Shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
