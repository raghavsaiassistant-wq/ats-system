#!/usr/bin/env python3
"""A mock OpenAI-compatible provider, used to test the LLM transport without
spending API credits (and to reproduce failure modes on demand).

    python tests/mock_provider.py --port 899 [--mode normal|fenced|prose|401|404|429|badjson]

Serves BOTH protocols on the same port:
  POST /v1/chat/completions   OpenAI-compatible shape (choices[0].message)
  POST /api/chat              Ollama native shape (top-level message)

Modes reproduce the things real providers actually do:
  normal   clean JSON in choices[0].message.content
  fenced   JSON wrapped in ```json fences
  prose    prose before and after the JSON object
  reason   content empty, answer in reasoning_content (GLM reasoning models)
  401/404/429   error statuses
  badjson  not JSON at all
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

MODE = "normal"
EXPECTED_KEY = "test-key-123"

SEMANTIC_PAYLOAD = {
    "semantic_score": 71,
    "strengths": ["Power BI and DAX match the JD directly", "Python automation experience"],
    "gaps": ["No Snowflake or cloud warehouse experience shown"],
    "reworded_matches": ["'automated reporting workflows' covers 'automate recurring reporting'"],
    "recommendation": "Strong tool overlap; add warehouse and stakeholder evidence.",
}

MANAGER_PAYLOAD = {
    "quantification": 35, "outcome_focus": 45, "evidence_backing": 60,
    "scope_match": 55, "domain_relevance": 78, "credibility": 70,
    "weak_bullets": [{
        "bullet": "Built automated monthly QA decks",
        "problem": "No scale or impact stated",
        "rewrite": "Built automated monthly QA decks across [X] programs, cutting prep by [Y] hours",
    }],
    "interview_risks": ["Claude API usage — be ready to explain what you actually built"],
    "manager_verdict": "Relevant stack, but bullets describe duties rather than results.",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep test output clean

    def _send(self, status: int, body: dict):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        native_ollama = self.path.endswith("/api/chat")
        if not (native_ollama or self.path.endswith("/chat/completions")):
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return

        auth = self.headers.get("Authorization", "")
        if MODE == "401" or auth != f"Bearer {EXPECTED_KEY}":
            self._send(401, {"error": {"message": "invalid api key"}})
            return
        if MODE == "404":
            self._send(404, {"error": {"message": "model not found"}})
            return
        if MODE == "429":
            self._send(429, {"error": {"message": "rate limit exceeded"}})
            return

        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        system = (req.get("messages") or [{}])[0].get("content", "")
        payload = MANAGER_PAYLOAD if "hiring manager" in system else SEMANTIC_PAYLOAD

        if MODE == "badjson":
            content = "I'm afraid I can't produce structured output."
        elif MODE == "fenced":
            content = "```json\n" + json.dumps(payload) + "\n```"
        elif MODE == "prose":
            content = ("Sure! Here is my assessment:\n\n" + json.dumps(payload)
                        + "\n\nLet me know if you'd like more detail.")
        else:
            content = json.dumps(payload)

        message = {"role": "assistant", "content": content}
        if MODE == "reason":
            message = {"role": "assistant", "content": "", "reasoning_content": content}

        if native_ollama:
            # Ollama's native response shape: message at the top level,
            # no "choices" array.
            self._send(200, {
                "model": req.get("model", "mock"),
                "created_at": "2026-01-01T00:00:00Z",
                "message": message,
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 100,
                "eval_count": 50,
            })
            return

        self._send(200, {
            "id": "mock-1",
            "object": "chat.completion",
            "model": req.get("model", "mock"),
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        })


def main():
    global MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--mode", default="normal",
                    choices=["normal", "fenced", "prose", "reason", "401", "404", "429", "badjson"])
    args = ap.parse_args()
    MODE = args.mode
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
