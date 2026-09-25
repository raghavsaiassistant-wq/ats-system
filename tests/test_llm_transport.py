#!/usr/bin/env python3
"""End-to-end transport tests against the mock provider.

Exercises the real HTTP path — auth header, request shape, response parsing,
and every failure mode — rather than stubbing the client out.

    python tests/test_llm_transport.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from ats_checker import llm_client  # noqa: E402
from ats_checker.scorer import run_full_check  # noqa: E402

PORT = 8099
BASE = f"http://127.0.0.1:{PORT}/v1"
KEY = "test-key-123"
MOCK = Path(__file__).parent / "mock_provider.py"

passed = failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


class Mock:
    def __init__(self, mode="normal"):
        self.mode = mode

    def __enter__(self):
        self.proc = subprocess.Popen(
            [sys.executable, str(MOCK), "--port", str(PORT), "--mode", self.mode],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(50):
            try:
                requests.post(f"{BASE}/chat/completions", timeout=1)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(0.1)
        return self

    def __exit__(self, *a):
        self.proc.terminate()
        self.proc.wait(timeout=5)


def call(**kw):
    return llm_client.call_json(
        "You are a connection test.", "Return JSON.",
        model="mock-model", host=BASE, api_key=KEY, provider="openai",
        timeout=15, retries=0, **kw,
    )


print("\n--- transport: success paths -------------------------------------")
for mode, label in [("normal", "clean JSON"), ("fenced", "```json fenced"),
                    ("prose", "prose around JSON"), ("reason", "reasoning_content fallback")]:
    with Mock(mode):
        parsed, err = call()
        check(f"{label} parses", parsed is not None and err is None, f"err={err}")
        if parsed:
            check(f"{label} has expected field", "semantic_score" in parsed, str(parsed)[:80])

print("\n--- transport: failure paths -------------------------------------")
with Mock("401"):
    parsed, err = call()
    check("401 reported clearly", parsed is None and err and "401" in err, f"err={err}")
with Mock("404"):
    parsed, err = call()
    check("404 mentions model name", parsed is None and err and "mock-model" in err, f"err={err}")
with Mock("429"):
    parsed, err = call()
    check("429 reported", parsed is None and err and "429" in err, f"err={err}")
with Mock("badjson"):
    parsed, err = call()
    check("non-JSON reported, not crashed", parsed is None and err and "JSON" in err, f"err={err}")

# No server at all
parsed, err = llm_client.call_json(
    "s", "u", model="m", host="http://127.0.0.1:9", api_key="k",
    provider="openai", timeout=5, retries=0,
)
check("dead endpoint fails soft", parsed is None and err and "Could not reach" in err, f"err={err}")

print("\n--- auth: wrong key ----------------------------------------------")
with Mock("normal"):
    parsed, err = llm_client.call_json(
        "s", "u", model="mock-model", host=BASE, api_key="wrong-key",
        provider="openai", timeout=15, retries=0,
    )
    check("wrong key rejected", parsed is None and err and "401" in err, f"err={err}")

print("\n--- test_connection helper ---------------------------------------")
with Mock("normal"):
    ok, msg = llm_client.test_connection(
        model="mock-model", host=BASE, api_key=KEY, provider="openai")
    check("test_connection succeeds", ok, msg)

print("\n--- full three-layer run through the real transport --------------")
with Mock("normal"):
    report = run_full_check(
        resume_path=str(Path(__file__).parent.parent / "samples" / "sample_resume.txt"),
        jd_text=(Path(__file__).parent.parent / "samples" / "sample_jd.txt").read_text(),
        model="mock-model", host=BASE, api_key=KEY, provider="openai",
    )
    check("semantic layer live", report.semantic_result.available,
          str(report.semantic_result.error))
    check("semantic score populated", report.semantic_result.semantic_score == 71,
          str(report.semantic_result.semantic_score))
    check("manager layer live", report.manager_result.available,
          str(report.manager_result.error))
    check("manager score computed", report.manager_score is not None,
          str(report.manager_score))
    check("weak bullets returned", len(report.manager_result.weak_bullets) == 1,
          str(report.manager_result.weak_bullets))
    check("ATS score uses semantic weighting", report.ats_weights["semantic"] > 0,
          str(report.ats_weights))
    check("HR expectations generated", len(report.recruiter_result.expectations) > 0)
    import json
    json.dumps(report.to_dict())  # raises if not serializable
    check("report JSON-serializable", True)

print("\n--- native Ollama protocol (/api/chat) ---------------------------")
OLLAMA_BASE = f"http://127.0.0.1:{PORT}"

def ollama_call(mode="normal", **kw):
    return llm_client.call_json(
        "You are a connection test.", "Return JSON.",
        model="gpt-oss:120b-cloud", host=OLLAMA_BASE, api_key=KEY,
        provider="ollama", timeout=15, retries=0, **kw,
    )

for mode, label in [("normal", "clean JSON"), ("fenced", "fenced"), ("prose", "prose-wrapped")]:
    with Mock(mode):
        parsed, err = ollama_call()
        check(f"ollama native: {label}", parsed is not None and err is None, f"err={err}")

with Mock("normal"):
    parsed, err = llm_client.call_json(
        "s", "u", model="m", host=OLLAMA_BASE, api_key="wrong-key",
        provider="ollama", timeout=15, retries=0)
    check("ollama native: bearer auth enforced", parsed is None and err and "401" in err, f"err={err}")

with Mock("normal"):
    report = run_full_check(
        resume_path=str(Path(__file__).parent.parent / "samples" / "sample_resume.txt"),
        jd_text=(Path(__file__).parent.parent / "samples" / "sample_jd.txt").read_text(),
        model="gpt-oss:120b-cloud", host=OLLAMA_BASE, api_key=KEY, provider="ollama",
    )
    check("ollama native: full run, semantic live", report.semantic_result.available,
          str(report.semantic_result.error))
    check("ollama native: full run, manager live", report.manager_result.available,
          str(report.manager_result.error))
    check("ollama native: all three scores present",
          report.ats_score and report.recruiter_score and report.manager_score,
          f"{report.ats_score}/{report.recruiter_score}/{report.manager_score}")

print("\n--- config resolution --------------------------------------------")
import os
def resolved(**env):
    saved = {k: os.environ.get(k) for k in
             ("ATS_LLM_PROVIDER", "ATS_LLM_BASE_URL", "ATS_LLM_MODEL", "ATS_LLM_API_KEY",
              "OLLAMA_HOST", "OLLAMA_MODEL", "OLLAMA_API_KEY")}
    for k in saved:
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in env.items() if v is not None})
    try:
        return llm_client.current_config()
    finally:
        for k in list(env):
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

c = resolved(ATS_LLM_BASE_URL="ollama-cloud", ATS_LLM_API_KEY="k")
check("preset 'ollama-cloud' -> native provider",
      c["provider"] == "ollama" and c["base_url"] == "https://ollama.com", str(c))
check("ollama-cloud picks a :cloud default model", c["model"].endswith("-cloud"), c["model"])

c = resolved(ATS_LLM_BASE_URL="https://ollama.com", ATS_LLM_API_KEY="k")
check("bare ollama.com URL + key -> native, not OpenAI", c["provider"] == "ollama", str(c))

c = resolved(ATS_LLM_BASE_URL="ollama-local")
check("preset 'ollama-local' expands", c["base_url"] == "http://localhost:11434", str(c))

c = resolved(ATS_LLM_BASE_URL="glm", ATS_LLM_API_KEY="k")
check("preset 'glm' -> openai provider",
      c["provider"] == "openai" and "z.ai" in c["base_url"], str(c))

c = resolved(OLLAMA_HOST="https://ollama.com", OLLAMA_API_KEY="k", OLLAMA_MODEL="kimi-k2:1t-cloud")
check("legacy OLLAMA_* vars still honoured",
      c["provider"] == "ollama" and c["model"] == "kimi-k2:1t-cloud" and c["api_key"] == "k", str(c))

c = resolved()
check("no config at all -> local ollama default",
      c["provider"] == "ollama" and "11434" in c["base_url"], str(c))

print("\n--- JSON extraction unit checks ----------------------------------")
cases = [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Here you go:\n{"a": 1}\nHope that helps!', {"a": 1}),
    ('{"a": "}"}', {"a": "}"}),                     # brace inside a string
    ('{"a": {"b": [1,2]}}', {"a": {"b": [1, 2]}}),  # nesting
    ("no json here", None),
]
for raw, expected in cases:
    check(f"extract_json({raw[:28]!r})", llm_client.extract_json(raw) == expected,
          f"got {llm_client.extract_json(raw)}")

print(f"\n{'='*60}\n{passed} passed, {failed} failed\n{'='*60}")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all_checks_passed():
    """pytest entry point: the checks above run at import (collection) time;
    a bare module-level sys.exit used to abort pytest's collection outright."""
    assert failed == 0, f"{failed} check(s) failed — run this file directly for details"
