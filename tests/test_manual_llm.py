#!/usr/bin/env python3
"""Copy-paste LLM mode tests (provider "manual", Phase 6) — offline.

Covers: prompt ids, the pending list and the answer cache, the bundle (one
paste, shared instructions written once), parsing pasted replies, the
run-until-answered driver (pasted at a terminal, or --answers FILE), the
llm_client routing (no network, no key), the web server's 202 / paste /
re-send flow (JSON-only, no double logging), the eval and score commands in
copy-paste mode, and that AI_GUIDE.md only names commands and flags that exist.

    python tests/test_manual_llm.py
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TD = tempfile.TemporaryDirectory()
TMP = Path(TD.name)
_OLD_HOME = os.environ.get("ATS_HOME")
os.environ["ATS_HOME"] = str(TMP / "home")

from ats_checker import llm_client, manual_llm  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from mock_provider import MANAGER_PAYLOAD, SEMANTIC_PAYLOAD  # noqa: E402

passed = failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def answer_bundle(bundle: str) -> dict:
    """Play the chat AI: answer every task in a bundle with canned payloads."""
    instr = dict(re.findall(r"## INSTRUCTIONS (I\d+)\n\n(.*?)\n-{20,}", bundle, re.S))
    tasks = re.findall(r"## TASK ID: ([0-9a-f]{12})  \(follow INSTRUCTIONS (I\d+)\)", bundle)
    out = {}
    for tid, i in tasks:
        system = instr[i]
        if "hiring manager" in system:
            out[tid] = MANAGER_PAYLOAD
        elif "extract the hiring requirements" in system:
            out[tid] = {"title": {"value": "data analyst", "quote": "Data Analyst"}, "skills": []}
        else:
            out[tid] = SEMANTIC_PAYLOAD
    return out


# --------------------------------------------------------------- core
print("\n== ids, pending, cache ==")
a, b = manual_llm.prompt_id("sys", "user"), manual_llm.prompt_id("sys", "user ")
check("prompt id is stable and 12 hex chars", a == manual_llm.prompt_id("sys", "user")
      and re.fullmatch(r"[0-9a-f]{12}", a) is not None)
check("any change to the prompt changes the id", a != b and a != manual_llm.prompt_id("sys2", "user"))

tok = manual_llm.new_session()
try:
    ans, err = manual_llm.call_manual("sys", "user")
    check("unanswered -> pending, not an exception", ans is None and err.startswith(manual_llm.PENDING_PREFIX))
    manual_llm.call_manual("sys", "user")
    check("asking twice queues it once", [p.id for p in manual_llm.pending()] == [a])
    manual_llm.store_answer(a, {"ok": True})
    ans, err = manual_llm.call_manual("sys", "user")
    check("stored answer comes back like an API reply", ans == {"ok": True} and err is None)
finally:
    manual_llm.end_session(tok)
check("the answer is cached on disk", (manual_llm.cache_dir() / f"{a}.json").exists())

orig_post = llm_client.requests.post
llm_client.requests.post = lambda *x, **k: (_ for _ in ()).throw(AssertionError("network used"))
try:
    tok = manual_llm.new_session()
    try:
        ans, err = llm_client.call_json("sys", "user", provider="manual", api_key="sk-should-not-matter")
    finally:
        manual_llm.end_session(tok)
    check("call_json(provider=manual) never touches the network", ans == {"ok": True}, str(err))
finally:
    llm_client.requests.post = orig_post
os.environ["ATS_LLM_PROVIDER"] = "manual"
cfg = llm_client.current_config()
del os.environ["ATS_LLM_PROVIDER"]
check("ATS_LLM_PROVIDER=manual config has no key", cfg["provider"] == "manual" and cfg["api_key"] == "")
ok, msg = llm_client.test_connection(provider="manual")
check("test-llm in manual mode explains itself", ok and "copy-paste" in msg.lower(), msg)

# --------------------------------------------------------------- bundle + parsing
print("\n== bundle and pasted replies ==")
prompts = [manual_llm.PendingPrompt(manual_llm.prompt_id("SYS-A", f"jd {i}"), "SYS-A", f"jd {i}")
           for i in range(3)] + [manual_llm.PendingPrompt(manual_llm.prompt_id("SYS-B", "x"), "SYS-B", "x")]
bundle = manual_llm.build_bundle(prompts)
check("every task id is in the bundle", all(f"TASK ID: {p.id}" in bundle for p in prompts))
check("shared instructions written once", bundle.count("SYS-A") == 1 and bundle.count("SYS-B") == 1)
check("the bundle says how to reply", "ONE JSON object" in bundle and "claude.ai" in bundle)
ids = [p.id for p in prompts]
reply = "Sure! Here:\n```json\n" + json.dumps({i: {"n": k} for k, i in enumerate(ids)}) + "\n```\nDone."
got, problems = manual_llm.parse_answers(reply, ids)
check("fenced reply with prose parses", got == {i: {"n": k} for k, i in enumerate(ids)} and not problems,
      str(problems))
got, problems = manual_llm.parse_answers(json.dumps({ids[0]: {"x": 1}}), ids)
check("missing answers are reported", list(got) == [ids[0]] and "No answer" in problems[0])
got, problems = manual_llm.parse_answers(json.dumps({"abcdefabcdef": {"x": 1}, ids[0]: {"y": 2}}), ids)
check("unknown ids are ignored, not stored", "abcdefabcdef" not in got and "unknown" in problems[0])
got, _ = manual_llm.parse_answers('{"rewrites": []}', [ids[0]])
check("a bare answer to the only pending prompt is accepted", got == {ids[0]: {"rewrites": []}})
got, problems = manual_llm.parse_answers("no json here", ids)
check("non-JSON reply -> clear problem", got == {} and "JSON" in problems[0])

# --------------------------------------------------------------- driver
print("\n== run until answered ==")
calls = []


def two_step():
    """Like a real run: the second prompt only exists once the first is answered."""
    first, _ = llm_client.call_json("STEP", "one", provider="manual")
    calls.append("run")
    if first is None:
        return None
    second, _ = llm_client.call_json("STEP", f"two after {first['v']}", provider="manual")
    return second


out = io.StringIO()
first_id = manual_llm.prompt_id("STEP", "one")
second_id = manual_llm.prompt_id("STEP", "two after 1")
paste = io.StringIO(json.dumps({first_id: {"v": 1}}) + "\nEND\n" +
                    json.dumps({second_id: {"v": 2}}) + "\nEND\n")
result, todo = manual_llm.run_with_answers(two_step, interactive=True, stream=paste, out=out,
                                           bundle_path=TMP / "bundle.md")
check("interactive: pastes until nothing is pending", result == {"v": 2} and todo == [], out.getvalue()[-300:])
check("ran once per paste plus the final run", len(calls) == 3, str(calls))
calls.clear()
result, todo = manual_llm.run_with_answers(two_step, interactive=False, out=io.StringIO(),
                                           bundle_path=TMP / "bundle.md")
check("a re-run is answered from the cache", result == {"v": 2} and todo == [] and len(calls) == 1)

fresh = lambda: llm_client.call_json("NEW", "never answered", provider="manual")[0]  # noqa: E731
result, todo = manual_llm.run_with_answers(fresh, interactive=False, out=io.StringIO(),
                                           bundle_path=TMP / "b2.md")
check("non-interactive: returns what's pending and writes the bundle",
      result is None and len(todo) == 1 and (TMP / "b2.md").exists())
ans_file = TMP / "reply.txt"
ans_file.write_text("```json\n" + json.dumps({todo[0].id: {"done": True}}) + "\n```", encoding="utf-8")
result, todo = manual_llm.run_with_answers(fresh, answers_file=str(ans_file), interactive=False,
                                           out=io.StringIO(), bundle_path=TMP / "b2.md")
check("--answers FILE finishes it", result == {"done": True} and todo == [])

# --------------------------------------------------------------- server
print("\n== web server: 202 -> paste -> re-send ==")
import server  # noqa: E402
from ats_checker import applog  # noqa: E402

server.DB_PATH = str(TMP / "apps.db")
server.PROVIDER_OVERRIDE = "manual"
client = server.app.test_client()
resume = (ROOT / "samples" / "sample_resume.txt").read_text(encoding="utf-8")
jd = (ROOT / "samples" / "sample_jd.txt").read_text(encoding="utf-8")
req = {"resume_text": resume, "jd_text": jd + "\nUnique line for this test.", "log": True,
       "company": "Acme"}
r = client.post("/score", json=req)
mp = (r.get_json() or {}).get("manual_pending") or {}
check("/score in copy-paste mode -> 202 with a bundle", r.status_code == 202 and mp.get("bundle")
      and len(mp.get("ids", [])) == 2, str(r.get_json())[:200])
check("no API key anywhere in the bundle", "sk-" not in mp.get("bundle", ""))
check("nothing logged on the incomplete pass", applog.list_applications(db_path=server.DB_PATH) == [])
r = client.post("/manual/answers", data="{}", content_type="text/plain")
check("/manual/answers is JSON-only", r.status_code == 415)
r = client.post("/manual/answers", json={"reply": "sorry, no", "ids": mp.get("ids", [])})
check("a reply with no JSON -> 400", r.status_code == 400 and "JSON" in r.get_json()["error"])
reply = json.dumps(answer_bundle(mp.get("bundle", "")))
r = client.post("/manual/answers", json={"reply": reply, "ids": mp.get("ids", [])})
check("a good reply is stored", r.status_code == 200 and r.get_json()["stored"] == 2, str(r.get_json()))
r = client.post("/score", json=req)
body = r.get_json() or {}
check("re-sent request completes with both LLM layers",
      r.status_code == 200 and body["scores"]["manager_evidence_strength_pct"] is not None
      and body["ats_layer"]["semantic"]["available"], str(body)[:300])
check("...and logs exactly once", len(applog.list_applications(db_path=server.DB_PATH)) == 1)
check("setup status reports copy-paste mode", client.get("/setup/status").get_json()["llm"]["provider"] == "manual")
page = client.get("/").get_data(as_text=True)
check("UI has the Copy prompt / Paste answer step",
      "function manualStep" in page and "Copy prompt" in page and 'fetch("/manual/answers"' in page)
server.PROVIDER_OVERRIDE = None

# --------------------------------------------------------------- CLI
print("\n== CLI: score and eval in copy-paste mode ==")


def cli(*argv, cwd=TMP):
    env = {**os.environ, "ATS_LLM_PROVIDER": "ollama", "ATS_LLM_BASE_URL": "ollama-local"}
    return subprocess.run([sys.executable, str(ROOT / "cli.py"), *argv], capture_output=True, text=True,
                          cwd=str(cwd), env=env, encoding="utf-8", errors="replace",
                          stdin=subprocess.DEVNULL)


jd_file = TMP / "jd.txt"
jd_file.write_text(jd + "\nAnother unique line.", encoding="utf-8")
args = ["score", "--resume", str(ROOT / "samples" / "sample_resume.txt"), "--jd", str(jd_file),
        "--provider", "manual", "--json"]
p = cli(*args)
check("score: unanswered -> exit 4, nothing half-done on stdout", p.returncode == 4 and p.stdout == "",
      p.stdout[:200] + p.stderr[-200:])
(TMP / "ans.json").write_text(json.dumps(answer_bundle((TMP / "llm_prompts.md").read_text(encoding="utf-8"))),
                              encoding="utf-8")
p = cli(*args, "--answers", str(TMP / "ans.json"))
d = json.loads(p.stdout) if p.returncode == 0 else {}
check("score with --answers: full three-layer JSON", d.get("scores", {}).get("manager_evidence_strength_pct")
      is not None, p.stderr[-300:])

corpus = TMP / "corpus"
corpus.mkdir()
for name in ("01_ba_fintech_pune.yaml", "02_data_analyst_blr.yaml"):     # tuning set, never holdout
    (corpus / name).write_text((ROOT / "tests" / "jd_corpus" / name).read_text(encoding="utf-8"),
                               encoding="utf-8")
p = cli("eval", "--extractor", "hybrid", "--corpus", str(corpus), "--provider", "manual", "--json")
bundle = (TMP / "llm_prompts.md").read_text(encoding="utf-8")
check("eval: ONE bundle with every JD's prompt", p.returncode == 4 and bundle.count("## TASK ID") == 2,
      p.stderr[-300:])
(TMP / "ans2.json").write_text(json.dumps(answer_bundle(bundle)), encoding="utf-8")
p = cli("eval", "--extractor", "hybrid", "--corpus", str(corpus), "--provider", "manual", "--json",
        "--answers", str(TMP / "ans2.json"))
check("eval with --answers produces metrics", p.returncode == 0 and "title.accuracy" in p.stdout,
      p.stderr[-300:])

# --------------------------------------------------------------- AI_GUIDE
print("\n== AI_GUIDE.md names only real commands and flags ==")
guide = (ROOT / "AI_GUIDE.md").read_text(encoding="utf-8")
from cli import build_parser  # noqa: E402

parser = build_parser()
subs = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction").choices
mentioned = 0
for line in guide.splitlines():
    m = re.search(r"python cli\.py ([a-z-]+)(.*)", line)
    if not m:
        continue
    mentioned += 1
    cmd, rest = m.group(1), m.group(2).split("#")[0]
    check(f"`cli.py {cmd}` exists", cmd in subs, cmd)
    if cmd in subs:
        opts = set(subs[cmd]._option_string_actions)
        for flag in re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", rest):
            check(f"`cli.py {cmd} {flag}` exists", flag in opts, flag)
check("the guide mentions the main commands", mentioned >= 8 and "cli.py run" in guide and "cli.py optimize" in guide)
server_src = (ROOT / "server.py").read_text(encoding="utf-8")
for flag in set(re.findall(r"python server\.py (--[a-z-]+)", guide)):
    check(f"`server.py {flag}` exists", f'"{flag}"' in server_src, flag)
readme = (ROOT / "README.md").read_text(encoding="utf-8")
check("README points chat-AI users at AI_GUIDE.md", "read AI_GUIDE.md" in readme)
check("guide states the truth rules", "Never add a skill" in guide and "No keyword stuffing" in guide)

TD.cleanup()
if _OLD_HOME is None:
    os.environ.pop("ATS_HOME", None)
else:
    os.environ["ATS_HOME"] = _OLD_HOME
print(f"\n{'=' * 60}\nManual-LLM tests: {passed} passed, {failed} failed")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all_checks_passed():
    """pytest entry point: the checks above run at import (collection) time."""
    assert failed == 0, f"{failed} check(s) failed — run this file directly for details"
