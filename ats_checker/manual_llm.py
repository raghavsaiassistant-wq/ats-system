"""Copy-paste LLM mode (provider "manual"): no API key, any chat AI.

Most people have Claude or ChatGPT in a browser, not an API key. In manual
mode every LLM call the run would make is written into ONE prompt bundle;
the user pastes it into a chat AI, pastes the JSON reply back (or saves it
and passes --answers FILE), and the run continues.

How it fits the existing code: llm_client.call_json() hands manual calls to
call_manual() here. Each prompt is identified by a short hash of its exact
system prompt + user content:

  - answered before (cache, or the answers loaded for this run) -> the
    stored JSON comes back exactly as an API reply would, and goes through
    every existing check (quote verification, sanity ranges, rewrite
    verification). A pasted answer is trusted no more than a model's.
  - not answered yet -> the prompt joins the PENDING list and the call
    returns (None, "manual: answer pending ..."). Every layer already
    treats an LLM error as "LLM unavailable" and degrades instead of
    failing, so one pass collects every prompt the run needs.

run_with_answers() drives the loop: run, bundle what's pending, get the
answers (pasted or from a file), store them, run again — until nothing is
pending. Later prompts can depend on earlier answers (round 2 rewordings
depend on round 1), which is why it may take more than one paste.

Answers are cached under $ATS_HOME/cache/manual/<id>.json, so re-running
the same command needs no paste at all.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROVIDER = "manual"
ID_LEN = 12
PENDING_PREFIX = "manual: answer pending"
CHAT_LINKS = ("https://claude.ai/new", "https://chatgpt.com/")


@dataclass
class PendingPrompt:
    id: str
    system: str
    user: str


@dataclass
class _Session:
    pending: dict[str, PendingPrompt] = field(default_factory=dict)   # id -> prompt, in call order
    answers: dict[str, dict] = field(default_factory=dict)             # answers supplied for this run


# One session per thread / request (the web server handles requests in threads).
_SESSION: contextvars.ContextVar[_Session | None] = contextvars.ContextVar("manual_llm_session",
                                                                          default=None)
_DEFAULT = _Session()


def _session() -> _Session:
    return _SESSION.get() or _DEFAULT


def new_session(answers: dict[str, dict] | None = None) -> contextvars.Token:
    """Start a fresh pending list (and optional answers) for this thread."""
    return _SESSION.set(_Session(answers=dict(answers or {})))


def end_session(token: contextvars.Token) -> None:
    _SESSION.reset(token)


def cache_dir() -> Path:
    return Path(os.environ.get("ATS_HOME") or Path.home() / ".ats-system") / "cache" / "manual"


def prompt_id(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n\x00\n{user}".encode("utf-8")).hexdigest()[:ID_LEN]


def _cache_read(pid: str) -> dict | None:
    try:
        data = json.loads((cache_dir() / f"{pid}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def store_answer(pid: str, answer: dict) -> None:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{pid}.json.tmp"
    tmp.write_text(json.dumps(answer, ensure_ascii=False), encoding="utf-8")
    tmp.replace(d / f"{pid}.json")


def call_manual(system: str, user: str) -> tuple[dict | None, str | None]:
    """The manual-provider half of llm_client.call_json."""
    pid = prompt_id(system, user)
    sess = _session()
    answer = sess.answers.get(pid)
    if answer is None:
        answer = _cache_read(pid)
    if answer is not None:
        return answer, None
    sess.pending.setdefault(pid, PendingPrompt(pid, system, user))
    return None, f"{PENDING_PREFIX} (prompt {pid}) — paste its answer to continue"


def pending() -> list[PendingPrompt]:
    return list(_session().pending.values())


def clear_pending() -> None:
    _session().pending.clear()


# ------------------------------------------------------------- the bundle

BUNDLE_HEADER = """# Prompts for a chat AI (ATS checker, copy-paste mode)

Paste this whole message into Claude (https://claude.ai) or ChatGPT
(https://chatgpt.com).

INSTRUCTIONS FOR THE AI: below are {n} separate tasks. Each task has an ID,
names the INSTRUCTIONS block it follows, and has an INPUT. Do each task
exactly as its instructions say, as if it were the only thing you were asked.
Quote job-description lines EXACTLY when the instructions ask for quotes, and
never add facts that aren't in the input.

Reply with ONE JSON object and nothing else, mapping each task ID to that
task's JSON answer:

{{"{first_id}": {{...answer to that task...}}{more}}}
"""

RULE = "\n---------------------------------------------------------------\n"


def build_bundle(prompts: list[PendingPrompt]) -> str:
    """One paste for every pending prompt. A system prompt shared by several
    tasks (e.g. ten JDs in an eval) is written once and referenced."""
    if not prompts:
        return ""
    more = ', "<next id>": {...}' if len(prompts) > 1 else ""
    parts = [BUNDLE_HEADER.format(n=len(prompts), first_id=prompts[0].id, more=more)]
    names: dict[str, str] = {}
    for p in prompts:
        if p.system not in names:
            names[p.system] = f"I{len(names) + 1}"
            parts.append(f"{RULE}## INSTRUCTIONS {names[p.system]}\n\n{p.system.strip()}\n")
    for p in prompts:
        parts.append(f"{RULE}## TASK ID: {p.id}  (follow INSTRUCTIONS {names[p.system]})\n\n"
                     f"### INPUT\n{p.user.strip()}\n")
    parts.append(f"{RULE}Remember: reply with ONE JSON object keyed by the task IDs above.\n")
    return "".join(parts)


def _is_id(key: str) -> bool:
    return len(key) == ID_LEN and all(c in "0123456789abcdef" for c in key)


def parse_answers(text: str, expected: list[str] | None = None) -> tuple[dict[str, dict], list[str]]:
    """Parse a pasted reply into {id: answer}. Returns (answers, problems).

    Accepts the requested {"<id>": {...}} object (also inside ``` fences or
    after some prose), or, when exactly one prompt was pending, that one
    answer on its own."""
    from .llm_client import extract_json

    problems: list[str] = []
    obj = extract_json(text or "")
    if obj is None:
        return {}, ["The reply doesn't contain a JSON object."]
    expected = list(expected or [])
    answers = {k: v for k, v in obj.items() if isinstance(v, dict) and _is_id(k)}
    if not answers and len(expected) == 1:
        answers = {expected[0]: obj}           # a bare answer to the only prompt
    unknown = [k for k in answers if expected and k not in expected]
    if unknown:
        problems.append(f"Ignored answers for unknown task IDs: {', '.join(unknown)}")
        answers = {k: v for k, v in answers.items() if k not in unknown}
    missing = [k for k in expected if k not in answers]
    if missing:
        problems.append(f"No answer for task ID(s): {', '.join(missing)}")
    return answers, problems


def load_answers_file(path: str | Path) -> dict[str, dict]:
    """--answers FILE: the pasted reply saved to a file (any of the shapes
    parse_answers accepts). Every answer is stored in the cache."""
    answers, _problems = parse_answers(Path(path).read_text(encoding="utf-8"))
    for pid, ans in answers.items():
        store_answer(pid, ans)
    return answers


def copy_to_clipboard(text: str) -> bool:
    """Best effort, no dependency: pbcopy / clip / wl-copy / xclip / xsel."""
    candidates = (["pbcopy"], ["clip"], ["wl-copy"], ["xclip", "-selection", "clipboard"],
                  ["xsel", "--clipboard", "--input"])
    for cmd in candidates:
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode("utf-8"), check=True, timeout=5,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except (OSError, subprocess.SubprocessError):
                continue
    return False


def read_pasted_reply(stream=None, out=None) -> str:
    """Read a pasted reply from the terminal: everything up to a line that
    is just END (or end of input)."""
    stream = stream or sys.stdin
    out = out or sys.stderr
    print("Paste the AI's JSON reply, then a line with just END (or Ctrl-D):", file=out)
    lines = []
    for line in stream:
        if line.strip() == "END":
            break
        lines.append(line)
    return "".join(lines)


def run_with_answers(fn, answers_file: str | None = None, interactive: bool | None = None,
                     bundle_path: str | Path = "llm_prompts.md", max_passes: int = 8,
                     out=None, stream=None):
    """Run fn() until no manual prompt is left pending.

    Each pass: run fn(); if prompts are pending, write them to bundle_path
    (and the clipboard), then get the answers — pasted at the terminal when
    interactive, otherwise the caller gets fn()'s result back with
    `pending` listed so it can tell the user to re-run with --answers.
    Returns (result, still_pending: list[PendingPrompt])."""
    out = out or sys.stderr
    if interactive is None:
        interactive = sys.stdin.isatty()
    if answers_file:
        load_answers_file(answers_file)
    result = None
    for _ in range(max_passes):
        token = new_session()
        try:
            result = fn()
            todo = pending()
        finally:
            end_session(token)
        if not todo:
            return result, []
        bundle = build_bundle(todo)
        Path(bundle_path).write_text(bundle, encoding="utf-8")
        copied = copy_to_clipboard(bundle)
        print(f"\nCopy-paste LLM mode: {len(todo)} prompt(s) need an answer.\n"
              f"  Prompt bundle: {Path(bundle_path).resolve()}"
              + ("  (also copied to your clipboard)" if copied else "") +
              f"\n  Paste it into {' or '.join(CHAT_LINKS)} and bring back the JSON reply.",
              file=out)
        if not interactive:
            return result, todo
        reply = read_pasted_reply(stream=stream, out=out)
        got, problems = parse_answers(reply, [p.id for p in todo])
        for p in problems:
            print(f"  note: {p}", file=out)
        if not got:
            print("  No usable answers — stopping here.", file=out)
            return result, todo
        for pid, ans in got.items():
            store_answer(pid, ans)
    return result, pending()
