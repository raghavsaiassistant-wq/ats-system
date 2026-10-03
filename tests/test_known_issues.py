#!/usr/bin/env python3
"""Regression tests for the handoff doc's "small known issues" — offline,
no network, no real LLM.

Covers: no personal details in the shipped samples / profile template, a
manager-rubric dimension the model leaves out counting as "not scored"
(never 0), and Rich markup in JD / resume / LLM text printed literally
instead of being interpreted (or crashing the report).

    python tests/test_known_issues.py
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rich.console  # noqa: E402

from ats_checker import applog  # noqa: E402
from ats_checker import llm_client  # noqa: E402
from ats_checker import manager as mgr_mod  # noqa: E402
from ats_checker import profile as profile_mod  # noqa: E402
from ats_checker import report as report_mod  # noqa: E402
from ats_checker.scorer import run_full_check  # noqa: E402

passed = failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


SAMPLES = ROOT / "samples"
FULL = {"quantification": 80, "outcome_focus": 60, "evidence_backing": 70,
        "scope_match": 50, "domain_relevance": 90, "credibility": 40}


def stub_llm(payload_for):
    """Replace llm_client.call_json; payload_for(system_prompt) -> dict."""
    original = llm_client.call_json
    llm_client.call_json = lambda system, user, **kw: (payload_for(system), None)
    return original


def manager_with(payload: dict):
    original = stub_llm(lambda system: payload)
    try:
        return mgr_mod.score_manager_review("resume", "jd", model="m", host="h", api_key="k")
    finally:
        llm_client.call_json = original


# --------------------------------------------------------------- personal data
print("\n== no personal details in shipped samples / template ==")
emails = re.findall(r"[\w.+-]+@([\w-]+\.[\w.]+)", (SAMPLES / "sample_resume.txt").read_text(encoding="utf-8"))
check("sample_resume.txt email is on example.com", emails == ["example.com"], str(emails))
try:
    import docx

    doc_text = "\n".join(p.text for p in docx.Document(str(SAMPLES / "sample_resume.docx")).paragraphs)
    doc_emails = re.findall(r"[\w.+-]+@([\w-]+\.[\w.]+)", doc_text)
    check("sample_resume.docx email is on example.com", doc_emails == ["example.com"], str(doc_emails))
    check("sample .docx and .txt name the same fictional person",
          doc_text.splitlines()[0] == (SAMPLES / "sample_resume.txt").read_text(encoding="utf-8").splitlines()[0])
except ImportError as e:
    check("python-docx importable", False, str(e))
check("profile TEMPLATE says its values are a fictional example",
      "FICTIONAL EXAMPLE" in profile_mod.TEMPLATE)
check("profile TEMPLATE has no personal email/phone",
      not re.search(r"@|\+\d{2}\s?\d{5}", profile_mod.TEMPLATE))

# --------------------------------------------------------------- manager
print("\n== manager rubric: a missing dimension is 'not scored', not 0 ==")
full = manager_with(dict(FULL))
expected = round(sum(FULL[k] * w for k, w in mgr_mod.DIMENSION_WEIGHTS.items()))
check("all six scored: same weighted score as before", full.available and full.score == expected,
      f"{full.score} vs {expected}")
check("all six scored: nothing listed as not scored", full.not_scored == [])

partial = dict(FULL)
del partial["credibility"]          # weight 0.10, score 40
m = manager_with(partial)
renorm = round(sum(FULL[k] * w for k, w in mgr_mod.DIMENSION_WEIGHTS.items() if k != "credibility") / 0.9)
check("missing dimension is None, not 0", m.dimensions["credibility"] is None, str(m.dimensions))
check("missing dimension listed in not_scored", m.not_scored == ["credibility"], str(m.not_scored))
check("score = weighted mean of the scored dimensions", m.available and m.score == renorm, f"{m.score} vs {renorm}")
check("missing a LOW dimension doesn't drag the score down like a 0 would",
      m.score > round(sum((partial.get(k) or 0) * w for k, w in mgr_mod.DIMENSION_WEIGHTS.items())))

m = manager_with({**FULL, "outcome_focus": "high", "scope_match": True, "credibility": None,
                  "quantification": "85"})
check("junk / bool / null are not scored; numeric strings still count",
      m.dimensions["outcome_focus"] is None and m.dimensions["scope_match"] is None
      and m.dimensions["credibility"] is None and m.dimensions["quantification"] == 85, str(m.dimensions))
check("out-of-range values are clamped, not dropped",
      manager_with({**FULL, "quantification": 140}).dimensions["quantification"] == 100)

m = manager_with({"quantification": 80, "credibility": 40})   # 0.30 of the weight
check("under half the rubric scored -> layer unavailable, says why",
      not m.available and m.score is None and "2 of 6" in (m.error or ""), str(m.error))
check("to_dict exposes not_scored", "not_scored" in manager_with(partial).to_dict())

comps = applog.numeric_components({"manager_dimensions": {"quantification": 70, "credibility": None}})
check("logged components skip not-scored dimensions", comps == {"manager: quantification": 70.0}, str(comps))

# --------------------------------------------------------------- rich markup
print("\n== report: Rich markup in data prints literally ==")
EVIL = "[/dim] [bold red]boom[/bold red] [link=http://x.example]click[/link]"
JD = ("Job Title: Data Analyst\nLocation: Pune, India\nRequirements:\n"
      f"- 3+ years of SQL and Power BI {EVIL}\n- Bachelor's degree required\n")


def payload_for(system: str) -> dict:
    if "hiring manager" in system:
        p = {k: v for k, v in FULL.items() if k != "credibility"}
        p.update({"weak_bullets": [{"bullet": f"Built dashboards {EVIL}", "problem": f"vague {EVIL}",
                                    "rewrite": f"Built 12 dashboards {EVIL}"}],
                  "interview_risks": [f"risk {EVIL}"], "manager_verdict": f"verdict {EVIL}"})
        return p
    return {"semantic_score": 70, "strengths": [EVIL], "gaps": [f"gap {EVIL}"],
            "reworded_matches": [f"match {EVIL}"], "recommendation": f"rec {EVIL}"}


resume = (SAMPLES / "sample_resume.txt").read_text(encoding="utf-8") + f"\n- Wrote SQL {EVIL}\n"
original = stub_llm(payload_for)
try:
    rep = run_full_check(resume_text=resume, jd_text=JD, model="m", host="h", api_key="k")
finally:
    llm_client.call_json = original
check("stubbed LLM layers ran", rep.semantic_result.available and rep.manager_result.available,
      f"{rep.semantic_result.error} / {rep.manager_result.error}")
# force a JD citation + check detail carrying markup, and a blocker
if rep.recruiter_result.checks:
    rep.recruiter_result.checks[0].detail += f" {EVIL}"
    rep.recruiter_result.checks[0].jd_citation = f"citation {EVIL}"
rep.recruiter_result.blockers.append(f"blocker {EVIL}")
rep.parse_result.warnings.append(f"warning {EVIL}")

buf = io.StringIO()
orig_console = rich.console.Console
rich.console.Console = lambda *a, **k: orig_console(file=buf, width=400, color_system=None,
                                                    force_terminal=False)
try:
    try:
        report_mod.print_report(rep, show_checks=True)
        crashed = ""
    except Exception as e:  # noqa: BLE001 — a MarkupError here is the bug
        crashed = f"{type(e).__name__}: {e}"
finally:
    rich.console.Console = orig_console
out = buf.getvalue()
check("print_report doesn't crash on markup in the data", not crashed, crashed)
literal = "[bold red]boom[/bold red]"
for label, needle in (("LLM gaps", "gap [/dim]"), ("weak bullet", "Built dashboards [/dim]"),
                      ("manager verdict", "verdict [/dim]"), ("interview risk", "risk [/dim]"),
                      ("check detail", literal), ("JD citation", "citation [/dim]"),
                      ("blocker", "blocker [/dim]"), ("parse warning", "warning [/dim]"),
                      ("recommendation", "rec [/dim]")):
    check(f"{label} printed literally", needle in out, needle)
check("manager table shows 'not scored' for the missing dimension", "not scored" in out)
check("and explains the score isn't penalised for it", "didn't score 1 of 6" in out)

# the web UI renders a null dimension as "not scored", not "null/100"
import server  # noqa: E402

page = server.UI_PAGE
check("web UI shows 'not scored' for null dimensions", 'v === null ? "not scored"' in page)

print(f"\n{'=' * 60}\nKnown-issue tests: {passed} passed, {failed} failed")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all_checks_passed():
    """pytest entry point: the checks above run at import (collection) time."""
    assert failed == 0, f"{failed} check(s) failed — run this file directly for details"
