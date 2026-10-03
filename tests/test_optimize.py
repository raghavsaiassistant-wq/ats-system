#!/usr/bin/env python3
"""Optimize-loop tests (Phase 6) — offline, no network, stubbed LLM.

Covers: the loop only ever keeps a better (or equal) version, stops on a
plateau / when nothing truthful is left / at max rounds, beats-or-equals a
single tailor, never fabricates (canary: a JD skill the bank lacks must stay
in the gaps list however hard the "LLM" pushes it), proposals are NOT
applied unless accepted (--auto-accept is off by default), the no-apply gate
still blocks, unreviewed bullets are never used, and the CLI's copy-paste
round trip + --apply-proposals.

    python tests/test_optimize.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ats_checker import llm_client  # noqa: E402
from ats_checker import manager as mgr_mod  # noqa: E402
from ats_checker import profile as profile_mod  # noqa: E402
from ats_checker.generator import (  # noqa: E402
    Bullet, Education, EvidenceBank, Role, save_bank, tailor,
)
from ats_checker.generator import loop  # noqa: E402
from ats_checker.generator.rewriter import REWORD_SYSTEM_PROMPT, RewriteResult  # noqa: E402

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
SAMPLE_JD = (SAMPLES / "sample_jd.txt").read_text(encoding="utf-8")
SAMPLE_PROFILE = profile_mod.load_profile(str(SAMPLES / "profile.yaml"))
TD = tempfile.TemporaryDirectory()
TMP = Path(TD.name)
_OLD_HOME = os.environ.get("ATS_HOME")
os.environ["ATS_HOME"] = str(TMP / "ats_home")       # isolated answer / JD caches


def monotonic(res) -> bool:
    scores = [r.score for r in res.rounds]
    return all(b >= a - 1e-9 for a, b in zip(scores, scores[1:]))


def bank_lines(bank: EvidenceBank) -> set[str]:
    return {b.text for r in bank.roles for b in r.bullets}


# --------------------------------------------------------------- samples, offline
print("\n== offline optimize on the samples ==")
res = loop.optimize(str(SAMPLES / "master_resume.yaml"), SAMPLE_JD, profile=SAMPLE_PROFILE,
                    offline=True)
single = tailor(master_path=str(SAMPLES / "master_resume.yaml"), jd_text=SAMPLE_JD,
                profile=SAMPLE_PROFILE, offline=True)
tailor_score, _ = loop.objective(single.report)
check("optimize >= a single tailor", res.final_score >= tailor_score - 1e-9,
      f"{res.final_score} vs {tailor_score}")
check("round 0 is exactly tailor's draft", res.rounds[0].score == tailor_score)
check("scores never go down", monotonic(res), str([r.score for r in res.rounds]))
check("it stops and says why", bool(res.stop_reason), res.stop_reason)
check("canary: bank lacks Snowflake -> not in the resume", "snowflake" not in res.resume_text.lower())
snow = [g for g in res.gaps if g.term.lower() == "snowflake"]
check("canary: Snowflake listed as 'nothing in your bank', required",
      snow and snow[0].status == "nothing-in-bank" and snow[0].required, str(snow))
check("offline: no rewordings proposed", res.proposals == [])
check("result is JSON-serialisable", json.loads(json.dumps(res.to_dict()))["final_score"] == res.final_score)

# --------------------------------------------------------------- selection climbs
print("\n== selection moves climb from a poor start ==")
weak_bank = EvidenceBank(
    name="T", email="t@example.com", phone="+91 90000 00000", headline="BI Analyst",
    roles=[
        Role("NowCo", "BI Analyst", "2024-01", "present", bullets=[
            Bullet("Organised the team offsite and the quarterly lunch", ["events"]),
            Bullet("Answered internal emails about the office move", ["communication"]),
            Bullet("Built 25 Power BI dashboards with DAX measures for sales", ["power bi", "dax"]),
            Bullet("Wrote SQL queries feeding the finance warehouse", ["sql"]),
            Bullet("Automated weekly reporting with Python scripts", ["python", "automation"]),
        ]),
    ],
    education=[Education("BBA", "Test College", "2022")],
)
weak_path = TMP / "weak.yaml"
save_bank(weak_bank, weak_path)
real_select = loop.select


def worst_select(bank, jd_keywords, **kw):
    """Start from the two bullets that cover nothing in the JD."""
    role = bank.roles[0]
    jd_map, reps = loop.build_jd_map(jd_keywords)
    poor = [loop.ChosenBullet(bullet=b, role=role, text=b.text) for b in role.bullets[:2]]
    return loop._rebuild(poor, bank, jd_map, reps)


loop.select = worst_select
try:
    res = loop.optimize(str(weak_path), SAMPLE_JD, profile=SAMPLE_PROFILE, offline=True,
                        max_current=4)
finally:
    loop.select = real_select
check("climbs above the poor start", res.final_score > res.rounds[0].score,
      str([r.score for r in res.rounds]))
check("scores never go down", monotonic(res), str([r.score for r in res.rounds]))
moves = [c["what"] for r in res.rounds[1:] for c in r.changes]
check("only swap/add moves offline", moves and all(m.startswith(("swap", "add")) for m in moves), str(moves))
resume_bullets = {ln[2:] for ln in res.resume_text.splitlines() if ln.startswith("- ")}
check("every resume bullet is a bank bullet, verbatim", resume_bullets <= bank_lines(weak_bank),
      str(resume_bullets - bank_lines(weak_bank)))
check("the Power BI bullet made it in", any("Power BI dashboards" in b for b in resume_bullets))

# --------------------------------------------------------------- plateau
print("\n== stops on a plateau (two rounds gaining < 1 point) ==")
plateau_bank = EvidenceBank(
    name="T", email="t@example.com", phone="+91 90000 00000", headline="Analyst",
    roles=[Role("NowCo", "Analyst", "2024-01", "present", bullets=[
        Bullet("Wrote queries for the finance team", ["sql"]),
        Bullet("Cleaned spreadsheets for the sales team", ["excel"]),
        Bullet("Prepared charts for the operations team", ["power bi"]),
        Bullet("Shared summaries with the leadership team", ["python"]),
    ])],
    education=[Education("BBA", "Test College", "2022")],
)
plateau_path = TMP / "plateau.yaml"
save_bank(plateau_bank, plateau_path)


RealScorer = loop._Scorer


class FakeScorer:
    """+0.5 per 'reviewed monthly' rewording, -5 for a 'worse' one."""
    def __init__(self, jd_text, profile):
        self.real = RealScorer(jd_text, profile)

    def __call__(self, text):
        _s, comps, rep = self.real(text)
        return 60 + 0.5 * text.count("reviewed monthly") - 5 * text.count("worse"), comps, rep


def fake_reword(chosen, targets, **kw):
    """One good small rewording per call, plus one that lowers the score."""
    out = [RewriteResult(0, chosen[0].text, chosen[0].text + ", reviewed monthly", True)]
    if len(chosen) > 1:
        out.append(RewriteResult(1, chosen[1].text, chosen[1].text + ", worse", True))
    return out


saved = (loop._Scorer, loop.reword_for_terms, loop.mgr_mod.score_manager_review)
loop._Scorer = FakeScorer
loop.reword_for_terms = fake_reword
loop.mgr_mod.score_manager_review = lambda **kw: mgr_mod.ManagerResult(available=False, error="stub")
try:
    res = loop.optimize(str(plateau_path), SAMPLE_JD, profile=SAMPLE_PROFILE, offline=False,
                        max_rounds=5)
finally:
    loop._Scorer, loop.reword_for_terms, loop.mgr_mod.score_manager_review = saved
check("stopped on the plateau", res.stop_reason.startswith("plateau"), res.stop_reason)
check("after exactly 2 small-gain rounds", len(res.rounds) == 3, str([r.score for r in res.rounds]))
check("scores never go down", monotonic(res), str([r.score for r in res.rounds]))
check("the score-lowering rewording was skipped",
      any("lower the score" in n for n in res.notes) and "worse" not in "".join(
          p.rewrite for p in res.proposals), str(res.notes))
check("two proposals, none accepted yet",
      len(res.proposals) == 2 and all(p.accepted is None for p in res.proposals))
check("not accepted -> final file keeps the original wording",
      "reviewed monthly" not in res.resume_text)
loop.finalize(res, auto_accept=True)
check("auto_accept applies them", res.resume_text.count("reviewed monthly") == 2)
loop.finalize(res, accepted_ids={1})
check("accepting one applies exactly one", res.resume_text.count("reviewed monthly") == 1)

res2 = None
loop._Scorer = FakeScorer
loop.reword_for_terms = fake_reword
loop.mgr_mod.score_manager_review = lambda **kw: mgr_mod.ManagerResult(available=False, error="stub")
try:
    res2 = loop.optimize(str(plateau_path), SAMPLE_JD, profile=SAMPLE_PROFILE, offline=False,
                         max_rounds=1)
finally:
    loop._Scorer, loop.reword_for_terms, loop.mgr_mod.score_manager_review = saved
check("--max-rounds is respected", len(res2.rounds) == 2 and "max-rounds" in res2.stop_reason,
      res2.stop_reason)

# --------------------------------------------------------------- fabrication canary
print("\n== fabrication canary: the LLM pushes Snowflake, the bank lacks it ==")
canary_bank = EvidenceBank(
    name="T", email="t@example.com", phone="+91 90000 00000", headline="BI Analyst",
    roles=[Role("NowCo", "BI Analyst", "2024-01", "present", bullets=[
        Bullet("Built dashboards for the sales team", ["power bi", "dashboard"]),
        Bullet("Wrote queries feeding the finance warehouse", ["sql"]),
        Bullet("Automated weekly reporting for three teams", ["python", "automation"]),
    ])],
    education=[Education("BBA", "Test College", "2022")],
)
canary_path = TMP / "canary.yaml"
save_bank(canary_bank, canary_path)
CANARY_JD = ("Job Title: BI Analyst\nRequirements:\n- 2+ years with Power BI and SQL\n"
             "- Hands-on Snowflake experience\n- Python for automation\n")


def canary_llm(system, user, **kw):
    if system == REWORD_SYSTEM_PROMPT:
        items = json.loads(user.split("\n", 1)[1].split("\n\nEach bullet")[0])["bullets"]
        out = []
        for it in items:
            if "dashboards" in it["bullet"]:
                out.append({"index": it["index"], "rewrite": "Built Power BI dashboards for the sales team"})
            else:
                out.append({"index": it["index"], "rewrite": it["bullet"] + " on Snowflake"})
        return {"rewrites": out}, None
    if "hiring manager" in system:
        return {"quantification": 50, "outcome_focus": 50, "evidence_backing": 50,
                "scope_match": 50, "domain_relevance": 50, "credibility": 50,
                "weak_bullets": [{"bullet": "Wrote queries feeding the finance warehouse",
                                  "problem": "vague",
                                  "rewrite": "Wrote SQL queries feeding the Snowflake finance warehouse"}],
                "interview_risks": [], "manager_verdict": "ok"}, None
    return {"semantic_score": 60, "strengths": [], "gaps": [], "reworded_matches": [],
            "recommendation": ""}, None


real_call = llm_client.call_json
llm_client.call_json = canary_llm
try:
    res = loop.optimize(str(canary_path), CANARY_JD, offline=False, model="m", host="h", api_key="k")
finally:
    llm_client.call_json = real_call
loop.finalize(res, auto_accept=True)          # even accepting everything...
check("...Snowflake never reaches the resume", "snowflake" not in res.resume_text.lower(),
      res.resume_text)
check("Snowflake stays in the gaps list as 'nothing in your bank'",
      any(g.term.lower() == "snowflake" and g.status == "nothing-in-bank" for g in res.gaps),
      str([(g.term, g.status) for g in res.gaps]))
check("the legit rewording (tag-backed Power BI) was proposed",
      any("Power BI dashboards" in p.rewrite for p in res.proposals), str([p.rewrite for p in res.proposals]))
check("no proposal mentions Snowflake", not any("snowflake" in p.rewrite.lower() for p in res.proposals))
check("rejections are reported", any("Snowflake".lower() in n.lower() or "rejected" in n.lower()
                                     for n in res.notes), str(res.notes))
check("every proposal passes the truth check again",
      all(loop._rewrite_is_truthful(loop.ChosenBullet(
          bullet=next(b for b in canary_bank.roles[0].bullets if b.text == p.original),
          role=canary_bank.roles[0], text=p.original), p.rewrite,
          loop.build_jd_map(res._jd_keywords)[0])[0] for p in res.proposals))

# --------------------------------------------------------------- gate + unreviewed
print("\n== no-apply gate and unreviewed bullets ==")
gate_jd = ("Job Title: BI Analyst\n\nRequirements:\n- No visa sponsorship available for this role\n"
           "- 1-3 years of experience\n- Strong SQL skills\n")
res = loop.optimize(str(SAMPLES / "master_resume.yaml"), gate_jd, profile=SAMPLE_PROFILE, offline=True)
check("gate blocks", res.blocked and any("sponsor" in b.lower() for b in res.candidacy_blockers),
      str(res.candidacy_blockers))
check("blocked run writes no resume", res.resume_text == "" and res.rounds == [])
res = loop.optimize(str(SAMPLES / "master_resume.yaml"), gate_jd, profile=SAMPLE_PROFILE,
                    offline=True, force=True)
check("--force optimizes anyway", not res.blocked and res.resume_text)

unrev = EvidenceBank(
    name="T", email="t@example.com", phone="+91 90000 00000", headline="BI Analyst",
    roles=[Role("NowCo", "BI Analyst", "2024-01", "present", bullets=[
        Bullet("Built Power BI dashboards for the sales team", ["power bi"]),
        Bullet("Migrated the warehouse to Snowflake", ["snowflake"], reviewed=False),
    ])],
    education=[Education("BBA", "Test College", "2022")],
)
unrev_path = TMP / "unrev.yaml"
save_bank(unrev, unrev_path)
res = loop.optimize(str(unrev_path), CANARY_JD, offline=True)
check("unreviewed bullet never used", "snowflake" not in res.resume_text.lower())
check("its term is reported as 'only an unconfirmed bullet'",
      any(g.term.lower() == "snowflake" and g.status == "unconfirmed" for g in res.gaps),
      str([(g.term, g.status) for g in res.gaps]))
check("and the run says bullets were left out", any("unreviewed" in p for p in res.bank_problems))

# --------------------------------------------------------------- CLI
print("\n== CLI: optimize in copy-paste mode, proposals, --apply-proposals ==")


def cli(*argv):
    env = {**os.environ, "ATS_HOME": str(TMP / "cli_home"), "ATS_LLM_PROVIDER": "ollama",
           "ATS_LLM_BASE_URL": "ollama-local"}
    return subprocess.run([sys.executable, str(ROOT / "cli.py"), *argv], capture_output=True,
                          text=True, cwd=str(TMP), env=env, encoding="utf-8", errors="replace",
                          stdin=subprocess.DEVNULL)


def answer_bundle(bundle: str) -> dict:
    instr = dict(re.findall(r"## INSTRUCTIONS (I\d+)\n\n(.*?)\n-{20,}", bundle, re.S))
    tasks = re.findall(r"## TASK ID: ([0-9a-f]{12})  \(follow INSTRUCTIONS (I\d+)\)\n\n### INPUT\n(.*?)\n-{20,}",
                       bundle, re.S)
    out = {}
    for tid, i, user in tasks:
        system = instr[i].strip()
        ans, _ = canary_llm(REWORD_SYSTEM_PROMPT if system == REWORD_SYSTEM_PROMPT.strip() else system,
                            user.strip())
        out[tid] = ans
    return out


cli_jd = TMP / "jd.txt"
cli_jd.write_text(CANARY_JD, encoding="utf-8")
base = ["optimize", "--master", str(canary_path), "--jd", str(cli_jd), "--profile", str(TMP / "none.yaml"),
        "--provider", "manual", "--out", str(TMP / "out.txt"), "--no-docx"]
p = cli(*base)
check("first run: prompts pending -> exit 4 and a bundle", p.returncode == 4 and (TMP / "llm_prompts.md").exists(),
      p.stderr[-400:])
answers_file = TMP / "answers.json"
rounds = 0
while p.returncode == 4 and rounds < 6:
    rounds += 1
    prev = json.loads(answers_file.read_text()) if answers_file.exists() else {}
    prev.update(answer_bundle((TMP / "llm_prompts.md").read_text(encoding="utf-8")))
    answers_file.write_text(json.dumps(prev), encoding="utf-8")
    p = cli(*base, "--answers", str(answers_file))
check("answered runs finish (exit 0)", p.returncode == 0, f"rc={p.returncode} {p.stderr[-400:]}")
props_path = TMP / "out.proposals.json"
check("non-interactive: proposals written, not applied",
      props_path.exists() and "Power BI dashboards" not in (TMP / "out.txt").read_text(encoding="utf-8"),
      p.stdout[-600:])
props = json.loads(props_path.read_text(encoding="utf-8")) if props_path.exists() else []
check("proposals file has no Snowflake", props and not any("snowflake" in x["rewrite"].lower() for x in props),
      str(props))
check("proposals start unaccepted", props and all(x["accepted"] is False for x in props), str(props))
for x in props:
    x["accepted"] = True
props_path.write_text(json.dumps(props), encoding="utf-8")
p = cli(*base, "--apply-proposals", str(props_path))
final = (TMP / "out.txt").read_text(encoding="utf-8")
check("--apply-proposals puts the accepted rewording in", p.returncode == 0 and "Built Power BI dashboards" in final,
      p.stdout[-400:] + p.stderr[-400:])
check("...and still no Snowflake", "snowflake" not in final.lower())
p = cli(*base, "--json")
d = json.loads(p.stdout) if p.returncode == 0 else {}
check("--json output has rounds, gaps and the review note",
      d.get("rounds") and "gaps_truth_cannot_close" in d and "review" in d, p.stderr[-300:])
check("cached answers: a re-run needs no paste", p.returncode == 0)

p = cli("optimize", "--master", str(SAMPLES / "master_resume.yaml"), "--jd", str(cli_jd),
        "--profile", str(SAMPLES / "profile.yaml"), "--offline", "--out", str(TMP / "o2.txt"))
check("offline CLI optimize writes .txt and .docx",
      p.returncode == 0 and (TMP / "o2.txt").exists() and (TMP / "o2.docx").exists(), p.stderr[-300:])
check("CLI prints the gaps table", "Gaps the truth can't close" in p.stdout)

from cli import build_parser  # noqa: E402

args = build_parser().parse_args(["optimize", "--jd", "x.txt"])
check("--auto-accept is off by default", args.auto_accept is False)

p = cli("run", "--resume", str(SAMPLES / "sample_resume.txt"), "--jd", str(cli_jd),
        "--master", str(TMP / "fresh_bank.yaml"), "--profile", str(TMP / "p.yaml"), "--offline")
check("run without a bank and --offline explains what's needed",
      p.returncode == 1 and "evidence bank" in p.stderr, p.stderr[-300:])
p = cli("run", "--resume", str(SAMPLES / "sample_resume.txt"), "--jd", str(cli_jd),
        "--master", str(unrev_path), "--profile", str(TMP / "p.yaml"), "--offline",
        "--out", str(TMP / "run.txt"))
check("run with a bank: optimizes end to end, says unreviewed bullets wait",
      p.returncode == 0 and (TMP / "run.txt").exists() and "unreviewed" in p.stderr,
      p.stderr[-300:])

TD.cleanup()
if _OLD_HOME is None:
    os.environ.pop("ATS_HOME", None)
else:
    os.environ["ATS_HOME"] = _OLD_HOME
print(f"\n{'=' * 60}\nOptimize tests: {passed} passed, {failed} failed")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all_checks_passed():
    """pytest entry point: the checks above run at import (collection) time."""
    assert failed == 0, f"{failed} check(s) failed — run this file directly for details"
