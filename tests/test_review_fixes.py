#!/usr/bin/env python3
"""Regression tests for the Phase 6 review findings — offline, stubbed LLM.

  0  no JD skill the bank lacks can reach tailor()/`/tailor` output, even
     when the "LLM" writes it into an unrelated bullet (no target fallback,
     the shared tag-claim truth rule, unsent bullets' answers ignored)
  1  --auto-accept never applies a manager rewrite
  2  copy-paste prompts still pending -> "waiting for pasted answers",
     pending_ids in --json, no resume file
  3  --answers implies --provider manual
  4  a note for every dropped rewording
  5  the manager review reads the round-0 draft (one bundle with both
     prompts), and a rejected bullet isn't re-asked
  6  finalize() matches proposals to bullets by identity, not text

    python tests/test_review_fixes.py
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
TD = tempfile.TemporaryDirectory()
TMP = Path(TD.name)
_OLD_HOME = os.environ.get("ATS_HOME")
os.environ["ATS_HOME"] = str(TMP / "home")

from ats_checker import llm_client, manual_llm  # noqa: E402
from ats_checker import manager as mgr_mod  # noqa: E402
from ats_checker import profile as profile_mod  # noqa: E402
from ats_checker.generator import (  # noqa: E402
    Bullet, Education, EvidenceBank, Role, plausible_targets, reword_for_terms, save_bank, tailor,
)
from ats_checker.generator import loop  # noqa: E402
from ats_checker.generator.rewriter import REWORD_SYSTEM_PROMPT, RewriteResult  # noqa: E402
from ats_checker.generator.selector import ChosenBullet  # noqa: E402

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
ROLE = Role("NowCo", "BI Analyst", "2024-01", "present")


def mgr_payload(weak: list[dict]) -> dict:
    return {"quantification": 50, "outcome_focus": 50, "evidence_backing": 50, "scope_match": 50,
            "domain_relevance": 50, "credibility": 50, "weak_bullets": weak,
            "interview_risks": [], "manager_verdict": "ok"}


SEMANTIC = {"semantic_score": 60, "strengths": [], "gaps": [], "reworded_matches": [],
            "recommendation": ""}


class StubLLM:
    """Plays every LLM role; the reword answer writes Snowflake into EVERY
    bullet it is shown, and (optionally) into bullets it was NOT shown."""

    def __init__(self, weak=None, extra_indices=()):
        self.weak = weak or []
        self.extra = extra_indices
        self.reword_calls: list[list[str]] = []

    def __call__(self, system, user, **kw):
        if system == REWORD_SYSTEM_PROMPT:
            items = json.loads(user.split("\n", 1)[1].split("\n\nEach bullet")[0])["bullets"]
            self.reword_calls.append([it["bullet"] for it in items])
            out = [{"index": it["index"], "rewrite": it["bullet"] + " on Snowflake"} for it in items]
            out += [{"index": i, "rewrite": "Migrated the warehouse to Snowflake"} for i in self.extra]
            return {"rewrites": out}, None
        if "hiring manager" in system:
            return mgr_payload(self.weak), None
        return SEMANTIC, None


def with_stub(stub):
    real = llm_client.call_json
    llm_client.call_json = stub
    return real


# ===================================================================== 0
print("\n== 0: no target fallback; Snowflake never reaches tailor output ==")
cb_plain = ChosenBullet(bullet=Bullet("Organised the quarterly offsite", ["events"]), role=ROLE,
                        text="Organised the quarterly offsite")
check("a bullet with no signal gets NO targets (fallback removed)",
      plausible_targets(cb_plain, ["snowflake", "aws"]) == [], str(plausible_targets(cb_plain, ["snowflake"])))

stub = StubLLM(extra_indices=(0,))
real = with_stub(stub)
try:
    res = reword_for_terms([cb_plain], ["snowflake"], provider="openai")
finally:
    llm_client.call_json = real
check("no tag claims the term -> the LLM isn't even asked", stub.reword_calls == [] and res == [],
      str(stub.reword_calls))

cb_pbi = ChosenBullet(bullet=Bullet("Built dashboards for the sales team", ["power bi"]), role=ROLE,
                      text="Built dashboards for the sales team")
stub = StubLLM(extra_indices=(1,))       # also answers for bullet 1, which is never sent
real = with_stub(stub)
try:
    res = reword_for_terms([cb_pbi, cb_plain], ["power bi", "snowflake"], provider="openai")
finally:
    llm_client.call_json = real
check("only the tag-claiming bullet is sent", stub.reword_calls == [["Built dashboards for the sales team"]],
      str(stub.reword_calls))
check("an answer for an unsent bullet is ignored, not trusted", all(r.index == 0 for r in res), str(res))
check("Snowflake written into the sent bullet fails verification",
      res and not res[0].verified and "snowflake" in res[0].note.lower(), str(res))

snow_weak = [{"bullet": "Built Power BI dashboards with DAX for the sales leadership team",
              "problem": "vague", "rewrite": "Built Power BI dashboards with DAX on Snowflake for the "
                                             "sales leadership team"}]
stub = StubLLM(weak=snow_weak)
real = with_stub(stub)
try:
    t = tailor(master_path=str(SAMPLES / "master_resume.yaml"), jd_text=SAMPLE_JD,
               profile=SAMPLE_PROFILE, offline=False, force=True, model="m", host="h", api_key="k")
finally:
    llm_client.call_json = real
check("tailor: no reword prompt carrying Snowflake was ever sent",
      not any("snowflake" in b.lower() for call in stub.reword_calls for b in call), str(stub.reword_calls))
check("tailor: Snowflake never reaches the resume", "snowflake" not in t.resume_text.lower(), t.resume_text[:400])
check("tailor: Snowflake stays an honest gap", any("snowflake" in g for g in t.honest_gaps), str(t.honest_gaps))

import server  # noqa: E402

stub = StubLLM(weak=snow_weak)
real = with_stub(stub)
try:
    r = server.app.test_client().post("/tailor", json={
        "jd_text": SAMPLE_JD, "master_path": str(SAMPLES / "master_resume.yaml"),
        "profile_path": str(SAMPLES / "profile.yaml"), "force": True})
finally:
    llm_client.call_json = real
body = r.get_json() or {}
check("/tailor: Snowflake never reaches the resume",
      r.status_code == 200 and body.get("resume_text") and "snowflake" not in body["resume_text"].lower(),
      str(body)[:300])

# ===================================================================== 1
print("\n== 1: --auto-accept never applies a manager rewrite ==")
PRICING = "Prepared ad-hoc analysis decks that guided pricing decisions for the category managers"
pricing_bank = EvidenceBank(
    name="T", email="t@example.com", phone="+91 90000 00000", headline="Pricing Analyst",
    roles=[Role("NowCo", "Pricing Analyst", "2024-01", "present", bullets=[
        Bullet(PRICING, ["analysis"]),
        Bullet("Wrote SQL queries feeding the pricing model", ["sql"]),
    ])],
    education=[Education("BBA", "Test College", "2022")],
)
pricing_path = TMP / "pricing.yaml"
save_bank(pricing_bank, pricing_path)
PRICING_JD = "Job Title: Pricing Analyst\nRequirements:\n- SQL\n- Pricing analysis for category managers\n"
REWRITE = ("Owned pricing strategy for the category managers, building ad-hoc analysis decks that "
           "guided pricing decisions")
stub = StubLLM(weak=[{"bullet": PRICING, "problem": "duties, not outcomes", "rewrite": REWRITE}])
real = with_stub(stub)
try:
    res = loop.optimize(str(pricing_path), PRICING_JD, offline=False, model="m", host="h", api_key="k")
finally:
    llm_client.call_json = real
mgr_props = [p for p in res.proposals if p.source == "manager-rewrite"]
check("the manager rewrite came back as a proposal", len(mgr_props) == 1 and mgr_props[0].rewrite == REWRITE,
      str([(p.source, p.rewrite) for p in res.proposals]) + str(res.notes))
check("it is flagged needs_review", mgr_props and mgr_props[0].needs_review
      and mgr_props[0].to_dict()["needs_review"] is True)
loop.finalize(res, auto_accept=True)
check("auto_accept leaves it out: the original wording stays", PRICING in res.resume_text
      and "Owned pricing strategy" not in res.resume_text, res.resume_text)
check("...and it stays undecided, not accepted", mgr_props and mgr_props[0].accepted is None)
loop.finalize(res, accepted_ids={mgr_props[0].id} if mgr_props else set())
check("an explicit yes applies it", "Owned pricing strategy" in res.resume_text)

# ===================================================================== 4 + 5
print("\n== 4/5: a note for every drop; round-0 manager review; no re-asking ==")
two_bank = EvidenceBank(
    name="T", email="t@example.com", phone="+91 90000 00000", headline="BI Analyst",
    roles=[Role("NowCo", "BI Analyst", "2024-01", "present", bullets=[
        Bullet("Built dashboards for the sales team", ["power bi"]),
        Bullet("Wrote queries feeding the finance warehouse", ["sql"]),
    ])],
    education=[Education("BBA", "Test College", "2022")],
)
two_path = TMP / "two.yaml"
save_bank(two_bank, two_path)
TWO_JD = "Job Title: BI Analyst\nRequirements:\n- Power BI\n- SQL\n- Snowflake\n"

tok = manual_llm.new_session()
try:
    res = loop.optimize(str(two_path), TWO_JD, offline=False, provider="manual")
    pend = manual_llm.pending()
finally:
    manual_llm.end_session(tok)
systems = {p.system for p in pend}
check("pass 1 bundles BOTH the rewording and the manager prompt",
      REWORD_SYSTEM_PROMPT in systems and any("hiring manager" in s for s in systems), str(len(pend)))
check("2: stop_reason says it is waiting for pasted answers",
      res.stop_reason == "waiting for pasted answers", res.stop_reason)
check("2: pending_ids are reported", sorted(res.pending_ids) == sorted(p.id for p in pend), str(res.pending_ids))
check("2: pending_ids are in to_dict()", res.to_dict()["pending_ids"] == res.pending_ids)
check("no 'Rewording skipped' noise for answers that are merely pending",
      not any("pending" in n for n in res.notes), str(res.notes))

calls = []


def reword_one_good_one_bad(chosen, targets, **kw):
    calls.append([cb.bullet.text for cb in chosen])
    out = []
    for i, cb in enumerate(chosen):
        if "dashboards" in cb.bullet.text:
            out.append(RewriteResult(i, cb.text, "Built Power BI dashboards for the sales team", True))
        else:
            out.append(RewriteResult(i, cb.text, cb.text + " on Snowflake", True))   # fabricates
    return out


saved = (loop.reword_for_terms, loop.mgr_mod.score_manager_review)
loop.reword_for_terms = reword_one_good_one_bad
loop.mgr_mod.score_manager_review = lambda **kw: mgr_mod.ManagerResult(
    available=True, score=50, weak_bullets=[{"bullet": "Wrote queries feeding the finance warehouse",
                                             "problem": "x", "rewrite": "Wrote 40 SQL queries for it"}])
two_bank.roles[0].bullets[1].skills = ["sql", "data warehouse"]   # so both bullets get a target
save_bank(two_bank, two_path)
try:
    res = loop.optimize(str(two_path), TWO_JD + "- Data warehouse\n", offline=False, max_rounds=4)
finally:
    loop.reword_for_terms, loop.mgr_mod.score_manager_review = saved
check("4: the fabricating rewording has a note", any("Dropped a gap-rewording" in n and "snowflake" in n.lower()
                                                     for n in res.notes), str(res.notes))
check("5: the rejected bullet is never asked about again (one reword call only)",
      len(calls) == 1, str(calls))
check("5: the manager suggestion for the rejected bullet isn't tried", not any(
    "Dropped a manager-rewrite" in n for n in res.notes), str(res.notes))

calls.clear()
saved = (loop.reword_for_terms, loop.mgr_mod.score_manager_review)
loop.reword_for_terms = lambda chosen, targets, **kw: [
    RewriteResult(0, chosen[0].text, chosen[0].text + " 99 times", False, "added number(s) ['99']")]
loop.mgr_mod.score_manager_review = lambda **kw: mgr_mod.ManagerResult(
    available=True, score=50, weak_bullets=[{"bullet": "Built dashboards for the sales team",
                                             "problem": "x",
                                             "rewrite": "Built dashboards on Snowflake for the sales team"}])
try:
    res = loop.optimize(str(two_path), TWO_JD, offline=False)
finally:
    loop.reword_for_terms, loop.mgr_mod.score_manager_review = saved
check("4: a verify_rewrite failure is reported with its reason",
      any("Dropped a gap-rewording" in n and "added number" in n for n in res.notes), str(res.notes))

saved = (loop.reword_for_terms, loop.mgr_mod.score_manager_review)
loop.reword_for_terms = lambda chosen, targets, **kw: []
loop.mgr_mod.score_manager_review = lambda **kw: mgr_mod.ManagerResult(
    available=True, score=50, weak_bullets=[{"bullet": "Built dashboards for the sales team",
                                             "problem": "x",
                                             "rewrite": "Built dashboards on Snowflake for the sales team"}])
try:
    res = loop.optimize(str(two_path), TWO_JD, offline=False)
finally:
    loop.reword_for_terms, loop.mgr_mod.score_manager_review = saved
check("4: a fabrication-guard / verify drop of a manager rewrite is reported",
      any("Dropped a manager-rewrite" in n and "snowflake" in n.lower() for n in res.notes), str(res.notes))

# ===================================================================== 6
print("\n== 6: finalize matches by bullet identity, not text ==")
twin = "Built dashboards for the sales team"
twin_bank = EvidenceBank(
    name="T", email="t@example.com", phone="+91 90000 00000", headline="BI Analyst",
    roles=[Role("NowCo", "BI Analyst", "2024-01", "present", bullets=[Bullet(twin, ["power bi"])]),
           Role("PastCo", "Analyst", "2022-01", "2023-12", bullets=[Bullet(twin, ["power bi"])])],
    education=[Education("BBA", "Test College", "2022")],
)
twin_path = TMP / "twin.yaml"
save_bank(twin_bank, twin_path)


def reword_first_only(chosen, targets, **kw):
    return [RewriteResult(0, chosen[0].text, "Built Power BI dashboards for the sales team", True)]


saved = (loop.reword_for_terms, loop.mgr_mod.score_manager_review)
loop.reword_for_terms = reword_first_only
loop.mgr_mod.score_manager_review = lambda **kw: mgr_mod.ManagerResult(available=False, error="stub")
try:
    res = loop.optimize(str(twin_path), "Job Title: BI Analyst\nRequirements:\n- Power BI\n",
                        offline=False, max_rounds=1)
finally:
    loop.reword_for_terms, loop.mgr_mod.score_manager_review = saved
check("one proposal for one of the two identical bullets", len(res.proposals) == 1, str(res.proposals))
loop.finalize(res, auto_accept=True)
lines = [ln for ln in res.resume_text.splitlines() if ln.startswith("- ")]
check("accepting it rewrites exactly that bullet, not its twin",
      lines.count("- Built Power BI dashboards for the sales team") == 1 and lines.count(f"- {twin}") == 1,
      str(lines))

# ===================================================================== 2 + 3 (CLI)
print("\n== 2/3: CLI pending output; --answers implies --provider manual ==")


def cli(*argv):
    env = {**os.environ, "ATS_HOME": str(TMP / "cli_home"), "ATS_LLM_PROVIDER": "ollama",
           "ATS_LLM_BASE_URL": "ollama-local"}
    return subprocess.run([sys.executable, str(ROOT / "cli.py"), *argv], capture_output=True, text=True,
                          cwd=str(TMP), env=env, encoding="utf-8", errors="replace",
                          stdin=subprocess.DEVNULL)


def answer_bundle(bundle: str, weak=None) -> dict:
    instr = dict(re.findall(r"## INSTRUCTIONS (I\d+)\n\n(.*?)\n-{20,}", bundle, re.S))
    tasks = re.findall(r"## TASK ID: ([0-9a-f]{12})  \(follow INSTRUCTIONS (I\d+)\)\n\n### INPUT\n(.*?)\n-{20,}",
                       bundle, re.S)
    out = {}
    for tid, i, user in tasks:
        system = instr[i].strip()
        if system == REWORD_SYSTEM_PROMPT.strip():
            out[tid] = {"rewrites": []}
        elif "hiring manager" in system:
            out[tid] = mgr_payload(weak or [])
        else:
            out[tid] = SEMANTIC
    return out


jd_file = TMP / "pricing_jd.txt"
jd_file.write_text(PRICING_JD, encoding="utf-8")
out_txt = TMP / "pricing_out.txt"
base = ["optimize", "--master", str(pricing_path), "--jd", str(jd_file), "--profile", str(TMP / "none.yaml"),
        "--out", str(out_txt), "--no-docx"]
p = cli(*base, "--provider", "manual", "--json")
d = json.loads(p.stdout) if p.stdout.strip() else {}
check("2: pending -> exit 4", p.returncode == 4, p.stderr[-300:])
check("2: --json says waiting, lists pending_ids", d.get("stop_reason") == "waiting for pasted answers"
      and len(d.get("pending_ids", [])) >= 1, str(d)[:300])
check("2: no resume file is written while answers are pending",
      not out_txt.exists() and d.get("resume_text") == "" and d.get("outputs") == {})

answers = TMP / "pricing_answers.json"
answers.write_text(json.dumps(answer_bundle((TMP / "llm_prompts.md").read_text(encoding="utf-8"),
                                            weak=[{"bullet": PRICING, "problem": "duties",
                                                   "rewrite": REWRITE}])), encoding="utf-8")
p = cli(*base, "--answers", str(answers), "--auto-accept")
check("3: --answers alone (no --provider) runs copy-paste mode to the end", p.returncode == 0,
      p.stdout[-300:] + p.stderr[-300:])
final = out_txt.read_text(encoding="utf-8") if out_txt.exists() else ""
check("1: CLI --auto-accept keeps the original wording for the manager rewrite",
      PRICING in final and "Owned pricing strategy" not in final, final)
held = TMP / "pricing_out.proposals.json"
check("1: the held manager rewrite is written for explicit review",
      held.exists() and json.loads(held.read_text())[0]["needs_review"] is True, p.stdout[-400:])
check("1: the CLI says why it was held", "need your explicit yes" in " ".join(p.stdout.split()), p.stdout[-500:])
props = json.loads(held.read_text()) if held.exists() else []
for x in props:
    x["accepted"] = True
held.write_text(json.dumps(props), encoding="utf-8")
p = cli(*base, "--answers", str(answers), "--apply-proposals", str(held))
final = out_txt.read_text(encoding="utf-8") if out_txt.exists() else ""
check("1: the explicit yes via --apply-proposals applies it", p.returncode == 0 and "Owned pricing strategy" in final,
      p.stdout[-300:] + p.stderr[-300:])
p = cli(*base, "--provider", "openai", "--answers", str(answers))
check("3: --answers with a non-manual provider is refused", p.returncode == 2 and "--answers" in p.stderr,
      p.stderr[-200:])

guide = (ROOT / "AI_GUIDE.md").read_text(encoding="utf-8")
for line in guide.splitlines():
    if "python cli.py" in line and "--apply-proposals" in line:
        check("AI_GUIDE: the --apply-proposals example keeps --provider manual", "--provider manual" in line, line)

TD.cleanup()
if _OLD_HOME is None:
    os.environ.pop("ATS_HOME", None)
else:
    os.environ["ATS_HOME"] = _OLD_HOME
print(f"\n{'=' * 60}\nReview-fix tests: {passed} passed, {failed} failed")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all_checks_passed():
    """pytest entry point: the checks above run at import (collection) time."""
    assert failed == 0, f"{failed} check(s) failed — run this file directly for details"
