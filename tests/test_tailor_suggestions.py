#!/usr/bin/env python3
"""tailor() never writes a manager-layer rewrite into the resume — offline, stubbed LLM.

The truth rule catches new numbers, tools and JD terms, but not an invented
outcome ("...that transformed pricing strategy"). optimize() holds manager
rewrites for the user's yes; tailor() has no review step, so it must only
SUGGEST them (TailorResult.suggestions / the "suggestions" list of POST
/tailor) and leave the bank's wording in the resume.

    python tests/test_tailor_suggestions.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TD = tempfile.TemporaryDirectory()
_OLD_HOME = os.environ.get("ATS_HOME")
os.environ["ATS_HOME"] = str(Path(TD.name) / "home")

from ats_checker import llm_client  # noqa: E402
from ats_checker import profile as profile_mod  # noqa: E402
from ats_checker.generator import tailor  # noqa: E402

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
MASTER = str(SAMPLES / "master_resume.yaml")
PROFILE = str(SAMPLES / "profile.yaml")
JD = (SAMPLES / "sample_jd.txt").read_text(encoding="utf-8")
ORIGINAL = "Prepared ad-hoc analysis decks for the category managers"
INVENTED = ORIGINAL + " that transformed pricing strategy"


def stub(system, user, **kw):
    """A manager layer that invents an outcome for one real resume bullet;
    every other LLM call answers with nothing to change."""
    if "hiring manager" in system:
        assert ORIGINAL in user.split("RESUME:", 1)[1], "fixture bullet not in the draft"
        return {"quantification": 60, "outcome_focus": 50, "evidence_backing": 60,
                "scope_match": 60, "domain_relevance": 60, "credibility": 60,
                "weak_bullets": [{"bullet": ORIGINAL, "problem": "vague", "rewrite": INVENTED}],
                "interview_risks": [], "manager_verdict": "ok"}, None
    if '"rewrites"' in system:
        return {"rewrites": []}, None
    return {"semantic_score": 50, "strengths": [], "gaps": [], "reworded_matches": [],
            "recommendation": ""}, None


real = llm_client.call_json
llm_client.call_json = stub
try:
    print("== tailor() ==")
    t = tailor(master_path=MASTER, jd_text=JD, profile=profile_mod.load_profile(PROFILE),
               offline=False, force=True)
    check("the invented outcome never reaches the resume", "transformed" not in t.resume_text,
          t.resume_text[:300])
    check("the bank's own wording stays", ORIGINAL in t.resume_text)
    check("the rewrite is offered as a suggestion that needs a yes",
          [s["rewrite"] for s in t.suggestions] == [INVENTED]
          and all(s["needs_review"] and s["source"] == "manager-rewrite" for s in t.suggestions),
          str(t.suggestions))
    check("no manager rewrite is listed as applied",
          not any(r["source"] == "manager-rewrite" for r in t.rewordings), str(t.rewordings))
    check("a note says the suggestions were not applied",
          any("NOT applied" in n for n in t.notes), str(t.notes))

    print("\n== POST /tailor ==")
    import server  # noqa: E402

    server.MASTER_PATH, server.PROFILE_PATH = MASTER, PROFILE
    r = server.app.test_client().post("/tailor", json={
        "jd_text": JD, "master_path": MASTER, "profile_path": PROFILE, "force": True})
    body = r.get_json() or {}
    check("/tailor answers 200", r.status_code == 200, f"{r.status_code} {str(body)[:200]}")
    check("/tailor: the invented outcome never reaches the resume",
          body.get("resume_text") and "transformed" not in body["resume_text"],
          str(body.get("resume_text"))[:300])
    check("/tailor: it comes back under 'suggestions'",
          [s.get("rewrite") for s in body.get("suggestions") or []] == [INVENTED],
          str(body.get("suggestions")))
finally:
    llm_client.call_json = real
    if _OLD_HOME is None:
        os.environ.pop("ATS_HOME", None)
    else:
        os.environ["ATS_HOME"] = _OLD_HOME

# a check that the stub really reached the manager pass (else every check above is vacuous)
check("the stub's rewrite passed the truth rule (so only the new rule keeps it out)",
      bool(re.search(r"transformed", " ".join(s["rewrite"] for s in t.suggestions))))

print(f"\n{'=' * 60}\nTailor suggestion tests: {passed} passed, {failed} failed")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all_checks_passed():
    """pytest entry point: the checks above run at import (collection) time."""
    assert failed == 0, f"{failed} check(s) failed — run this file directly for details"
