#!/usr/bin/env python3
"""External judgments: the two LLM layers fed from a JSON file instead of a
live LLM call (`score --judgments`). Offline, no network.

    python tests/test_layers_judgments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ats_checker import llm_client  # noqa: E402
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


def _no_network(*a, **k):
    raise AssertionError("LLM transport must not be called when judgments are supplied")


llm_client.call_json = _no_network

RESUME = ("Jane Doe\njane@example.com | +91 98765 43210 | linkedin.com/in/jane\n"
          "EXPERIENCE\nData Analyst — Acme | Jan 2023 – Present\n"
          "- Built SQL and Excel reports for 12 stakeholders\n"
          "EDUCATION\nB.Com — Some College | 2019 – 2022\n")
JD = "Requirements:\n- SQL\n- Excel\n- Stakeholder communication\n"
J = {
    "semantic": {"semantic_score": 80, "strengths": ["sql"], "gaps": [],
                 "reworded_matches": [], "recommendation": "ok"},
    "manager": {"quantification": 50, "outcome_focus": 50, "evidence_backing": 50,
                "scope_match": 50, "domain_relevance": 50, "credibility": 100,
                "weak_bullets": [], "interview_risks": [], "manager_verdict": "ok"},
}

print("== external judgments ==")
rep = run_full_check(resume_text=RESUME, jd_text=JD, judgments=J)
check("semantic layer uses supplied score", rep.semantic_result.semantic_score == 80,
      str(rep.semantic_result.semantic_score))
check("semantic layer marked external", rep.semantic_result.model == "external")
check("manager weighted exactly like the rubric (50*0.9 + 100*0.1 = 55)",
      rep.manager_score == 55, str(rep.manager_score))
check("ATS uses LLM weights when semantic supplied", rep.ats_weights.get("semantic", 0) > 0,
      str(rep.ats_weights))

rep2 = run_full_check(resume_text=RESUME, jd_text=JD, judgments={"manager": J["manager"]},
                      skip_semantic=True)
check("partial judgments: semantic skipped, manager still scored",
      (not rep2.semantic_result.available) and rep2.manager_score == 55)

print(f"\n{'=' * 60}\nJudgment tests: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
