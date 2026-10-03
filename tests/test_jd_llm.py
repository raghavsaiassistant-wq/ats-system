#!/usr/bin/env python3
"""LLM JD extraction tests — offline: the LLM call is replaced by canned
answers, so these test the parts that must hold whatever the model says:
quote verification, value sanity checks, the merge with the rules, the
fallback, the cache and the learned-terms file.

    python tests/test_jd_llm.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.TemporaryDirectory()
os.environ["ATS_HOME"] = _TMP.name           # cache + learned terms go to a temp dir
os.environ.pop("ATS_LEARNED_TERMS", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ats_checker import evaluation as ev  # noqa: E402
from ats_checker import jd_llm  # noqa: E402
from ats_checker import learned  # noqa: E402
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


JD = """Job Title: Senior Quality Engineer (Hybrid - Pune)

Responsibilities:
- Own GD&T reviews on machined parts with the design team
- Run PFMEA workshops for new product launches

Requirements:
- 5+ years in quality engineering for automotive suppliers
- Bachelor's degree in Mechanical Engineering
- Hands-on SolidWorks and Minitab
- Six Sigma Green Belt preferred

Location: Pune, India - Hybrid, 3 days in office
CTC: 18-24 LPA
"""

GOOD = {
    "title": {"value": "quality engineer", "quote": "Job Title: Senior Quality Engineer"},
    "seniority": {"value": "senior", "quote": "Senior Quality Engineer"},
    "min_years": {"value": 5, "mandatory": True, "quote": "5+ years in quality engineering"},
    "degree": {"value": "bachelors", "mandatory": True,
               "quote": "Bachelor's degree in Mechanical Engineering"},
    "certifications": [{"name": "Six Sigma Green Belt", "quote": "Six Sigma Green Belt preferred"}],
    "work_modes": [{"value": "hybrid", "quote": "Hybrid, 3 days in office"}],
    "sponsorship_unavailable": {"value": False, "quote": ""},
    "salary": {"min": 1800000, "max": 2400000, "currency": "INR", "quote": "CTC: 18-24 LPA"},
    "countries": [{"value": "india", "quote": "Location: Pune, India"}],
    "skills": [
        # bullet marker dropped and spacing changed: still verifies
        {"term": "GD&T", "required": True, "quote": "Own GD&T  reviews on machined parts"},
        {"term": "PFMEA", "required": True, "quote": "Run PFMEA workshops for new product launches"},
        {"term": "SolidWorks", "required": True, "quote": "- Hands-on SolidWorks and Minitab"},
        {"term": "Minitab", "required": True, "quote": "Hands-on SolidWorks and Minitab"},
        # hallucinations and bad items
        {"term": "Kubernetes", "required": True, "quote": "Kubernetes in production"},
        {"term": "AutoCAD", "required": True, "quote": "Hands-on SolidWorks and Minitab"},
        {"term": "Python", "required": True, "quote": JD},               # whole-JD quote
        {"term": "the", "required": True, "quote": "the design team"},
    ],
}

print("== verification: quotes must be in the JD, values in their quotes ==")
v = jd_llm.verify(GOOD, JD)
check("grounded skills kept", {"gd&t", "pfmea", "solidworks", "minitab"} <= set(v.skills),
      str(v.skills.keys()))
reasons = {(r.field, str(r.value)): r.reason for r in v.rejected}
check("skill quoted from a line that isn't in the JD is dropped",
      reasons.get(("skills", "kubernetes")) == "quote not found in the JD", str(reasons))
check("real quote that doesn't contain the skill is dropped",
      reasons.get(("skills", "autocad")) == "term not in quote", str(reasons))
check("quoting the whole JD can't vouch for a skill",
      "python" not in v.skills and "longer than" in reasons.get(("skills", "python"), ""))
check("boilerplate words are never skills", "the" not in v.skills)
check("title verified and cleaned", v.title == "quality engineer", repr(v.title))
check("years verified", v.min_years == 5 and v.years_mandatory)
check("degree verified", v.min_degree == "bachelors" and v.degree_mandatory)
check("salary verified", v.salary == [1800000.0, 2400000.0, "INR"], str(v.salary))
check("work mode verified", v.work_modes == ["hybrid"])
check("country verified", v.countries == ["india"])
check("cert verified", v.certifications == ["Six Sigma Green Belt"])

BAD = {
    "title": {"value": "data scientist", "quote": "Senior Quality Engineer"},
    "min_years": {"value": 7, "quote": "5+ years in quality engineering"},
    "degree": {"value": "masters", "quote": "Hands-on SolidWorks and Minitab"},
    "work_modes": [{"value": "remote", "quote": "Hybrid, 3 days in office"},
                   {"value": "flexible", "quote": "Hybrid"}],
    "salary": {"min": 2400000, "max": 1800000, "currency": "INR", "quote": "CTC: 18-24 LPA"},
    "countries": [{"value": "remote", "quote": "Hybrid"},
                  {"value": "germany", "quote": "Location: Pune, India"}],
    "sponsorship_unavailable": {"value": True, "quote": "Bachelor's degree in Mechanical Engineering"},
    "skills": "not a list",
}
v = jd_llm.verify(BAD, JD)
fields = {r.field for r in v.rejected}
check("title whose words aren't in the quote is dropped", v.title is None)
check("years number not in quote is dropped", v.min_years is None)
check("degree from a non-degree line is dropped", v.min_degree is None)
check("work mode contradicting its quote / unknown mode dropped", v.work_modes == [])
check("salary min > max dropped", v.salary is None)
check("'remote' is not a country; country not in quote dropped", v.countries == [])
check("sponsorship claim needs sponsorship wording", v.sponsorship_unavailable is None)
check("malformed sections don't crash", v.skills == {})
check("every drop is listed with a reason",
      {"title", "min_years", "min_degree", "work_modes", "salary", "countries",
       "sponsorship_unavailable"} <= fields, str(fields))
check("a title whose words are aliased in the quote still verifies ('Data Engineer')",
      jd_llm.verify({"title": {"value": "data engineer", "quote": "Data Engineer"}},
                    "Data Engineer\nRemote").title == "data engineer")
check("garbage answer verifies to nothing", jd_llm.verify({"x": 1}, JD).skills == {})
check("years over the sanity cap dropped",
      jd_llm.verify({"min_years": {"value": 50, "quote": "50 years"}}, "50 years").min_years is None)

print("\n== merge with the rules ==")
calls = []


def fake_call(answer):
    def _call(system, user, **kw):
        calls.append(user)
        return (answer, None) if answer is not None else (None, "HTTP 401: bad key")
    return _call


orig = llm_client.call_json
try:
    llm_client.call_json = fake_call(GOOD)
    hy = jd_llm.extract_hybrid(JD)
    terms = {k.term: k.section for k in hy.keywords}
    check("source is llm+rules", hy.source == "llm+rules" and hy.error is None)
    check("a skill the taxonomy lacks (GD&T) reaches the keywords as a requirement",
          terms.get("gd&t") == "hard", str(terms))
    check("hallucinated skill never reaches the keywords", "kubernetes" not in terms)
    check("rules' structured fields kept where they agree",
          hy.reqs.min_years == 5 and hy.reqs.salary_currency == "INR")
    check("LLM-sourced values cite their JD line",
          "5+ years in quality engineering" in hy.reqs.cite("min_years"), hy.reqs.cite("min_years"))
    check("unknown verified skills recorded as learned candidates (not auto-added)",
          "gd&t" in hy.learned and "gd&t" in [t for t, _c, _e in learned.candidates()]
          and "gd&t" not in learned.approved_terms(), str(hy.learned))

    n = len(calls)
    hy2 = jd_llm.extract_hybrid(JD)
    check("same JD again comes from the cache (no second LLM call)",
          len(calls) == n and hy2.from_cache, f"{len(calls)} calls")
    jd_llm.extract_hybrid(JD + "\nApply by Friday.")
    check("a different JD is a cache miss", len(calls) == n + 1)
    jd_llm.extract_hybrid(JD, use_cache=False)
    check("use_cache=False asks again", len(calls) == n + 2)

    # the LLM says SolidWorks is only preferred while the rules filed it under Requirements
    pref = dict(GOOD, skills=[{"term": "SolidWorks", "required": False,
                               "quote": "Hands-on SolidWorks and Minitab"}])
    llm_client.call_json = fake_call(pref)
    hy = jd_llm.extract_hybrid(JD, use_cache=False)
    check("a section disagreement is recorded, not silent",
          any(c.field == "skill:solidworks" for c in hy.conflicts), str(hy.conflicts))

    # a term only the LLM found, and only in a duty line, is not added
    duty = dict(GOOD, skills=[{"term": "machined parts", "required": False,
                               "quote": "Own GD&T reviews on machined parts with the design team"}])
    llm_client.call_json = fake_call(duty)
    hy = jd_llm.extract_hybrid(JD, use_cache=False)
    check("an LLM-only term from a duty line is left out of the keywords",
          "machined parts" not in {k.term for k in hy.keywords})

    # an LLM that says nothing about years leaves the rules' answer in place
    llm_client.call_json = fake_call({"skills": []})
    hy = jd_llm.extract_hybrid(JD, use_cache=False)
    check("rules fill every field the LLM left empty",
          hy.reqs.min_years == 5 and hy.reqs.min_degree == "bachelors" and hy.reqs.jd_title)

    llm_client.call_json = fake_call(None)
    hy = jd_llm.extract_hybrid(JD, use_cache=False)
    check("LLM down -> rules only, and the report says why",
          hy.source == "rules" and "401" in (hy.error or ""), str(hy.error))

    print("\n== scorer wiring ==")
    resume = ("Quality Engineer\nexperienced@example.com | +91 90000 00000\n"
              "Experience\n- Led GD&T reviews and PFMEA on machined parts in SolidWorks\n"
              "Education\nB.E. Mechanical\nSkills\nMinitab, SolidWorks, GD&T\n") * 4
    llm_client.call_json = fake_call(GOOD)
    rep = run_full_check(resume_text=resume, jd_text=JD, skip_semantic=True, skip_manager=True,
                         jd_extractor="llm")
    d = rep.to_dict()
    check("report records how the JD was read", d["jd_extraction"]["source"] == "llm+rules")
    check("report lists dropped LLM items", len(d["jd_extraction"]["rejected"]) >= 3)
    check("searches can use LLM-found skills",
          any("gd&t" in s["query"] for s in d["visibility_layer"]["searches"]),
          str([s["query"] for s in d["visibility_layer"]["searches"]]))
    rules_rep = run_full_check(resume_text=resume, jd_text=JD, skip_semantic=True, skip_manager=True)
    check("default stays rules-only", rules_rep.to_dict()["jd_extraction"] == {"source": "rules"})
    llm_client.call_json = fake_call(None)
    rep = run_full_check(resume_text=resume, jd_text=JD + " ", skip_semantic=True,
                         skip_manager=True, jd_extractor="llm")
    check("LLM down during scoring -> a note, not a crash",
          any("rules only" in n for n in rep.notes), str(rep.notes))

    print("\n== evaluator: a well-behaved LLM can only add recall ==")
    items = ev.load_corpus()[:12]

    def oracle(system, user, **kw):
        """Answers with the corpus labels, quoting a real JD line for each."""
        jd = user.split("\n\n", 1)[1]
        item = next(i for i in items if i.jd == jd)
        lines = [ln.strip() for ln in jd.splitlines() if ln.strip()]
        skills = []
        for t in item.expected.get("required_terms") or []:
            line = next((ln for ln in lines if jd_llm._term_in_quote(t, ln)), None)
            if line:
                skills.append({"term": t, "required": True, "quote": line[:200]})
        return {"skills": skills}, None

    llm_client.call_json = oracle
    rules = ev.evaluate(items, ev.rule_extractor).metrics()
    hybrid = ev.evaluate(items, ev.make_llm_extractor("hybrid", use_cache=False)).metrics()
    check("hybrid required-skill recall >= rules",
          hybrid["terms_required.recall"] >= rules["terms_required.recall"],
          f"{hybrid['terms_required.recall']} vs {rules['terms_required.recall']}")
    check("hybrid 'found as requirement' recall >= rules",
          hybrid["terms_required_hard.recall"] >= rules["terms_required_hard.recall"],
          f"{hybrid['terms_required_hard.recall']} vs {rules['terms_required_hard.recall']}")
    check("structured fields unchanged when the LLM is silent on them",
          all(hybrid[k] == rules[k] for k in rules if k.endswith(".accuracy")))
    llm_client.call_json = fake_call(None)
    try:
        ev.evaluate(items[:1], ev.make_llm_extractor("llm", use_cache=False))
        check("evaluating an unreachable LLM fails loudly", False)
    except RuntimeError:
        check("evaluating an unreachable LLM fails loudly (never scores the rules instead)", True)
finally:
    llm_client.call_json = orig

print("\n== learned terms ==")
check("promote moves a candidate into approved", learned.promote("gd&t")
      and "gd&t" in learned.approved_terms() and "gd&t" not in [t for t, _c, _e in learned.candidates()])
check("promoting twice is a no-op", not learned.promote("GD&T"))
os.environ["ATS_LEARNED_TERMS"] = "off"
check("ATS_LEARNED_TERMS=off ignores the file", learned.approved_terms() == set())
os.environ.pop("ATS_LEARNED_TERMS")
check("reject of an unknown candidate reports False", not learned.reject("nonexistent"))

print(f"\n{'=' * 60}\nJD LLM tests: {passed} passed, {failed} failed")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all_checks_passed():
    """pytest entry point: the checks above run at import (collection) time."""
    assert failed == 0, f"{failed} check(s) failed — run this file directly for details"
