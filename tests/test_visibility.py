#!/usr/bin/env python3
"""Search Visibility (Layer 1) tests — offline, no network, no LLM.

Covers: JD title cleaning, the parse gate, recruiter-search construction
(title + requirement groups, OR only for same-family tools named on one JD
line), strongest-section keyword weighting, line-level softeners, and the
application-log migration for the new column.

    python tests/test_visibility.py
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ats_checker import applog  # noqa: E402
from ats_checker import jd_requirements as jdr  # noqa: E402
from ats_checker import keywords as kw  # noqa: E402
from ats_checker import parsing  # noqa: E402
from ats_checker import visibility as vis  # noqa: E402
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


SAMPLES = Path(__file__).resolve().parent.parent / "samples"

print("== JD title cleaning ==")
cases = [
    ("Job Title: Senior Business Intelligence Manager (Hybrid - Dubai)", "business intelligence manager"),
    ("Sr. Data Analyst - Payments Team", "data analyst"),
    ("Business Analyst | Bengaluru", "business analyst"),
    ("About us", ""),                                   # no role noun -> no title
    ("We are a fast-growing fintech changing how people pay online", ""),
]
for line, want in cases:
    got = jdr.clean_title(line)
    check(f"clean_title({line[:34]!r}) -> {want!r}", got == want, repr(got))
check("explicit 'Job Title:' line wins",
      jdr.extract("About us\nJob Title: Data Engineer\n").jd_title == "data engineer")

print("\n== keyword weighting: strongest section, bullets aren't headers ==")
jd = ("Responsibilities:\n- Write SQL queries\n- Strong SQL and Power BI skills\n"
      "Requirements:\n- Advanced SQL\n- PL-300 certification preferred\n- Python\n")
kws = {k.term: k for k in kw.extract_jd_keywords(jd)}
check("SQL takes its Requirements weight, not its first (Responsibilities) mention",
      kws["sql"].section == "hard", kws["sql"].section)
check("a bullet containing 'skills' is not a Skills header",
      kws["power bi"].section == "responsibilities", kws["power bi"].section)
check("'preferred' on a requirement line demotes it to nice-to-have",
      kws["pl-300"].section == "nice", kws["pl-300"].section)
check("'BI' inside 'Power BI' is not its own keyword",
      "bi" not in {k.term for k in kw.extract_jd_keywords("Skills:\n- Power BI dashboards\n")})

print("\n== recruiter searches ==")
jd = ("Job Title: Data Analyst\n\nRequirements:\n- SQL and Python\n"
      "- Power BI or Tableau\n- Snowflake\n- Excellent communication skills\n")
reqs = jdr.extract(jd)
keys = kw.extract_jd_keywords(jd)
resume = ("Data Analyst\nBuilt Tableau dashboards on SQL and Python pipelines.\n" * 3)
pr = parsing.analyze(text=resume)
res = vis.score_visibility(resume, jd, keys, reqs.jd_title, pr)
queries = [s.query for s in res.searches]
check("title search is first", queries and queries[0] == '"data analyst"', str(queries))
check("same-family tools named on one JD line are OR-ed",
      any('("power bi" OR tableau)' in q or '(tableau OR "power bi")' in q for q in queries), str(queries))
check("soft skills are never searched", not any("communication" in q for q in queries), str(queries))
check("Tableau satisfies the Power BI OR Tableau group",
      all("tableau" not in " ".join(s.missing) and "power bi" not in " ".join(s.missing)
          for s in res.searches), str([s.missing for s in res.searches]))
check("missing Snowflake is reported on the searches that need it",
      any("snowflake" in m for s in res.searches for m in s.missing))
check("score = share of searches matched",
      res.score == round(res.matched_count / len(res.searches) * 100, 1), str(res.score))

jd2 = "Job Title: Data Analyst\nRequirements:\n- Power BI\n- Tableau\n"
groups = vis._skill_groups(jd2, kw.extract_jd_keywords(jd2), "data analyst")
check("tools on DIFFERENT lines are both required (never assumed substitutable)",
      ["power bi"] in groups and ["tableau"] in groups, str(groups))

jd3 = "Job Title: Business Intelligence Analyst\nRequirements:\n- BI experience\n- SQL\n"
groups = vis._skill_groups(jd3, kw.extract_jd_keywords(jd3), "business intelligence analyst")
check("title initials ('BI') aren't a separate search term", ["bi"] not in groups, str(groups))

empty = vis.score_visibility("x " * 200, "Some text with no structure", [], "", parsing.analyze(text="x " * 200))
check("nothing searchable -> score None, not 0", empty.score is None and empty.notes, str(empty.score))

print("\n== parse gate ==")
safe, checks = vis.parse_gate(parsing.analyze(text="hello " * 20))
check("too little text fails the gate", not safe and any(c.name == "Text extraction" and c.status == "fail"
                                                          for c in checks))
good = (SAMPLES / "sample_resume.txt").read_text(encoding="utf-8")
check("sample resume passes the gate", vis.parse_gate(parsing.analyze(text=good))[0])

print("\n== full report wiring ==")
rep = run_full_check(resume_text=good, jd_text=(SAMPLES / "sample_jd.txt").read_text(encoding="utf-8"),
                     skip_semantic=True, skip_manager=True)
d = rep.to_dict()
check("scores expose search_visibility_pct", d["scores"]["search_visibility_pct"] == rep.visibility.score)
check("legacy ats_score key kept for old scripts", "ats_score" in d["scores"])
check("visibility_layer lists every search",
      len(d["visibility_layer"]["searches"]) == len(rep.visibility.searches) > 0)

print("\n== application log migration ==")
with tempfile.TemporaryDirectory() as tmp:
    db = str(Path(tmp) / "old.db")
    con = sqlite3.connect(db)
    con.execute(applog.SCHEMA.replace(",\n    visibility_score REAL", ""))  # pre-visibility schema
    con.execute("INSERT INTO applications (applied_date, company, role, ats_score) "
                "VALUES ('2026-01-01', 'Old', 'Role', 55)")
    con.commit()
    con.close()
    apps = applog.list_applications(db_path=db)
    check("old database gains the column and keeps its rows",
          len(apps) == 1 and apps[0].visibility_score is None and apps[0].ats_score == 55)
    new_id = applog.log_application("New", "Role", 60, 70, None, visibility_score=83.3, db_path=db)
    check("new rows store visibility_score",
          [a.visibility_score for a in applog.list_applications(db_path=db) if a.id == new_id] == [83.3])

print(f"\n{'=' * 60}\nVisibility tests: {passed} passed, {failed} failed")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all_checks_passed():
    """pytest entry point: the checks above run at import (collection) time."""
    assert failed == 0, f"{failed} check(s) failed — run this file directly for details"
