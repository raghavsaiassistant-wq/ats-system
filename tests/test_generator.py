#!/usr/bin/env python3
"""Generator tests — offline, no network, no LLM.

Covers the four load-bearing pieces of the tailor:
  - alias canonicalization (no accidental keyword stuffing)
  - greedy marginal-coverage selection (duplicate coverage is waste)
  - deterministic assembly (formatting score must be 100 BY CONSTRUCTION)
  - the fact-preservation verifier (the gate that keeps the LLM honest)
plus end-to-end runs: offline tailor on the samples, and the no-apply gate.

    python tests/test_generator.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ats_checker import profile as profile_mod  # noqa: E402
from ats_checker.generator import (  # noqa: E402
    EvidenceBank, Role, Bullet, Education,
    alias_normalize, assemble, canonical, select, tailor, text_covers,
    verify_rewrite,
)
from ats_checker.keywords import JDKeyword  # noqa: E402
from ats_checker.parsing import analyze  # noqa: E402
from ats_checker.recruiter import extract_date_ranges  # noqa: E402

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


def jd(terms_sections):
    """Build a fake JDKeyword list: [(term, weight)] -> [JDKeyword]."""
    return [JDKeyword(term=t, weight=w, section="hard") for t, w in terms_sections]


def small_bank() -> EvidenceBank:
    return EvidenceBank(
        name="Test Person", email="test@example.com", phone="+91 98765 43210",
        location="Testville", headline="BI Analyst",
        roles=[
            Role(company="NowCo", title="BI Analyst", start="2024-01", end="present", bullets=[
                Bullet("Built 30+ Power BI dashboards and SQL models for sales teams",
                      ["power bi", "sql", "dashboard"]),
                Bullet("Ran SQL queries and ETL jobs nightly to refresh warehouse tables",
                      ["sql", "etl", "data warehouse"]),
                Bullet("Presented findings to stakeholders in monthly review calls",
                      ["stakeholder management", "communication"]),
                Bullet("Partnered with business owners to define KPIs and reporting requirements",
                      ["kpi", "reporting", "stakeholder management"]),
            ]),
            Role(company="PastCo", title="Data Analyst", start="2022-01", end="2023-12", bullets=[
                Bullet("Wrote Python scripts to automate Excel reporting for three business units",
                      ["python", "excel", "automation"]),
                Bullet("Built Excel models and stakeholder-facing reports for quarterly business reviews",
                      ["excel", "reporting", "stakeholder management"]),
            ]),
        ],
        skills_extra=["git"],
        education=[Education(degree="BBA in Business Analytics", institution="Test College", year="2022")],
        certifications=["Test Cert"],
    )


print("\n== alias canonicalization ==")
check("PowerBI normalizes to 'power bi'", "power bi" == canonical("PowerBI"),
      f"got {canonical('PowerBI')!r}")
check("postgresql normalizes to 'postgresql'", "postgresql" == canonical("postgres"),
      f"got {canonical('postgres')!r}")
check("alias_normalize rewrites in-text variants",
      "power bi" in alias_normalize("PowerBI dashboards and Postgres tables").lower()
      and "postgresql" in alias_normalize("PowerBI dashboards and Postgres tables").lower(),
      alias_normalize("PowerBI dashboards and Postgres tables"))

print("\n== selection: greedy marginal coverage ==")
bank = small_bank()
keywords = jd([("sql", 3.0), ("python", 3.0), ("power bi", 3.0)])
sel = select(bank, keywords)
role_bullets = [(cb.role.company, cb.text) for cb in sel.chosen]
# bullet covering sql+power bi must be picked first for the current role
check("strongest-coverage bullet picked first",
      role_bullets and "30+ Power BI" in role_bullets[0][1], str(role_bullets))
check("current role got multiple bullets",
      sum(1 for c in sel.chosen if c.role.company == "NowCo") >= 2, str(role_bullets))
check("past role kept for timeline credibility",
      any(c.role.company == "PastCo" for c in sel.chosen), str(role_bullets))
check("covered terms include sql, power bi, python",
      {"sql", "power bi", "python"} <= sel.covered, str(sel.covered))
check("uncovered is a list of JDKeyword", all(isinstance(k, JDKeyword) for k in sel.uncovered))

print("\n== selection: capacity respected ==")
big_bank = small_bank()
# 10 bullets each covering sql plus one distinct JD term -> all have marginal
# value, so the greedy runs until the cap, and the cap must stop it.
big_bank.roles[0].bullets = [
    Bullet(f"Bullet {i} using sql and t{i}", ["sql", f"t{i}"]) for i in range(10)
]
sel_cap = select(big_bank, jd([(f"t{i}", 1.0) for i in range(10)] + [("sql", 3.0)]),
                max_current=5, max_other=3)
check("bullet cap for current role enforced",
      sum(1 for c in sel_cap.chosen if c.role.company == "NowCo") == 5,
      f"NowCo count = {sum(1 for c in sel_cap.chosen if c.role.company == 'NowCo')}")

print("\n== review fixes: current role uses real greedy, not static top-k ==")
# Six current-role bullets: three cover 'a' only, one each cover b/c/d.
# A STATIC sort by pre-pick marginal takes the three 'a' bullets first and
# misses 'd' entirely; true marginal greedy (recomputed between picks)
# takes one 'a' then b, c, d, then a second 'a' as reinforcement.
overlap_bank = EvidenceBank(
    name="T", email="t@e.com", phone="+91 98765 43210",
    roles=[Role(company="R", title="Analyst", start="2024-01", end="present", bullets=[
        Bullet("wrote a thing", ["a"]),
        Bullet("wrote another a thing again", ["a"]),
        Bullet("wrote a third thing too", ["a"]),
        Bullet("wrote b thing", ["b"]),
        Bullet("wrote c thing", ["c"]),
        Bullet("wrote d thing", ["d"]),
    ])],
)
sel_ov = select(overlap_bank, jd([("a", 3.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)]))
check("greedy not fooled by duplicate coverage: 'd' gets a slot",
      "d" in sel_ov.covered_by_bullets, str(sorted(sel_ov.covered_by_bullets)))
check("second 'a' kept as reinforcement (saturation > 0)",
      sum(1 for cb in sel_ov.chosen if "a" in cb.covered) >= 2,
      str([cb.text for cb in sel_ov.chosen]))
check("cap of 5 respected despite 6 bullets", len(sel_ov.chosen) == 5,
      str(len(sel_ov.chosen)))

print("\n== review fixes: weight merge is max, not sum ==")
from ats_checker.generator.selector import build_jd_map  # noqa: E402
w, _reps = build_jd_map([JDKeyword(term="powerbi", weight=3.0, section="hard"),
                         JDKeyword(term="power bi", weight=2.0, section="skills")])
check("aliased terms merge with max", w.get("power bi") == 3.0, f"got {w.get('power bi')}")

print("\n== review fixes: saturation keeps reinforcing evidence ==")
# Three earlier-role bullets all covering sql: binary marginal (old) stopped
# at 2; saturation gives the third a small positive value, so with budget
# left it is kept.
sat_bank = EvidenceBank(
    name="T", email="t@e.com", phone="+91 98765 43210",
    roles=[Role(company="Old", title="Dev", start="2020-01", end="2021-12", bullets=[
        Bullet("sql work one", ["sql"]),
        Bullet("sql work two", ["sql"]),
        Bullet("sql work three", ["sql"]),
    ])],
)
sel_sat = select(sat_bank, jd([("sql", 3.0)]), max_current=5, max_other=3, max_lines=26)
check("third sql mention still earns a slot (saturation, not binary)",
      len(sel_sat.chosen) == 3, f"got {len(sel_sat.chosen)}")

print("\n== review fixes: skills line is bounded ==")
fat_bank = small_bank()
fat_bank.skills_extra = [f"skill{i:02d}" for i in range(20)]
sel_fat = select(fat_bank, jd([("sql", 3.0), ("python", 3.0), ("power bi", 3.0), ("etl", 2.5)]))
text_fat = assemble(fat_bank, sel_fat, jd([("sql", 3.0), ("python", 3.0), ("power bi", 3.0), ("etl", 2.5)]))
skills_line = text_fat.split("Skills")[1].split("\n")[1]
check("skills line capped at 15 items", len(skills_line.split(",")) <= 15,
      f"got {len(skills_line.split(','))}: {skills_line}")
check("JD-covered skills survive the cap", "SQL" in skills_line and "Power BI" in skills_line)

print("\n== review fixes: summary only claims bullet-backed experience ==")
# 'tableau' is in skills_extra (a skill held) but no bullet demonstrates it:
# it may appear in the Skills line, never in the Summary.
claim_bank = EvidenceBank(
    name="T", email="t@e.com", phone="+91 98765 43210",
    headline="Analyst",
    roles=[Role(company="R", title="Analyst", start="2024-01", end="present", bullets=[
        Bullet("built sql models for the team", ["sql"]),
    ])],
    skills_extra=["tableau"],
)
sel_claim = select(claim_bank, jd([("sql", 3.0), ("tableau", 2.0)]))
text_claim = assemble(claim_bank, sel_claim, jd([("sql", 3.0), ("tableau", 2.0)]))
summary_line = text_claim.split("Summary")[1].split("\n")[1]
check("skills-only term NOT claimed as hands-on experience in Summary",
      "Tableau" not in summary_line, summary_line)
check("skills-only term still listed in Skills line", "Tableau" in text_claim.split("Skills")[1].split("\n")[1])
check("covered_by_bullets excludes skills-only term",
      "tableau" not in sel_claim.covered_by_bullets and "tableau" in sel_claim.covered,
      f"bullets={sel_claim.covered_by_bullets} all={sel_claim.covered}")

print("\n== review fixes: shared line budget ==")
# Three roles x six 30-word bullets: mandatory minimums run first, then the
# greedy must respect the global line budget, not per-role counts alone.
from ats_checker.generator.selector import lines_cost  # noqa: E402
long_words = "delivered measurable reporting outcomes for stakeholders across programs "
budget_bank = EvidenceBank(
    name="T", email="t@e.com", phone="+91 98765 43210",
    roles=[
        Role(company=f"C{i}", title="Analyst",
             start=f"202{3 - i}-01", end="present" if i == 0 else f"202{4 - i}-06",
             bullets=[Bullet(f"{long_words} using sql and tool{i}{j}", ["sql", f"t{i}{j}"])
                      for j in range(6)])
        for i in range(3)
    ],
)
sel_bud = select(budget_bank, jd([("sql", 3.0)] + [(f"t{i}{j}", 1.0) for i in range(3) for j in range(6)]),
                max_current=5, max_other=3, max_lines=26)
check("bullet lines within budget",
      sum(lines_cost(cb.bullet) for cb in sel_bud.chosen) <= 26,
      f"used {sum(lines_cost(cb.bullet) for cb in sel_bud.chosen)} lines (reported {sel_bud.total_lines})")
check("budget counter matches", sel_bud.total_lines == sum(lines_cost(cb.bullet) for cb in sel_bud.chosen),
      f"{sel_bud.total_lines}")
check("per-role caps still hold",
      all(len([cb for cb in sel_bud.chosen if cb.role.company == r.company]) <= (5 if r.is_current else 3)
          for r in budget_bank.roles))

print("\n== assembly: formatting by construction ==")
bank = small_bank()
keywords = jd([("sql", 3.0), ("python", 3.0), ("power bi", 3.0), ("etl", 2.5)])
sel = select(bank, keywords)
text = assemble(bank, sel, keywords)
parse = analyze(text=text)
check("formatting score is 100", parse.formatting_score() == 100,
      f"got {parse.formatting_score()}, missing={parse.sections_missing}, "
      f"email={parse.has_email}, phone={parse.has_phone}")
check("all four standard sections found",
      set(parse.sections_found) >= {"experience", "education", "skills", "summary"},
      str(parse.sections_found))
ranges = extract_date_ranges(text)
check("at least two dated roles parsed", len(ranges) >= 2, f"got {len(ranges)}")
check("dates in 'Mon YYYY - Mon YYYY' form", "Jan 2024 - Present" in text, text[:200])
check("email and phone present in header", "test@example.com" in text and "+91" in text)
check("skills ordered: JD-covered first", text.index("SQL") < text.index("Git"),
      text.split("Skills")[1].split("\n")[0] if "Skills" in text else "no skills line")

print("\n== verifier: fact preservation ==")
ok, why = verify_rewrite(
    "Built 30+ dashboards for 5 teams",
    "Built 30+ dashboards for 5 teams using Power BI")
check("kept numbers -> accepted", ok, why)
ok, why = verify_rewrite(
    "Built 30+ dashboards for 5 teams",
    "Built 40+ dashboards for 5 teams")
check("changed number -> rejected", not ok, why)
ok, why = verify_rewrite(
    "Built dashboards for 5 teams",
    "Built dashboards for 12 teams")
check("added number -> rejected", not ok, why)
ok, why = verify_rewrite(
    "Built dashboards across programs",
    "Built dashboards across [X] programs, cutting prep by [Y] hours")
check("[X] placeholders allowed", ok, why)
ok, why = verify_rewrite(
    "Automated monthly QA decks using Power BI, DAX, and Python",
    "Made decks")
check("truncating rewrite -> rejected", not ok, why)
ok, why = verify_rewrite("Same text", "Same text")
check("no-change -> rejected", not ok, why)

print("\n== text_covers guard ==")
jd_map = {"sql": 3.0, "python": 3.0}
check("covers sql", "sql" in text_covers("Wrote SQL queries nightly", jd_map))
check("PowerBI counts as power bi via alias",
      "power bi" in text_covers("PowerBI work", {"power bi": 3.0}),
      text_covers("PowerBI work", {"power bi": 3.0}))
check("no false positive", "sql" not in text_covers("sequenced genomes", jd_map))

print("\n== end-to-end: offline tailor on the samples ==")
result = tailor(
    master_path=str(SAMPLES / "master_resume.yaml"),
    jd_text=(SAMPLES / "sample_jd.txt").read_text(encoding="utf-8"),
    profile=profile_mod.load_profile(str(SAMPLES / "profile.yaml")),
    offline=True,
)
check("not blocked", not result.blocked, str(result.candidacy_blockers))
check("keyword match >= 50 on the samples", (result.final_keyword_pct or 0) >= 50,
      f"got {result.final_keyword_pct}")
check("formatting 100 in final report",
      result.report is not None and result.report.parse_result.formatting_score() == 100)
check("honest gaps reported (bank lacks snowflake)",
      any("snowflake" in g for g in result.honest_gaps), str(result.honest_gaps))
check("resume contains real skills", "Power BI" in result.resume_text and "DAX" in result.resume_text)
check("resume did not invent snowflake", "snowflake" not in result.resume_text.lower())
check("no-LLM run made no rewordings", len(result.rewordings) == 0)
check("baseline == final keyword (deterministic, offline)",
      result.baseline_keyword_pct == result.final_keyword_pct)

print("\n== no-apply gate ==")
sponsorship_jd = (
    "Job Title: BI Analyst\n\nRequirements:\n- No visa sponsorship available for this role\n"
    "- 1-3 years of experience\n- Strong SQL skills\n"
)
result_gate = tailor(
    master_path=str(SAMPLES / "master_resume.yaml"),
    jd_text=sponsorship_jd,
    profile=profile_mod.load_profile(str(SAMPLES / "profile.yaml")),
    offline=True,
)
check("gate blocked on sponsorship", result_gate.blocked, str(result_gate.candidacy_blockers))
check("blocker text cites the issue",
      any("sponsor" in b.lower() for b in result_gate.candidacy_blockers),
      str(result_gate.candidacy_blockers))
result_force = tailor(
    master_path=str(SAMPLES / "master_resume.yaml"),
    jd_text=sponsorship_jd,
    profile=profile_mod.load_profile(str(SAMPLES / "profile.yaml")),
    offline=True,
    force=True,
)
check("--force generates anyway", not result_force.blocked and result_force.resume_text.strip() != "")

print("\n== bank validation ==")
check("validate flags missing email",
      any("email" in p for p in EvidenceBank(name="x", roles=[Role("C", "T", bullets=[Bullet("b")])]).validate()))
try:
    EvidenceBank(name="x").validate()
    check("no roles raises", False, "no ValueError")
except ValueError:
    check("no roles raises", True)

print(f"\n{'=' * 50}\nGenerator tests: {passed} passed, {failed} failed")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all_checks_passed():
    """pytest entry point: the checks above run at import (collection) time;
    a bare module-level sys.exit used to abort pytest's collection outright."""
    assert failed == 0, f"{failed} check(s) failed — run this file directly for details"
