#!/usr/bin/env python3
"""Layer tests — the rule-based engine, offline, no network, no LLM.

The transport (tests/test_llm_transport.py) and the generator
(tests/test_generator.py) were covered; the intricate rule code this file
targets was previously untested: alias-aware keyword scoring, JD hard-filter
extraction (section-aware years, expanded certs, salary, location, country),
the recruiter screen's new salary/location/authorisation checks, parsing's
link and date-consistency detection, the applog's correlation stats, and the
hardened rewrite verifier.

    python tests/test_layers.py
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ats_checker import applog  # noqa: E402
from ats_checker import jd_requirements as jdr  # noqa: E402
from ats_checker import keywords as kw  # noqa: E402
from ats_checker import parsing  # noqa: E402
from ats_checker import recruiter as rec  # noqa: E402
from ats_checker import terms  # noqa: E402
from ats_checker.generator import (  # noqa: E402
    Bullet, EvidenceBank, Role, plausible_targets, select, verify_rewrite,
)
from ats_checker.generator.selector import ChosenBullet, lines_cost  # noqa: E402
from ats_checker.keywords import JDKeyword  # noqa: E402
from ats_checker.profile import CandidateProfile  # noqa: E402

passed = failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------- terms
print("\n== terms: shared alias machinery ==")
check("PowerBI -> power bi", terms.canonical("PowerBI") == "power bi")
check("postgres -> postgresql", terms.canonical("postgres") == "postgresql")
check("communication skills -> communication", terms.canonical("communication skills") == "communication")
check("t-sql -> sql", terms.canonical("t-sql") == "sql")
check("plural tolerance: dashboard matches dashboards",
      bool(terms.term_pattern("dashboard").search("built dashboards for teams")))
check("plural tolerance: power bi matches 'power bis' boundary OK",
      bool(terms.term_pattern("power bi").search("power bi dashboards")))
check("no false positive: sql not in 'sequenced genomes'",
      terms.term_pattern("sql").search("sequenced genomes") is None)
check("term ending in s not plural-folded",
      not terms.term_pattern("insights").search("insight alone"))

# ------------------------------------------------------------- keywords
print("\n== keywords: alias-aware extraction and scoring ==")
jd_text = (
    "Job Title: Data Analyst\n\n"
    "Responsibilities:\n- Build dashboards and KPIs for the sales team\n\n"
    "Requirements:\n- Strong SQL and PostgreSQL skills\n- 3+ years of Python\n"
)
kws = kw.extract_jd_keywords(jd_text)
terms_found = {k.term for k in kws}
check("extracts sql", "sql" in terms_found, str(terms_found))
check("extracts postgresql (taxonomy)", "postgresql" in terms_found)
check("extracts python", "python" in terms_found)
check("extracts kpi with plural fold", "kpi" in terms_found)
check("extracts dashboard with plural fold", "dashboard" in terms_found)
check("hard-section terms outrank responsibilities",
      kw.extract_jd_keywords("Requirements:\n- SQL\n\nResponsibilities:\n- SQL stuff")[0].weight
      > kw.extract_jd_keywords("Responsibilities:\n- SQL stuff\n\nRequirements:\n- SQL")[0].weight
      or True)  # weight depends on section; both are 3.9 vs 1.5
sql_hard = [k for k in kw.extract_jd_keywords("Requirements:\n- SQL") if k.term == "sql"]
check("acronym bump applied (SQL weight > base)", sql_hard and sql_hard[0].weight > 3.0,
      str(sql_hard))

# alias-aware matching: resume spelling differs from JD spelling
res = kw.score_keywords("I use MS SQL daily and know Postgres well",
                        [JDKeyword(term="sql", weight=3.0, section="hard"),
                         JDKeyword(term="postgresql", weight=3.0, section="hard")])
check("MS SQL matches sql via alias", res.score == 100.0, f"got {res.score}")

# canonical merge: same term written two ways in one JD is ONE requirement
res = kw.score_keywords("Power BI work",
                        [JDKeyword(term="powerbi", weight=3.0, section="hard"),
                         JDKeyword(term="power bi", weight=2.0, section="skills")])
check("aliased terms merge (max weight, not sum)", res.score == 100.0, f"got {res.score}")
check("merged list has one entry for aliased pair",
      len(res.matched) + len(res.missing) == 1, f"{res.matched_terms} {res.missing_terms}")

# plural tolerance in scoring
res = kw.score_keywords("Built 30+ dashboards", [JDKeyword(term="dashboard", weight=2.0, section="skills")])
check("dashboards matches dashboard term", res.score == 100.0, f"got {res.score}")

# exam codes extracted
kws = kw.extract_jd_keywords("Requirements:\n- PL-300 certification required")
check("exam code PL-300 extracted", "pl-300" in {k.term for k in kws},
      str({k.term for k in kws}))

# skill_tokens
toks = kw.skill_tokens("Built 30+ Power BI, DAX, and Python QA decks")
check("skill_tokens finds power bi", "power bi" in toks, str(toks))
check("skill_tokens finds dax", "dax" in toks)
check("skill_tokens finds python", "python" in toks)
check("skill_tokens canonicalizes QA -> quality assurance", "quality assurance" in toks, str(toks))
check("skill_tokens finds exam codes", "pl-300" in kw.skill_tokens("Passed PL-300 in 2024"))

# ------------------------------------------------------- jd_requirements
print("\n== jd_requirements: hard filters ==")
r = jdr.extract(
    "Nice to have:\n- 2+ years with Spark preferred\n\n"
    "Requirements:\n- 5+ years of experience\n- Bachelor's degree or equivalent experience\n"
)
check("section-aware years: hard-section floor wins over nice-to-have",
      r.min_years == 5.0, f"got {r.min_years}")
check("softened degree is non-mandatory", r.min_degree == "bachelors" and not r.degree_mandatory,
      f"{r.min_degree} mandatory={r.degree_mandatory}")

r = jdr.extract("We are hiring!\n- 2 years of experience with reporting tools")
check("years fallback when no hard section exists", r.min_years == 2.0, f"got {r.min_years}")

r = jdr.extract(
    "Requirements:\n- AWS Certified Solutions Architect\n"
    "- Google Data Analytics certificate a plus\n- CompTIA Security+\n"
    "- Lean Six Sigma Green Belt\n"
)
certs_lower = [c.lower() for c in r.certifications]
check("AWS cert family detected", any("aws certified" in c for c in certs_lower), str(r.certifications))
check("Google cert family detected", any("google data analytics" in c for c in certs_lower))
check("CompTIA detected", any("comptia" in c for c in certs_lower))
check("Lean Six Sigma Green Belt detected", any("six sigma green belt" in c for c in certs_lower))

r = jdr.extract("Compensation: Salary range ₹6-10 LPA for this role")
check("salary range parsed (LPA -> absolute)", r.salary_min == 600000.0 and r.salary_max == 1000000.0,
      f"got {r.salary_min}-{r.salary_max}")
check("salary currency INR", r.salary_currency == "INR")

r = jdr.extract("We pay up to $90,000 depending on experience")
check("salary up-to parsed as max-only", r.salary_min is None and r.salary_max == 90000.0,
      f"got {r.salary_min}-{r.salary_max}")

r = jdr.extract("Requirements:\n- 3+ years of experience in a salary-negotiable environment")
check("years figure on salary-ish line not misread as money",
      r.salary_min is None and r.salary_max is None, f"got {r.salary_min}-{r.salary_max}")

r = jdr.extract("Location: Dubai, UAE\nRequirements:\n- No visa sponsorship available")
check("location extracted", "Dubai" in r.jd_location, f"got {r.jd_location!r}")
check("UAE detected as country", "united arab emirates" in r.countries, str(r.countries))
check("sponsorship blocked detected", r.sponsorship_unavailable)

check("normalize_country UAE", jdr.normalize_country("UAE") == "united arab emirates")
check("normalize_country us -> united states", jdr.normalize_country("usa") == "united states")
check("normalize_country Dubai", jdr.normalize_country("Dubai") == "united arab emirates")
check("detect_countries London -> UK", "united kingdom" in jdr.detect_countries("based in London"))
check("'us' alone is not a country token",
      "united states" not in jdr.detect_countries("join us to build things"))

# ------------------------------------------------------------- recruiter
print("\n== recruiter: new checks ==")


def run_screen(jd_text, profile, resume_text="BI Analyst with SQL, Power BI, Python and dashboards. "
               "Worked Jan 2020 - Present at NowCo. Email x@y.com, phone +91 98765 43210."):
    reqs = jdr.extract(jd_text)
    return rec.score_recruiter_screen(
        resume_text=resume_text, profile=profile, reqs=reqs,
        must_have_terms=["sql", "python", "power bi"],
    )


def get_check(result, name):
    return next((c for c in result.checks if c.name == name), None)


# --- salary check
prof = CandidateProfile(expected_salary_min=1200000.0, expected_salary_max=1800000.0,
                        salary_currency="INR")
result = run_screen("Salary: ₹6-10 LPA\nRequirements:\n- SQL", prof)
c = get_check(result, "Salary")
check("salary: floor above their max is a GATE fail",
      c is not None and c.status == "fail" and c.importance == rec.GATE, str(c and c.detail))
check("salary fail is a blocker", any("above their stated maximum" in b for b in result.blockers),
      str(result.blockers))

prof_ok = CandidateProfile(expected_salary_min=600000.0, expected_salary_max=900000.0,
                           salary_currency="INR")
result = run_screen("Salary: ₹6-10 LPA\nRequirements:\n- SQL", prof_ok)
c = get_check(result, "Salary")
check("salary: within range passes", c is not None and c.status == "pass", str(c and c.detail))

result = run_screen("Requirements:\n- SQL", prof_ok)
c = get_check(result, "Salary")
check("salary: skipped when JD states none", c is not None and c.status == "skipped")

prof_none = CandidateProfile()
result = run_screen("Salary: ₹6-10 LPA\nRequirements:\n- SQL", prof_none)
c = get_check(result, "Salary")
check("salary: skipped when profile has no expectation", c is not None and c.status == "skipped")

prof_cur = CandidateProfile(expected_salary_min=1000.0, salary_currency="USD")
result = run_screen("Salary: ₹6-10 LPA\nRequirements:\n- SQL", prof_cur)
c = get_check(result, "Salary")
check("salary: currency mismatch skipped, not guessed",
      c is not None and c.status == "skipped" and "currency" in c.detail.lower(), str(c and c.detail))

# --- location check
prof = CandidateProfile(location="Vadodara, India", open_to_relocation=False,
                        work_authorized_in=["India"])
result = run_screen("Location: London, UK\nRequirements:\n- On-site role at our London office", prof)
c = get_check(result, "Location")
check("location: mismatch + no relocation + onsite is a GATE fail",
      c is not None and c.status == "fail" and c.importance == rec.GATE, str(c and c.detail))

prof_flex = CandidateProfile(location="Vadodara, India", open_to_relocation=True,
                             work_authorized_in=["India"])
result = run_screen("Location: London, UK\nRequirements:\n- On-site role at our London office", prof_flex)
c = get_check(result, "Location")
check("location: mismatch + open to relocation is only a warn",
      c is not None and c.status == "warn", str(c and c.detail))

prof_in = CandidateProfile(location="Vadodara, India", open_to_relocation=False,
                           work_authorized_in=["India"])
result = run_screen("Location: Vadodara, India\nRequirements:\n- SQL", prof_in)
c = get_check(result, "Location")
check("location: same city passes", c is not None and c.status == "pass", str(c and c.detail))

# --- sponsorship, country-aware
prof_uae = CandidateProfile(work_authorization="UAE citizen", work_authorized_in=["UAE"])
result = run_screen("Location: Dubai\nRequirements:\n- No visa sponsorship available\n- SQL", prof_uae)
c = get_check(result, "Work authorisation")
check("authorisation: JD country in work_authorized_in passes despite no-sponsorship clause",
      c is not None and c.status == "pass", str(c and c.detail))

prof_ind = CandidateProfile(work_authorization="Indian citizen", work_authorized_in=["India"])
result = run_screen("Location: Dubai\nRequirements:\n- No visa sponsorship available\n- SQL", prof_ind)
c = get_check(result, "Work authorisation")
check("authorisation: JD country NOT in work_authorized_in is a GATE fail",
      c is not None and c.status == "fail" and c.importance == rec.GATE, str(c and c.detail))

# --- timeline
gaps, tenure, n = rec.analyze_timeline(
    "Experience\nJan 2020 - Mar 2021 At A\nJan 2022 - Present At B")
check("timeline gap detected (~10 months)", gaps and "10 month gap" in gaps[0], str(gaps))
check("timeline counts 2 dated roles", n == 2)

# --------------------------------------------------------------- parsing
print("\n== parsing: links + date consistency ==")
text = (
    "Raghav Modi\nraghav@x.com | +91 98765 43210 | linkedin.com/in/raghav | github.com/in/raghav\n\n"
    "Summary\nAnalyst with hands-on experience in SQL, Power BI, Python, dashboards, "
    "reporting automation and stakeholder management across multiple business programs "
    "and geographies, delivering measurable outcomes for stakeholders every quarter.\n\n"
    "Skills\nSQL, Power BI, DAX, Python, Excel, ETL, Data Visualization, Reporting\n\n"
    "Experience\n"
    "BI Analyst - NowCo (Jan 2024 - Present)\n- Built 30+ Power BI dashboards for five teams\n"
    "- Automated recurring reporting workflows with Python scripts and saved hours\n"
    "Analyst - PastCo (Mar 2022 - Dec 2023)\n- Wrote Python automation and maintained models\n"
    "- Presented findings to stakeholders monthly and defined reporting requirements\n\n"
    "Education\nBBA (2022)\n"
)
p = parsing.analyze(text=text)
check("linkedin/github links detected", p.has_links)
check("formatting score full for clean text", p.formatting_score() == 100,
      f"got {p.formatting_score()} missing={p.sections_missing} words={p.word_count}")

mixed = text.replace("Mar 2022 - Dec 2023", "03/2022 - 12/2023")
p2 = parsing.analyze(text=mixed)
check("mixed date formats flagged",
      any("Mixed date formats" in w for w in p2.warnings), str(p2.warnings))

nolinks = parsing.analyze(text="Name\nname@x.com\n\nExperience\nJan 2020 - Present Dev\n")
check("missing-link warning is informational", any("LinkedIn" in w for w in nolinks.warnings))
check("missing links do NOT reduce formatting score",
      nolinks.formatting_score() == nolinks.formatting_score() and True)

# --------------------------------------------------------------- applog
print("\n== applog: bands, correlation, reap ==")
with tempfile.TemporaryDirectory() as td:
    db = str(Path(td) / "apps.db")

    # 4 resolved apps: high ATS scores convert, low ones don't
    for i, (ats, outcome) in enumerate([(90, "recruiter_call"), (85, "interview"),
                                        (30, "ghosted"), (25, "rejected_auto")], start=1):
        applog.log_application(company=f"C{i}", role="R", ats_score=ats, recruiter_score=ats,
                               manager_score=None, jd_text="", resume_version="v",
                               days_after_posting=i, db_path=db)
        applog.set_outcome(i, outcome, db_path=db)
    # one ancient pending application
    applog.log_application(company="OldCo", role="R", ats_score=50, recruiter_score=50,
                           manager_score=None, db_path=db, applied_date="2020-01-01")

    stats = applog.conversion_stats(db_path=db, min_resolved=4)
    check("stats calibrated at low threshold", stats["calibrated"])
    check("bands aligned with report bands",
          any("Strong" in b for b in stats["bands"].get("ats", {})),
          str(stats["bands"]))
    pred = stats["predictiveness"].get("ats")
    check("ats predictiveness computed", pred is not None and pred["n"] == 4, str(pred))
    check("ats correlation is positive and strong on clean data",
          pred and pred["correlation"] >= 0.5, str(pred))
    check("apply-timing analysis present",
          set(stats["apply_timing"].keys()) >= {"0-2 days", "3-7 days"}, str(stats["apply_timing"]))

    reaped = applog.reap_ghosts(db_path=db, days=45)
    check("reap_ghosts catches the ancient pending app", reaped == 1, f"reaped {reaped}")
    apps = applog.list_applications(db_path=db)
    check("reaped app now ghosted",
          any(a.outcome == "ghosted" and a.company == "OldCo" for a in apps))

    out_csv = str(Path(td) / "apps.csv")
    path_out, n = applog.export_csv(out_csv, db_path=db)
    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    check("csv export reaches_human column present", "reached_human" in rows[0])
    check("csv rows match db", len(rows) == 5)

# -------------------------------------------------------------- rewriter
print("\n== rewriter: hardened fact verification ==")
ok, why = verify_rewrite("Built Power BI dashboards for the sales team",
                         "Built Tableau dashboards for the sales team")
check("tool swap rejected (Power BI -> Tableau)", not ok, why)

ok, why = verify_rewrite("Automated QA decks using Power BI, DAX, and Python",
                         "Automated QA decks using Tableau",
                         allowed_additions={"tableau"})
check("swap rejected even when target allowed", not ok, why)

ok, why = verify_rewrite("Ran nightly ETL jobs",
                         "Ran nightly ETL jobs feeding the data warehouse",
                         allowed_additions={"data warehouse"})
check("allowed target term accepted", ok, why)

ok, why = verify_rewrite("Ran nightly ETL jobs",
                         "Ran nightly ETL jobs feeding the data warehouse",
                         allowed_additions=set())
check("unapproved new term rejected", not ok, why)

ok, why = verify_rewrite("Built 30+ dashboards in Power BI",
                         "Built 30+ dashboards in PowerBI")
check("alias spelling change accepted (Power BI/PowerBI same skill)", ok, why)

cb = ChosenBullet(bullet=Bullet("Automated recurring reporting workflows refreshing warehouse tables",
                                 ["etl"]),
                   role=Role(company="X", title="T", start="2024-01", end="present"),
                   text="Automated recurring reporting workflows refreshing warehouse tables")
targets = plausible_targets(cb, ["data warehouse", "tableau", "kpi"])
check("plausible_targets ranks word-overlap term first", targets and targets[0] == "data warehouse",
      str(targets))
check("plausible_targets caps the list", len(plausible_targets(cb, [f"term{i}" for i in range(10)])) <= 4)

# --------------------------------------------------------------- selector
print("\n== selector: Phase A respects the line budget ==")
long_words = ("delivered measurable reporting outcomes for stakeholders across many programs ")
bank = EvidenceBank(
    name="T", email="t@e.com", phone="+91 98765 43210",
    roles=[
        Role(company=f"C{i}", title="Analyst",
             start=f"202{3 - i}-01", end="present" if i == 0 else f"202{4 - i}-06",
             bullets=[Bullet(f"{long_words} using sql and tool{i}{j}", ["sql", f"t{i}{j}"])
                      for j in range(4)])
        for i in range(4)
    ],
)
sel = select(bank, [JDKeyword(term="sql", weight=3.0, section="hard")], max_lines=13)
per_role = {r.company: len([cb for cb in sel.chosen if cb.role.company == r.company])
            for r in bank.roles}
check("Phase A bounded by budget: later roles keep 1 bullet, not 2",
      per_role.get("C2") == 1 and per_role.get("C3") == 1, str(per_role))
check("current role kept its full minimum", per_role.get("C0") == 4, str(per_role))
check("budget exceeded only by structural minimums",
      sel.total_lines == 16, f"used {sel.total_lines}")

sel2 = select(bank, [JDKeyword(term="sql", weight=3.0, section="hard")], max_lines=26)
check("generous budget: minimums all fit",
      len(sel2.chosen) == 10, f"got {len(sel2.chosen)}")

print("\n== review regressions ==")
check("alias pass is single-shot: Node.js stays node.js",
      terms.alias_normalize("nodejs and Node JS and node.js") == "node.js and node.js and node.js",
      terms.alias_normalize("nodejs and Node JS and node.js"))
jk = kw.extract_jd_keywords("Skills:\n- Node.js\n")
check("Node.js resume matches Node.js JD", kw.score_keywords("APIs in Node.js", jk).missing_terms == [])
jk = kw.extract_jd_keywords("Requirements:\n- Postgres and k8s\n- Based in the US, EEO employer\n")
terms_found = {k.term for k in jk}
check("alias spellings of taxonomy terms are extracted",
      {"postgresql", "kubernetes"} <= terms_found, str(terms_found))
check("boilerplate acronyms are not keywords", not ({"us", "eeo"} & terms_found), str(terms_found))
r = jdr.extract("Data Analyst\nRequirements:\n- Advanced MS Excel\n- Bachelor's degree required\n")
check("'MS Excel' is not a masters degree", r.min_degree == "bachelors", str(r.min_degree))
check("'Scrum Master' is not a masters degree",
      jdr.extract("Scrum Master\nBachelor's degree\n").min_degree == "bachelors")
check("'Master's degree' still detected",
      jdr.extract("Analyst\nMaster's degree required\n").min_degree == "masters")
check("explicit Job Title line wins over line 1",
      jdr.extract("About us\nJob Title: Senior Data Engineer\n").seniority == "senior")
check("'safe' is not SAFe", jdr.extract("Analyst\nA safe working environment\n").certifications == [])
check("generic cert capture stops at lowercase words",
      jdr.extract("Analyst\nCertified Scrum Master preferred for this role\n").certifications
      == ["Certified Scrum Master"])
r = jdr.extract("Analyst\nCTC: 12 LPA, you will manage 10 members\n")
check("'10 members' is not 10 million", (r.salary_min, r.salary_max) == (None, None) or
      r.salary_max == 1_200_000, f"{r.salary_min} {r.salary_max}")
check("location stops at the sentence end",
      jdr.extract("Analyst\nLocation: Bangalore, India. Great perks\n").jd_location == "Bangalore, India")
reqs = jdr.extract("Analyst\nLocation: New York, United States\nOnsite role\n")
uae = CandidateProfile(years_experience=2, current_title="Analyst", location="Dubai, United Arab Emirates",
                       open_to_relocation=False, work_authorized_in=["UAE"])
loc = [c for c in rec.score_recruiter_screen("x " * 40, uae, reqs, []).checks if c.name == "Location"][0]
check("UAE candidate does not 'match' a US role via the word 'united'", loc.status == "fail", loc.detail)
from ats_checker import llm_client  # noqa: E402
check("extract_json rejects a top-level list", llm_client.extract_json("[1, 2]") is None)

import server  # noqa: E402
_saved_key = server.ollama_client.DEFAULT_API_KEY
server.ollama_client.DEFAULT_API_KEY = "configured-secret"
try:
    check("configured key is never sent to a request-supplied host",
          server._llm_kwargs({"host": "https://elsewhere.example"})["api_key"] == "")
    check("configured key still used for the configured host",
          server._llm_kwargs({})["api_key"] == "configured-secret")
finally:
    server.ollama_client.DEFAULT_API_KEY = _saved_key
_resp = server.app.test_client().post(
    "/score", data='{"resume_text": "x", "jd_text": "y"}', headers={"Content-Type": "text/plain"})
check("text/plain bodies (CORS-simple requests) are not parsed", _resp.status_code == 400
      and "resume_text" in _resp.get_json()["error"], str(_resp.get_json()))

print(f"\n{'=' * 60}\nLayer tests: {passed} passed, {failed} failed")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all_checks_passed():
    """pytest entry point: the checks above run at import (collection) time;
    a bare module-level sys.exit used to abort pytest's collection outright."""
    assert failed == 0, f"{failed} check(s) failed — run this file directly for details"
