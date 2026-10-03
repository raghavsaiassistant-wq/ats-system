#!/usr/bin/env python3
"""Application-log tests (Phase 4) — offline, no network, no LLM.

Covers: Wilson intervals against known values, Fisher-z correlation
intervals and their wording (CI crossing zero, no "strong" below n=30), the
old-schema migration, the component-score round-trip from a FullReport,
per-component predictiveness, apply-timing bucket order, ghost reaping in
stats, company/role defaults from the JD, and the low-friction outcome
logging paths (CLI --last / company match, web UI buttons over JSON).

    python tests/test_applog.py
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ats_checker import applog  # noqa: E402
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


def close(a: float, b: float, tol: float = 0.001) -> bool:
    return abs(a - b) <= tol


SAMPLES = ROOT / "samples"

# --------------------------------------------------------------- Wilson
print("\n== Wilson 95% intervals ==")
lo, hi = applog.wilson_interval(0, 10)
check("0/10 -> 0.0-27.8%", close(lo, 0.0) and close(hi, 0.2775), f"{lo:.4f}-{hi:.4f}")
lo, hi = applog.wilson_interval(5, 10)
check("5/10 -> 23.7-76.3%", close(lo, 0.2366) and close(hi, 0.7634), f"{lo:.4f}-{hi:.4f}")
lo, hi = applog.wilson_interval(10, 10)
check("10/10 -> 72.2-100%", close(lo, 0.7225) and close(hi, 1.0), f"{lo:.4f}-{hi:.4f}")
lo, hi = applog.wilson_interval(1, 20)
check("1/20 -> 0.9-23.6%", close(lo, 0.0089) and close(hi, 0.2361), f"{lo:.4f}-{hi:.4f}")
check("n=0 is the uninformative interval", applog.wilson_interval(0, 0) == (0.0, 1.0))
check("interval always inside [0, 1]",
      all(0 <= a <= b <= 1 for k in range(0, 6) for a, b in [applog.wilson_interval(k, 5)]))
check("rate text carries n and CI",
      applog.format_rate(12, 40) == "12/40 = 30% (95% CI 18–45%)", applog.format_rate(12, 40))

# --------------------------------------------------------------- Fisher z
print("\n== Fisher-z correlation intervals ==")
lo, hi = applog.fisher_interval(0.5, 30)
check("r=0.5, n=30 -> 0.170 to 0.729", close(lo, 0.1704) and close(hi, 0.7292), f"{lo:.4f} {hi:.4f}")
lo, hi = applog.fisher_interval(0.0, 103)
check("r=0, n=103 -> symmetric +-0.194", close(lo, -0.1940) and close(hi, 0.1940), f"{lo:.4f} {hi:.4f}")
check("n<=3 gives the whole range", applog.fisher_interval(0.9, 3) == (-1.0, 1.0))
lo, hi = applog.fisher_interval(1.0, 10)
check("r=1 doesn't blow up", -1 <= lo <= hi <= 1)

# weak signal on few points: CI crosses zero
xs = [10, 20, 30, 40, 50, 60, 70, 80]
ys = [0, 1, 0, 0, 1, 0, 1, 1]
s = applog.correlation_summary(xs, ys)
check("crossing CI is flagged", s and not s["distinguishable_from_zero"] and s["ci_low"] < 0 < s["ci_high"],
      str(s))
check("crossing CI reads 'not distinguishable from zero'",
      s and s["strength"] == "not distinguishable from zero"
      and "not distinguishable from zero" in s["reading"], str(s))

# clean separation on n=12: clearly non-zero, but not graded "strong"
xs = [90, 88, 85, 82, 80, 78, 30, 28, 25, 22, 20, 18]
ys = [1] * 6 + [0] * 6
s = applog.correlation_summary(xs, ys)
check("n=12 clean data is distinguishable", s and s["distinguishable_from_zero"], str(s))
check("never 'strong' below n=30", s and "strong" not in s["strength"] and "n<30" in s["strength"], str(s))
check("reading shows r, CI and n", s and "95% CI" in s["reading"] and "n=12" in s["reading"], str(s))

xs = [90 - i for i in range(15)] + [30 - i for i in range(15)]
ys = [1] * 15 + [0] * 15
s = applog.correlation_summary(xs, ys)
check("n=30 clean data is graded strong", s and s["strength"] == "strong", str(s))
check("negative relationships say FEWER",
      "FEWER" in applog.correlation_summary(xs, [1 - y for y in ys])["reading"])
check("zero variance -> no correlation", applog.correlation_summary([5, 5, 5, 5], [0, 1, 0, 1]) is None)

# --------------------------------------------------------------- migration
print("\n== old-schema migration ==")
with tempfile.TemporaryDirectory() as td:
    db = str(Path(td) / "old.db")
    con = sqlite3.connect(db)
    old_schema = (applog.SCHEMA.replace(",\n    visibility_score REAL", "")
                  .replace(",\n    components TEXT", ""))
    check("test schema really is the old one",
          "components" not in old_schema and "visibility_score" not in old_schema)
    con.execute(old_schema)
    con.execute("INSERT INTO applications (applied_date, company, role, ats_score, outcome) "
                "VALUES ('2025-01-01', 'Old', 'Role', 55, 'ghosted')")
    con.commit()
    con.close()
    apps = applog.list_applications(db_path=db)
    check("old rows survive the migration",
          len(apps) == 1 and apps[0].ats_score == 55 and apps[0].components is None)
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(applications)")}
    check("both columns added", {"visibility_score", "components"} <= cols, str(cols))
    new_id = applog.log_application("New", "Role", 60, 70, None, db_path=db,
                                    components={"keyword_score": 42.0})
    got = [a.components for a in applog.list_applications(db_path=db) if a.id == new_id]
    check("migrated db stores components", got == [{"keyword_score": 42.0}], str(got))
    applog.list_applications(db_path=db)  # second connect: migrations are idempotent
    check("migration is idempotent", True)
    con = sqlite3.connect(db)
    con.execute("UPDATE applications SET components = 'not json' WHERE id = 1")
    con.commit()
    con.close()
    check("corrupt components read as None, not a crash",
          [a.components for a in applog.list_applications(db_path=db) if a.id == 1] == [None])

# --------------------------------------------------------------- components
print("\n== component round-trip from FullReport ==")
jd_text = (SAMPLES / "sample_jd.txt").read_text(encoding="utf-8")
report = run_full_check(resume_path=str(SAMPLES / "sample_resume.txt"), jd_text=jd_text,
                        skip_semantic=True, skip_manager=True)
comps = applog.components_from_report(report)
check("visibility searches stored",
      comps["visibility"]["searches_total"] == len(report.visibility.searches)
      and comps["visibility"]["searches_matched"] == report.visibility.matched_count, str(comps["visibility"]))
check("parse_safe stored", comps["visibility"]["parse_safe"] is report.visibility.parse_safe)
check("every recruiter check stored with its status",
      comps["recruiter_checks"] == {c.name: c.status for c in report.recruiter_result.checks})
check("keyword score stored", comps["keyword_score"] == report.keyword_result.score)
check("offline: no semantic / manager scores", comps["semantic_score"] is None
      and comps["manager_dimensions"] == {})
check("jd source stored", comps["jd_source"] == "rules")
check("components are plain JSON", json.loads(json.dumps(comps)) == comps)

with tempfile.TemporaryDirectory() as td:
    db = str(Path(td) / "apps.db")
    app_id = applog.log_application("Acme", "BI Analyst", report.ats_score, report.recruiter_score,
                                    report.manager_score, db_path=db, report=report,
                                    visibility_score=report.visibility.score)
    back = applog.list_applications(db_path=db)[0]
    check("components round-trip through SQLite", back.id == app_id and back.components == comps)
    flat = applog.numeric_components(back.components)
    check("flattened: searches %, parse gate, keyword, recruiter checks",
          "visibility: searches matched %" in flat and "visibility: parse gate passed" in flat
          and "keyword match" in flat and any(k.startswith("recruiter: ") for k in flat), str(flat))
    check("skipped recruiter checks are missing data, not zeros",
          all(f"recruiter: {c.name}" not in flat
              for c in report.recruiter_result.checks if c.status == "skipped"))
    out_csv = str(Path(td) / "apps.csv")
    applog.export_csv(out_csv, db_path=db)
    with open(out_csv, encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    check("CSV export carries components_json", json.loads(row["components_json"]) == comps)

# manager dimensions + semantic, as an LLM run would store them
flat = applog.numeric_components({"manager_dimensions": {"quantification": 70, "credibility": 40},
                                  "semantic_score": 66, "recruiter_checks": {"Salary": "warn"}})
check("manager dimensions and semantic flatten",
      flat == {"manager: quantification": 70.0, "manager: credibility": 40.0,
               "llm fit (semantic)": 66.0, "recruiter: Salary": 0.5}, str(flat))

# --------------------------------------------------------------- stats
print("\n== conversion stats: CIs, components, bucket order, reaping ==")
with tempfile.TemporaryDirectory() as td:
    db = str(Path(td) / "apps.db")
    # 24 resolved: keyword match tracks callbacks, the parse gate never varies,
    # Salary check is noise. Days are inserted out of time order on purpose.
    days_cycle = [20, 1, 9, 4]
    for i in range(24):
        good = i % 2 == 0
        comp = {
            "visibility": {"searches_matched": 5 if good else 1, "searches_total": 6, "parse_safe": True},
            "recruiter_checks": {"Salary": "pass" if i % 3 else "fail", "Location": "skipped"},
            "keyword_score": (80 if good else 30) + i % 5,
        }
        aid = applog.log_application(f"Co{i}", "Analyst", 50, 50 + (i % 7), None,
                                     visibility_score=83 if good else 17,
                                     days_after_posting=days_cycle[i % 4],
                                     db_path=db, components=comp)
        applog.set_outcome(aid, "recruiter_call" if good else "rejected_auto", db_path=db)
    applog.log_application("Ghost", "Analyst", 50, 50, None, applied_date="2020-01-01", db_path=db)

    st = applog.conversion_stats(db_path=db, min_resolved=20)
    check("calibrated at 24 resolved", st["calibrated"] and st["resolved"] == 24, str(st["resolved"]))
    check("stale pending NOT reaped by default, and it says so",
          st["reaped"] is None and "NOT reaped" in st["message"] and "1 pending" in st["message"],
          st["message"])
    check("overall rate has a CI", st["overall"]["rate"].startswith("12/24 = 50% (95% CI"),
          str(st["overall"]))
    all_buckets = [b for bands in st["bands"].values() for b in bands.values()] + list(st["apply_timing"].values())
    check("every band and timing bucket has n, ci_low, ci_high, rate",
          all_buckets and all({"n", "ci_low", "ci_high", "rate", "conversion_pct"} <= b.keys() and
                              b["ci_low"] <= b["conversion_pct"] <= b["ci_high"] for b in all_buckets))
    strong = st["bands"]["visibility"]["80-100 (Strong)"]
    lo, hi = applog.wilson_interval(12, 12)
    check("band CI matches wilson_interval",
          strong["ci_low"] == round(lo * 100, 1) and strong["ci_high"] == round(hi * 100, 1), str(strong))
    check("timing buckets come out in time order",
          list(st["apply_timing"]) == ["0-2 days", "3-7 days", "8-14 days", "15+ days"],
          str(list(st["apply_timing"])))
    check("bands come out high to low",
          list(st["bands"]["visibility"]) == ["80-100 (Strong)", "0-39 (Very weak)"],
          str(list(st["bands"]["visibility"])))
    vis = st["predictiveness"]["visibility"]
    check("layer correlation carries a Fisher CI", {"ci_low", "ci_high", "n", "reading"} <= vis.keys()
          and vis["ci_low"] <= vis["correlation"] <= vis["ci_high"], str(vis))
    check("n<30 layer correlation isn't called strong", "strong" not in vis["strength"], vis["strength"])

    comp_rows = st["component_predictiveness"]
    names = [c["component"] for c in comp_rows]
    check("one row per varying component",
          set(names) == {"visibility: searches matched %", "keyword match", "recruiter: Salary"}, str(names))
    check("constant / skipped components produce no row",
          "visibility: parse gate passed" not in names and "recruiter: Location" not in names)
    check("ranked: distinguishable first, then |r|",
          comp_rows[-1]["component"] == "recruiter: Salary"
          and not comp_rows[-1]["distinguishable_from_zero"]
          and comp_rows[0]["distinguishable_from_zero"]
          and abs(comp_rows[0]["correlation"]) >= abs(comp_rows[1]["correlation"]), str(comp_rows))

    st2 = applog.conversion_stats(db_path=db, min_resolved=20, reap_days=45)
    check("reap_days reaps first and reports it",
          st2["reaped"] == 1 and st2["resolved"] == 25 and "ghosted first" in st2["message"],
          st2["message"])

    st3 = applog.conversion_stats(db_path=db, min_resolved=100)
    check("below the threshold: no rates, explanation instead",
          not st3["calibrated"] and st3["bands"] == {} and "Need at least 100" in st3["message"])

# --------------------------------------------------------------- defaults
print("\n== company / role defaults from the JD ==")
cases = [
    ("https://boards.greenhouse.io/acme-corp/jobs/123", "Acme Corp"),
    ("https://jobs.lever.co/zeta/abc-123", "Zeta"),
    ("https://acme.wd5.myworkdayjobs.com/en-US/External/job/1", "Acme"),
    ("https://careers.tata.co.in/job/1", "Tata"),
    ("https://stripe.com/jobs/listing/analyst/1", "Stripe"),
    ("https://www.linkedin.com/jobs/view/12345", None),
    ("https://in.indeed.com/viewjob?jk=1", None),
    (None, None),
]
for url, want in cases:
    got = applog.company_from_url(url)
    check(f"url {url} -> {want}", got == want, repr(got))
check("'Company:' line wins", applog.company_from_jd("Job Title: Analyst\nCompany: Foo Ltd\n") == "Foo Ltd")
check("'About X' heading", applog.company_from_jd("About the role\nx\nAbout Acme Analytics\nWe build") ==
      "Acme Analytics")
check("'About us: X, ...'", applog.company_from_jd("About us: Zeta, a fintech in Pune") == "Zeta")
check("'About the team' is not a company", applog.company_from_jd("About the team\nAbout you\n") is None)
check("a sentence starting 'About' is not a company",
      applog.company_from_jd("About 20% travel is expected.") is None)
c, r = applog.default_company_role("Company: Foo\n", "https://jobs.lever.co/bar/1", "Data Analyst")
check("JD text beats the URL; role = JD title", (c, r) == ("Foo", "Data Analyst"), str((c, r)))
c, r = applog.default_company_role("no hints", "https://jobs.lever.co/bar/1", "")
check("URL fallback; empty title -> None", (c, r) == ("Bar", None), str((c, r)))

# --------------------------------------------------------------- lookups
print("\n== outcome lookups ==")
with tempfile.TemporaryDirectory() as td:
    db = str(Path(td) / "apps.db")
    check("no last id on an empty log", applog.last_application_id(db_path=db) is None)
    a1 = applog.log_application("ACME Corp", "Analyst", None, None, None, db_path=db)
    a2 = applog.log_application("Google", "Analyst", None, None, None, db_path=db)
    check("last id = newest insert", applog.last_application_id(db_path=db) == a2)
    check("substring match ignores case", [a.id for a in applog.find_applications("acme", db_path=db)] == [a1])
    check("close spelling matches", [a.id for a in applog.find_applications("gogle", db_path=db)] == [a2])
    check("no match -> empty", applog.find_applications("Microsoft", db_path=db) == [])

# --------------------------------------------------------------- CLI
print("\n== CLI: log outcome --last / company, --log defaults ==")


def cli(*argv, env_extra=None):
    env = {**os.environ, "ATS_LLM_PROVIDER": "ollama", "ATS_LLM_BASE_URL": "ollama-local"}
    env.pop("ATS_AUTO_LOG", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(ROOT / "cli.py"), *argv], capture_output=True,
                          text=True, cwd=str(ROOT), env=env, encoding="utf-8", errors="replace")


with tempfile.TemporaryDirectory() as td:
    db = str(Path(td) / "cli.db")
    resume = str(SAMPLES / "sample_resume.txt")
    jd_with_company = "Job Title: Data Analyst\nCompany: Acme Analytics\n" + jd_text
    p = cli("--db", db, "score", "--resume", resume, "--jd-text", jd_with_company,
            "--offline", "--brief", "--log")
    apps = applog.list_applications(db_path=db)
    check("score --log works with no --company/--role", p.returncode == 0 and len(apps) == 1,
          p.stderr[-400:])
    check("company and role defaulted from the JD",
          apps and apps[0].company == "Acme Analytics" and apps[0].role, str(apps and apps[0]))
    check("the CLI says what it assumed", "from the JD" in p.stdout, p.stdout[-300:])
    check("score --log stores components", apps and apps[0].components
          and apps[0].components.get("visibility") is not None)

    p = cli("--db", db, "score", "--resume", resume, "--jd-text", jd_with_company,
            "--offline", "--brief", "--company", "Other Co", "--role", "BI Analyst",
            env_extra={"ATS_AUTO_LOG": "1"})
    check("ATS_AUTO_LOG=1 logs without --log", p.returncode == 0 and
          len(applog.list_applications(db_path=db)) == 2, p.stderr[-300:])
    p = cli("--db", db, "score", "--resume", resume, "--jd-text", jd_with_company,
            "--offline", "--brief", "--no-log", env_extra={"ATS_AUTO_LOG": "1"})
    check("--no-log beats ATS_AUTO_LOG", len(applog.list_applications(db_path=db)) == 2)

    p = cli("--db", db, "log", "outcome", "--last", "--status", "interview")
    newest = max(applog.list_applications(db_path=db), key=lambda a: a.id)
    check("log outcome --last updates the newest", p.returncode == 0 and newest.outcome == "interview"
          and newest.company == "Other Co", p.stdout + p.stderr)
    p = cli("--db", db, "log", "outcome", "acme anal", "--status", "rejected_auto")
    acme = [a for a in applog.list_applications(db_path=db) if a.company == "Acme Analytics"][0]
    check("log outcome <company> fuzzy-matches", p.returncode == 0 and acme.outcome == "rejected_auto",
          p.stdout + p.stderr)
    applog.log_application("Acme Analytics", "Second", None, None, None, db_path=db)
    applog.log_application("Acme Analytics", "Third", None, None, None, db_path=db)
    p = cli("--db", db, "log", "outcome", "acme", "--status", "offer")
    check("ambiguous company match refuses and lists ids", p.returncode == 1 and "use the id" in p.stderr,
          p.stderr)
    p = cli("--db", db, "log", "outcome", str(acme.id), "--status", "ghosted")
    check("numeric id still works", p.returncode == 0)
    p = cli("--db", db, "log", "outcome", "--status", "offer")
    check("no target -> clear error", p.returncode == 1 and "--last" in p.stderr)

    p = cli("--db", db, "score", "--resume", resume, "--jd-text", "We need an analyst. SQL, Excel.",
            "--offline", "--brief", "--log")
    check("--log with nothing to infer fails clearly", p.returncode == 1 and "Not logged" in p.stderr,
          p.stderr[-300:])

    p = cli("--db", db, "log", "stats", "--json", "--min-resolved", "1")
    try:
        js = json.loads(p.stdout)
    except ValueError:
        js = {}
    check("log stats --json reaps first by default", p.returncode == 0 and js.get("reaped") == 0
          and "ghosted first" in js.get("message", ""), p.stdout[-300:] + p.stderr[-300:])
    p = cli("--db", db, "log", "stats", "--min-resolved", "1")
    check("log stats prints rates with a 95% CI", p.returncode == 0 and "95% CI" in p.stdout,
          p.stdout[-500:] + p.stderr[-300:])

# --------------------------------------------------------------- server
print("\n== server: outcome buttons over JSON, /log defaults ==")
try:
    import server
except ImportError as e:  # flask missing: say so rather than pass silently
    check("server importable", False, str(e))
else:
    with tempfile.TemporaryDirectory() as td:
        server.DB_PATH = str(Path(td) / "srv.db")
        client = server.app.test_client()
        r = client.post("/log", json={"jd_text": jd_with_company})
        body = r.get_json()
        check("/log defaults company/role from jd_text",
              r.status_code == 200 and body["company"] == "Acme Analytics" and body["role"], str(body))
        r = client.post("/log", json={"company": "X", "role": "Y", "components": {"keyword_score": 1}})
        check("/log stores supplied components",
              applog.list_applications(db_path=server.DB_PATH)[0].components == {"keyword_score": 1})
        app_id = r.get_json()["id"]
        r = client.post("/outcome", json={"id": app_id, "status": "interview"})
        check("POST /outcome JSON updates", r.status_code == 200 and
              [a.outcome for a in applog.list_applications(db_path=server.DB_PATH)
               if a.id == app_id] == ["interview"])
        r = client.post("/outcome", data=json.dumps({"id": app_id, "status": "offer"}),
                        content_type="text/plain")
        check("text/plain outcome is refused (CSRF fix kept)", r.status_code == 400)
        r = client.post("/score", json={"resume_text": (SAMPLES / "sample_resume.txt").read_text(
            encoding="utf-8"), "jd_text": jd_with_company, "offline": True, "log": True})
        newest = max(applog.list_applications(db_path=server.DB_PATH), key=lambda a: a.id)
        check("/score log defaults from JD and stores components",
              r.status_code == 200 and newest.company == "Acme Analytics" and newest.components,
              str(r.get_json().get("error")))
        applog.log_application("Old", "R", None, None, None, applied_date="2020-01-01",
                               db_path=server.DB_PATH)
        r = client.get("/stats?min_resolved=1")
        check("GET /stats is read-only and says ghosts weren't reaped",
              r.get_json()["reaped"] is None and "NOT reaped" in r.get_json()["message"]
              and any(a.outcome == "pending" and a.company == "Old"
                      for a in applog.list_applications(db_path=server.DB_PATH)))
        page = client.get("/").get_data(as_text=True)
        check("history UI has outcome buttons posting JSON",
              "function setOutcome" in page and 'post("/outcome"' in page and "outcomeButtons(a)" in page)
        check("UI no longer demands company + role", "needs company + role" not in page)

print(f"\n{'=' * 60}\nApplog tests: {passed} passed, {failed} failed")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all_checks_passed():
    """pytest entry point: the checks above run at import (collection) time."""
    assert failed == 0, f"{failed} check(s) failed — run this file directly for details"
