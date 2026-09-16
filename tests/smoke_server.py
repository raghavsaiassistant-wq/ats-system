"""Quick smoke test of the local HTTP API (run with the server up)."""
import json
import urllib.request

BASE = "http://127.0.0.1:8421"


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


jd = open("samples/sample_jd.txt", encoding="utf-8").read()
resume = open("samples/sample_resume.txt", encoding="utf-8").read()

r = post("/score", {"resume_text": resume, "jd_text": jd, "offline": True})
print("SCORE  -> ATS:", r["scores"]["ats_score"],
      "| HR:", r["scores"]["hr_screen_criteria_met_pct"],
      "| Mgr:", r["scores"]["manager_evidence_strength_pct"])
print("        blockers:", len(r["recruiter_layer"]["blockers"]),
      "| checks:", len(r["recruiter_layer"]["checks"]),
      "| expectations:", len(r["recruiter_layer"]["hr_expectations"]))
print("        matched keywords:", len(r["ats_layer"]["keywords"]["matched"]),
      "| missing:", len(r["ats_layer"]["keywords"]["missing"]))
print("        jd salary:", r["jd_requirements"]["salary_min"], r["jd_requirements"]["salary_max"],
      "| location:", repr(r["jd_requirements"]["jd_location"]))

r = post("/score", {"resume_text": resume, "jd_text": jd, "offline": True,
                    "log": True, "company": "Acme", "role": "BI Analyst"})
print("SCORE+LOG -> logged_id:", r["logged_id"])

r = post("/tailor", {"jd_text": jd, "offline": True})
print("TAILOR -> blocked:", r["blocked"], "| gaps:", len(r["honest_gaps"]),
      "| resume chars:", len(r["resume_text"]))
if r["scores"]:
    print("         ATS:", r["scores"]["ats_score"], "| HR:", r["scores"]["hr_screen_criteria_met_pct"])
print("         sample resume head:", r["resume_text"][:80].replace("\n", " / "))

# outcome update
r = post("/outcome", {"id": 1, "status": "recruiter_call"})
print("OUTCOME ->", r)