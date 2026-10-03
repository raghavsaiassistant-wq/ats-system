#!/usr/bin/env python3
"""First-run setup tests (Phase 5) — offline, no network, no real LLM.

Covers: profile form round-trip + validation (and the "no"-is-truthy fix),
the evidence bank's review state (LLM-transcribed bullets start unreviewed,
the tailor skips them, hand-written YAML stays reviewed), the setup JSON
endpoints (JSON-only, writes only to the configured paths, the API key only
to the configured host), and the web UI wiring.

    python tests/test_setup.py
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from ats_checker import llm_client  # noqa: E402
from ats_checker import profile as profile_mod  # noqa: E402
from ats_checker import generator as gen  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from mock_provider import ATOMIZE_PAYLOAD  # noqa: E402

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
JD_TEXT = (SAMPLES / "sample_jd.txt").read_text(encoding="utf-8")


class FakeLLM:
    """Stands in for llm_client.call_json; records what it was sent."""

    def __init__(self, payload=None, error=None):
        self.payload = payload if payload is not None else json.loads(json.dumps(ATOMIZE_PAYLOAD))
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, system, user, **kw):
        self.calls.append({"system": system, "user": user, **kw})
        return (None, self.error) if self.error else (self.payload, None)


def with_fake_llm(fake):
    original = llm_client.call_json
    llm_client.call_json = fake
    return original


# --------------------------------------------------------------- profile
print("\n== profile: form round-trip + validation ==")
form = {
    "current_title": " Data Analyst ", "years_experience": "2.5", "education_level": "bachelors",
    "location": "Pune, India", "open_to_relocation": "no", "notice_period_days": "30",
    "acceptable_work_modes": ["Remote", "hybrid"], "work_authorization": "",
    "work_authorized_in": "India, UAE", "expected_salary_min": "", "expected_salary_max": None,
    "salary_currency": "INR", "certifications": "PL-300\nAWS CCP",
}
prof = profile_mod.profile_from_dict(form)
check("strings coerced to numbers", prof.years_experience == 2.5 and prof.notice_period_days == 30)
check("blank numbers are None (check skipped)", prof.expected_salary_min is None)
check("'no' is False, not truthy", prof.open_to_relocation is False)
check("comma / newline lists split", prof.work_authorized_in == ["India", "UAE"]
      and prof.certifications == ["PL-300", "AWS CCP"], str(prof))
check("work modes lowercased", prof.acceptable_work_modes == ["remote", "hybrid"])
check("text trimmed", prof.current_title == "Data Analyst")
check("valid profile has no errors", profile_mod.validate_profile(prof) == [])
check("unknown text bool -> None", profile_mod.profile_from_dict({"open_to_relocation": "maybe"})
      .open_to_relocation is None)

bad = profile_mod.profile_from_dict({"education_level": "wizard", "acceptable_work_modes": ["moon"],
                                     "years_experience": -1, "expected_salary_min": 10,
                                     "expected_salary_max": 5})
errs = profile_mod.validate_profile(bad)
check("bad education, mode, years and salary order all flagged", len(errs) == 4, str(errs))
check("an all-blank profile is valid (checks get skipped)",
      profile_mod.validate_profile(profile_mod.profile_from_dict({})) == [])

with tempfile.TemporaryDirectory() as td:
    path = str(Path(td) / "profile.yaml")
    profile_mod.save_profile(prof, path)
    back = profile_mod.load_profile(path)
    check("save -> load round-trips", profile_mod.profile_to_dict(back) == profile_mod.profile_to_dict(prof))
    check("no .tmp left behind", not (Path(td) / "profile.yaml.tmp").exists())
    text = Path(path).read_text(encoding="utf-8")
    check("written file explains blank fields", text.startswith("# Your fixed profile"))
    try:
        profile_mod.save_profile(bad, path)
        refused = False
    except ValueError:
        refused = True
    check("invalid profile is not written", refused and profile_mod.load_profile(path).current_title == "Data Analyst")
    Path(path).write_text('open_to_relocation: "no"\ncurrent_title: X\n', encoding="utf-8")
    check("quoted 'no' in an existing profile.yaml reads as False",
          profile_mod.load_profile(path).open_to_relocation is False)

# --------------------------------------------------------------- bank review state
print("\n== evidence bank: review state ==")
data = {
    "contact": {"name": "A", "email": "a@x.com", "phone": "1"},
    "headline": "Analyst",
    "roles": [{"company": "C", "title": "T", "start": "2023-01", "end": "present", "bullets": [
        {"text": "Built SQL models for finance reporting", "skills": ["SQL"]},
        {"text": "Invented a fusion reactor", "skills": ["physics"], "reviewed": False},
        "Plain string bullet stays reviewed",
    ]}],
}
bank = gen.bank_from_dict(data)
flags = [b.reviewed for b in bank.roles[0].bullets]
check("missing flag = reviewed (hand-written), explicit false kept", flags == [True, False, True], str(flags))
check("skills lowercased", bank.roles[0].bullets[0].skills == ["sql"])
check("unreviewed_count", bank.unreviewed_count() == 1)
kept, dropped = bank.reviewed_only()
check("reviewed_only drops only unreviewed", dropped == 1 and len(kept.roles[0].bullets) == 2
      and bank.unreviewed_count() == 1)  # original untouched
d = gen.bank_to_dict(bank)
check("bank_to_dict writes reviewed:false only where false",
      ["reviewed" in b for b in d["roles"][0]["bullets"]] == [False, True, False])
check("flat contact fields accepted (LLM shape)",
      gen.bank_from_dict({"name": "Flat", "email": "f@x.com", "roles": []}).email == "f@x.com")
check("'false' string flag honoured", not gen.bank_from_dict(
    {"roles": [{"company": "C", "bullets": [{"text": "t", "reviewed": "false"}]}]}).roles[0].bullets[0].reviewed)

with tempfile.TemporaryDirectory() as td:
    path = str(Path(td) / "master.yaml")
    gen.save_bank(bank, path)
    back = gen.load_bank(path)
    check("save -> load keeps review flags",
          [b.reviewed for b in back.roles[0].bullets] == [True, False, True])
    check("saved YAML says how to confirm", "reviewed: false" in Path(path).read_text(encoding="utf-8")
          and "Setup tab" in Path(path).read_text(encoding="utf-8"))

    # the tailor never uses an unreviewed bullet
    res = gen.tailor(master_path=path, jd_text=JD_TEXT, offline=True, force=True)
    check("tailor leaves the unreviewed bullet out", "fusion reactor" not in res.resume_text)
    check("tailor says it left one out", any("1 unreviewed" in p for p in res.bank_problems), str(res.bank_problems))

    for b in bank.roles[0].bullets:
        b.reviewed = False
    gen.save_bank(bank, path)
    try:
        gen.tailor(master_path=path, jd_text=JD_TEXT, offline=True, force=True)
        msg = ""
    except ValueError as e:
        msg = str(e)
    check("all-unreviewed bank refuses with a clear message", "unreviewed" in msg and "Setup tab" in msg, msg)

# --------------------------------------------------------------- init_from_resume
print("\n== init-master --from: everything starts unreviewed ==")
fake = FakeLLM()
original = with_fake_llm(fake)
try:
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "master.yaml")
        path, notes = gen.init_from_resume(str(SAMPLES / "sample_resume.txt"), out_path=out,
                                           model="m", host="h", api_key="k", provider="openai")
        back = gen.load_bank(path)
        bullets = [b for r in back.roles for b in r.bullets]
        check("all transcribed bullets unreviewed (even one the LLM marked reviewed)",
              len(bullets) == 3 and not any(b.reviewed for b in bullets), str(bullets))
        check("notes tell the user to confirm", any("UNREVIEWED" in n for n in notes), str(notes))
        check("contact + extras transcribed", back.email == "asha.rao@example.com"
              and back.skills_extra == ["dax", "tableau"] and back.certifications == ["PL-300"])
        check("LLM got the resume text", "RESUME TO TRANSCRIBE" in fake.calls[0]["user"])
        try:
            gen.init_from_resume(str(SAMPLES / "sample_resume.txt"), out_path=out)
            exists_refused = False
        except FileExistsError:
            exists_refused = True
        check("won't overwrite an existing bank without overwrite", exists_refused)
finally:
    llm_client.call_json = original

# --------------------------------------------------------------- server
print("\n== server: setup endpoints ==")
try:
    import server
except ImportError as e:
    check("server importable", False, str(e))
else:
    with tempfile.TemporaryDirectory() as td:
        server.PROFILE_PATH = str(Path(td) / "profile.yaml")
        server.MASTER_PATH = str(Path(td) / "master.yaml")
        server.DB_PATH = str(Path(td) / "apps.db")
        client = server.app.test_client()

        orig_cfg = server.ollama_client.current_config
        server.ollama_client.current_config = lambda: {
            "provider": "openai", "model": "m", "base_url": "https://configured.example/v1",
            "api_key": "sk-SECRET-123"}
        try:
            r = client.get("/setup/status")
            st = r.get_json()
            check("first run detected", st["first_run"] and not st["profile"]["exists"]
                  and not st["bank"]["exists"], str(st))
            check("status says a key is set but never shows it",
                  st["llm"]["api_key_set"] is True and "sk-SECRET" not in r.get_data(as_text=True))
        finally:
            server.ollama_client.current_config = orig_cfg

        r = client.get("/profile")
        check("GET /profile on first run: empty + allowed values",
              r.get_json()["exists"] is False and "bachelors" in r.get_json()["education_levels"])

        # JSON-only: a cross-site text/plain POST arrives as {} — which would be
        # a VALID (all blank) profile — so it must be refused, not saved.
        profile_mod.save_profile(prof, server.PROFILE_PATH)
        before = Path(server.PROFILE_PATH).read_text(encoding="utf-8")
        for url in ("/profile", "/master", "/master/import", "/setup/test-llm"):
            r = client.post(url, data="{}", content_type="text/plain")
            check(f"text/plain POST {url} refused (415)", r.status_code == 415, str(r.status_code))
        r = client.post("/profile", data="", content_type="application/x-www-form-urlencoded")
        check("form POST /profile refused", r.status_code == 415)
        check("profile untouched by refused posts", Path(server.PROFILE_PATH).read_text(encoding="utf-8") == before)

        r = client.post("/profile", json={"profile": {**form, "current_title": "BI Analyst"}})
        check("POST /profile saves", r.status_code == 200 and
              profile_mod.load_profile(server.PROFILE_PATH).current_title == "BI Analyst", str(r.get_json()))
        r = client.post("/profile", json={"profile": {"education_level": "wizard"}})
        check("invalid profile -> 400 with errors", r.status_code == 400 and r.get_json()["errors"])
        check("invalid profile not written",
              profile_mod.load_profile(server.PROFILE_PATH).current_title == "BI Analyst")
        r = client.post("/profile", json={"profile": {"current_title": "X"},
                                          "path": str(Path(td) / "elsewhere.yaml")})
        check("a request can't choose where the profile is written",
              not (Path(td) / "elsewhere.yaml").exists())

        r = client.get("/master")
        check("GET /master with no bank -> 404 exists:false", r.status_code == 404 and r.get_json()["exists"] is False)

        # ---- import
        b64 = base64.b64encode((SAMPLES / "sample_resume.txt").read_bytes()).decode()
        r = client.post("/master/import", json={"filename": "cv.exe", "content_b64": b64})
        check("unsupported file type refused", r.status_code == 400)
        r = client.post("/master/import", json={"filename": "cv.txt", "content_b64": "%%%not base64"})
        check("bad base64 refused", r.status_code == 400)
        r = client.post("/master/import", json={"filename": "cv.txt", "content_b64": b64, "offline": True})
        check("offline import explains it needs the LLM", r.status_code == 400 and "blank" in r.get_json()["error"])

        fake = FakeLLM()
        original = with_fake_llm(fake)
        # a real configured key, so "the attacker host got no key" can't pass vacuously
        orig_key = server.ollama_client.DEFAULT_API_KEY
        server.ollama_client.DEFAULT_API_KEY = "sk-CONFIGURED-456"
        try:
            r = client.post("/master/import", json={"filename": "cv.txt", "content_b64": b64})
            body = r.get_json()
            check("import transcribes into the configured bank", r.status_code == 200 and body["bullets"] == 3
                  and Path(server.MASTER_PATH).exists(), str(body)[:300])
            check("every imported bullet is unreviewed", body["unreviewed"] == 3 and all(
                b.get("reviewed") is False for role in body["bank"]["roles"] for b in role["bullets"]))
            check("configured host gets the configured key",
                  fake.calls[-1]["host"] == server.ollama_client.DEFAULT_HOST
                  and fake.calls[-1]["api_key"] == "sk-CONFIGURED-456")

            r = client.post("/master/import", json={"filename": "cv.txt", "content_b64": b64})
            check("existing bank needs overwrite:true (409)", r.status_code == 409 and r.get_json()["exists"])

            r = client.post("/master/import", json={"filename": "cv.txt", "content_b64": b64, "overwrite": True,
                                                    "host": "https://attacker.example/v1"})
            check("a request naming its own host never gets the configured key",
                  r.status_code == 200 and fake.calls[-1]["host"] == "https://attacker.example/v1"
                  and fake.calls[-1]["api_key"] == "", str(fake.calls[-1].get("api_key")))
            r = client.post("/master/import", json={"filename": "cv.txt", "content_b64": b64, "overwrite": True,
                                                    "host": "https://attacker.example/v1", "api_key": "their-own"})
            check("...unless it brings its own key", fake.calls[-1]["api_key"] == "their-own")
        finally:
            llm_client.call_json = original
            server.ollama_client.DEFAULT_API_KEY = orig_key

        failing = FakeLLM(error="HTTP 401: invalid api key")
        original = with_fake_llm(failing)
        try:
            r = client.post("/master/import", json={"filename": "cv.txt", "content_b64": b64, "overwrite": True})
            check("LLM failure -> 502 with the reason", r.status_code == 502 and "401" in r.get_json()["error"])
        finally:
            llm_client.call_json = original

        # ---- editor save
        bank_json = client.get("/master").get_json()["bank"]
        bank_json["roles"][0]["bullets"][0]["reviewed"] = True                       # confirm
        bank_json["roles"][0]["bullets"][1]["text"] = "Automated weekly Excel reports"  # edit, still unreviewed
        del bank_json["roles"][0]["bullets"][2]                                       # delete
        r = client.post("/master", json={"bank": bank_json, "path": str(Path(td) / "evil.yaml")})
        body = r.get_json()
        check("POST /master saves edits", r.status_code == 200 and body["bullets"] == 2 and body["unreviewed"] == 1,
              str(body)[:300])
        saved = yaml.safe_load(Path(server.MASTER_PATH).read_text(encoding="utf-8"))
        check("confirmed bullet loses its reviewed:false; edited one keeps it",
              [b.get("reviewed", True) for b in saved["roles"][0]["bullets"]] == [True, False]
              and saved["roles"][0]["bullets"][1]["text"] == "Automated weekly Excel reports")
        check("a request can't choose where the bank is written", not (Path(td) / "evil.yaml").exists())
        r = client.post("/master", json={"nope": 1})
        check("POST /master without a bank -> 400", r.status_code == 400)

        st = client.get("/setup/status").get_json()
        check("status: no longer first run, 1 unreviewed", not st["first_run"] and st["bank"]["unreviewed"] == 1)

        # tailor over HTTP uses only the confirmed bullet
        r = client.post("/tailor", json={"jd_text": JD_TEXT, "offline": True, "force": True})
        tj = r.get_json()
        check("/tailor skips unreviewed bullets and says so",
              "Automated weekly Excel reports" not in (tj.get("resume_text") or "")
              and any("unreviewed" in p for p in tj.get("bank_problems", [])), str(tj)[:300])

        page = client.get("/").get_data(as_text=True)
        check("UI has the Setup tab + wizard", 'id="tab-setup"' in page and "function initFirstRun" in page)
        check("UI posts JSON for profile, bank and import",
              'post("/profile"' in page and 'post("/master"' in page and 'post("/master/import"' in page)
        check("UI confirm is per bullet (no confirm-all)", "confirmBullet" in page and "confirmAll" not in page)

# --------------------------------------------------------------- offline still works
print("\n== offline scoring unaffected ==")
from ats_checker.scorer import run_full_check  # noqa: E402

rep = run_full_check(resume_path=str(SAMPLES / "sample_resume.txt"), jd_text=JD_TEXT,
                     skip_semantic=True, skip_manager=True)
check("offline score still runs", rep.visibility is not None and rep.recruiter_score is not None)

print(f"\n{'=' * 60}\nSetup tests: {passed} passed, {failed} failed")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all_checks_passed():
    """pytest entry point: the checks above run at import (collection) time."""
    assert failed == 0, f"{failed} check(s) failed — run this file directly for details"
