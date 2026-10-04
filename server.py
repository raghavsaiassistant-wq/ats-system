#!/usr/bin/env python3
"""Local HTTP API + web UI so a human (or James, or n8n, or any script) can
use the checker.

Run:
    python server.py                 # http://127.0.0.1:8420

Endpoints:
    GET  /               single-page web UI (paste, click, read the report)
    GET  /health
    POST /score          three-layer scoring
    POST /tailor         generate a JD-tailored resume from the evidence bank
    POST /log            record an application (company/role default from jd_text)
    POST /outcome        update an application's outcome
    GET  /applications   list logged applications
    GET  /setup/status   first-run state: profile? evidence bank (+ unreviewed bullets)? LLM?
    POST /setup/test-llm one test request to the configured (or given) LLM
    GET  /profile        profile.yaml as JSON (+ allowed values for the form)
    POST /profile        write profile.yaml from the setup wizard's form
    GET  /master         the evidence bank as JSON, with review state + problems
    POST /master         save the evidence bank from the editor
    POST /master/import  resume upload (base64 in JSON) -> LLM transcription
                         (init-master --from); every bullet starts unreviewed
    POST /manual/answers copy-paste LLM mode: store a chat AI's pasted JSON reply
                         (any endpoint may answer 202 {"manual_pending": {bundle, ids}})
    GET  /stats          conversion by score band + predictiveness, each rate with n and a
                         95% CI (once enough outcomes); read-only, so it reports stale
                         pendings rather than reaping them

POST /score body:
    {
      "resume_text": "...",          # or "resume_path": "/abs/path.pdf"
      "jd_text": "...",              # or "jd_url": "https://..."
      "offline": false,              # true = skip both LLM layers
      "model": "...", "host": "...", "api_key": "..."   # optional overrides
    }

POST /tailor body:
    {
      "jd_text": "...",              # or "jd_url": "https://..."
      "master_path": "...",          # evidence bank (default: master_resume.yaml)
      "offline": false, "force": false,
      "max_current": 5, "max_other": 3, "max_lines": 26,
      "log": true, "company": "...", "role": "..."   # company/role default from the JD
    }

Returns the same JSON shape as `cli.py score --json`: read
`scores.search_visibility_pct` (+ `scores.parse_safe`), `scores.hr_screen_criteria_met_pct`,
`scores.manager_evidence_strength_pct`, plus `recruiter_layer.blockers` and
`manager_layer.weak_bullets` for the actionable parts.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import tempfile
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, request

from ats_checker import applog
from ats_checker import generator as gen_mod
from ats_checker import llm_client as ollama_client
from ats_checker import manual_llm
from ats_checker import profile as profile_mod
from ats_checker.jd_fetch import fetch_jd_url
from ats_checker.scorer import run_full_check
from ats_checker import parsing, writing_check

app = Flask(__name__)
# Resume uploads arrive base64-encoded inside JSON (see /master/import), so
# allow a little over MAX_UPLOAD_BYTES for the encoding overhead.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES * 4 // 3 + 64 * 1024
UPLOAD_TYPES = (".pdf", ".docx", ".txt")
# `python server.py --llm manual`: copy-paste mode for every request that
# doesn't name its own provider (no API key needed).
PROVIDER_OVERRIDE: str | None = None
PROFILE_PATH = profile_mod.DEFAULT_PROFILE_PATH
DB_PATH = applog.DEFAULT_DB
MASTER_PATH = gen_mod.DEFAULT_MASTER_PATH


def _json_body() -> dict:
    """The request's JSON object, or {}.

    Deliberately NOT force=True: that accepted text/plain bodies, which a
    browser sends cross-origin without a CORS preflight — so any web page
    the user visited could drive this localhost API. Requiring
    application/json makes the browser preflight, which Flask never grants."""
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else {}


def _llm_kwargs(body: dict) -> dict:
    """LLM overrides from the request. The configured API key is only ever
    sent to the configured host: a request that names its own host must
    bring its own key, or it could redirect the user's key (and the resume
    text in the prompt) to a server of its choosing."""
    host = body.get("host") or ollama_client.DEFAULT_HOST
    default_key = ollama_client.DEFAULT_API_KEY if host == ollama_client.DEFAULT_HOST else ""
    return {
        "model": body.get("model") or ollama_client.DEFAULT_MODEL,
        "host": host,
        "api_key": body.get("api_key") or default_key,
        "provider": body.get("provider") or PROVIDER_OVERRIDE,
    }


def _manual_aware(view):
    """Copy-paste LLM mode for an endpoint: if the request left prompts
    waiting for a pasted answer, reply 202 with the prompt bundle instead of
    a half-computed result. The UI shows "Copy prompt / Paste answer", posts
    the reply to /manual/answers, and sends the same request again."""
    @wraps(view)
    def inner(*args, **kwargs):
        token = manual_llm.new_session()
        try:
            resp = view(*args, **kwargs)
            todo = manual_llm.pending()
        finally:
            manual_llm.end_session(token)
        if todo:
            return jsonify({"manual_pending": {
                "bundle": manual_llm.build_bundle(todo),
                "ids": [p.id for p in todo],
                "links": list(manual_llm.CHAT_LINKS),
            }}), 202
        return resp
    return inner


def _resolve_jd_text(body: dict) -> tuple[str, str | None]:
    """(jd_text, error) — accepts jd_text directly or fetches jd_url."""
    jd_text = body.get("jd_text") or ""
    jd_url = body.get("jd_url") or ""
    if not jd_text and jd_url:
        try:
            jd_text = fetch_jd_url(str(jd_url))
        except Exception as e:  # noqa: BLE001 — surface fetch errors to the caller
            return "", f"Could not fetch jd_url: {e}"
    return jd_text, None


def _log_from_report(body: dict, report, jd_text: str, resume_version: str) -> tuple[int | None, str | None]:
    """(logged id, error). Company/role default from the JD when the request
    leaves them blank: role = the JD title, company = a 'Company:'/'About X'
    line or the posting URL's host."""
    jd_title = report.jd_reqs.jd_title if report is not None else None
    company_guess, role_guess = applog.default_company_role(jd_text, body.get("jd_url"), jd_title)
    company = body.get("company") or company_guess
    role = body.get("role") or role_guess
    if not (company and role):
        return None, "'log': true needs 'company' and 'role' (couldn't tell them from the JD)"
    app_id = applog.log_application(
        company=company,
        role=role,
        ats_score=report.ats_score if report else None,
        visibility_score=report.visibility.score if report and report.visibility else None,
        recruiter_score=report.recruiter_score if report else None,
        manager_score=report.manager_score if report else None,
        jd_text=jd_text,
        resume_version=resume_version,
        days_after_posting=body.get("days_after_posting"),
        db_path=DB_PATH,
        report=report,
    )
    return app_id, None


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/score")
@_manual_aware
def score():
    body = _json_body()
    resume_text = body.get("resume_text")
    resume_path = body.get("resume_path")
    jd_text, err = _resolve_jd_text(body)
    if err:
        return jsonify({"error": err}), 400

    if not resume_text and not resume_path:
        return jsonify({"error": "Provide 'resume_text' or 'resume_path'"}), 400
    if not jd_text.strip():
        return jsonify({"error": "Provide 'jd_text' or 'jd_url'"}), 400

    offline = bool(body.get("offline", False))
    try:
        result = run_full_check(
            resume_path=resume_path,
            resume_text=resume_text,
            jd_text=jd_text,
            profile=profile_mod.load_profile(body.get("profile_path", PROFILE_PATH)),
            **_llm_kwargs(body),
            skip_semantic=offline or bool(body.get("skip_semantic", False)),
            skip_manager=offline or bool(body.get("skip_manager", False)),
            jd_extractor="rules" if offline else str(body.get("jd_extractor") or "rules"),
        )
    except Exception as e:  # noqa: BLE001 — surface parsing/scoring errors to the caller
        return jsonify({"error": str(e)}), 400

    payload = result.to_dict()

    # in copy-paste mode a pass with prompts still pending is thrown away and
    # re-run, so only the complete pass may write to the log
    if body.get("log") and not manual_llm.pending():
        app_id, err = _log_from_report(body, result, jd_text,
                                       body.get("resume_version", resume_path or "inline"))
        if err:
            return jsonify({"error": err}), 400
        payload["logged_id"] = app_id

    return jsonify(payload)


@app.post("/writing-check")
def writing_review():
    body = _json_body()
    if body.get('consent') is not None and not isinstance(body['consent'], bool):
        return jsonify({'error': 'consent must be a JSON boolean'}), 400
    try:
        if not body.get('resume_text') and not body.get('resume_path'):
            raise ValueError('Provide resume_text or resume_path')
        parsed = parsing.analyze(path=body.get('resume_path'), text=body.get('resume_text'))
        result = writing_check.check_writing(parsed.text, provider=body.get('provider', 'local'),
                                            consent=body.get('consent') is True)
        result['extraction_warnings'] = parsed.warnings
        return jsonify(result)
    except (ValueError, TypeError, OSError) as exc:
        return jsonify({'error': str(exc)}), 400


@app.post("/tailor")
@_manual_aware
def tailor():
    """Generate a JD-tailored resume from the evidence bank over HTTP —
    the same `cli.py tailor` pipeline (no-apply gate, verified rewording,
    honest gaps), JSON in, JSON out."""
    body = _json_body()
    jd_text, err = _resolve_jd_text(body)
    if err:
        return jsonify({"error": err}), 400
    if not jd_text.strip():
        return jsonify({"error": "Provide 'jd_text' or 'jd_url'"}), 400

    master_path = body.get("master_path") or MASTER_PATH
    master_text = body.get("master_text")
    tmp_dir = None
    if master_text:
        # bank supplied inline — write it to a temp file for this request
        tmp_dir = tempfile.TemporaryDirectory(prefix="ats_master_")
        master_path = str(Path(tmp_dir.name) / "master_resume.yaml")
        Path(master_path).write_text(str(master_text), encoding="utf-8")

    try:
        result = gen_mod.tailor(
            master_path=master_path,
            jd_text=jd_text,
            profile=profile_mod.load_profile(body.get("profile_path", PROFILE_PATH)),
            offline=bool(body.get("offline", False)),
            force=bool(body.get("force", False)),
            max_current=body.get("max_current", 5),
            max_other=body.get("max_other", 3),
            max_lines=body.get("max_lines", 26),
            **_llm_kwargs(body),
        )
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001 — unexpected failures reach the caller
        return jsonify({"error": str(e)}), 500
    finally:
        # the inline bank is only needed for the call itself — clean up on
        # success too, not just on the error paths
        if tmp_dir:
            tmp_dir.cleanup()

    rep = result.report
    payload = {
        "blocked": result.blocked,
        "candidacy_blockers": result.candidacy_blockers,
        "bank_problems": result.bank_problems,
        "resume_text": "" if result.blocked else result.resume_text,
        "honest_gaps": result.honest_gaps,
        "rewordings": result.rewordings,
        "rejected_rewrites": result.rejected_rewrites,
        "suggestions": result.suggestions,
        "notes": result.notes,
        "used_llm": result.used_llm,
        "scores": (
            {
                "search_visibility_pct": rep.visibility.score if rep.visibility else None,
                "parse_safe": rep.visibility.parse_safe if rep.visibility else None,
                "ats_score": rep.ats_score,
                "hr_screen_criteria_met_pct": rep.recruiter_score,
                "hr_resume_fixable_pct": rep.recruiter_result.resume_pct if rep.recruiter_result else None,
                "hr_candidacy_fit_pct": rep.recruiter_result.candidacy_pct if rep.recruiter_result else None,
                "manager_evidence_strength_pct": rep.manager_score,
            } if rep else None
        ),
    }

    if body.get("log") and not result.blocked and rep is not None and not manual_llm.pending():
        app_id, err = _log_from_report(body, rep, jd_text, "tailored:inline")
        if err:
            return jsonify({"error": err}), 400
        payload["logged_id"] = app_id

    return jsonify(payload)


@app.post("/log")
def log():
    body = _json_body()
    company, role = body.get("company"), body.get("role")
    if not (company and role) and body.get("jd_text"):
        from ats_checker import jd_requirements

        guess_c, guess_r = applog.default_company_role(
            body["jd_text"], body.get("jd_url"), jd_requirements.extract(body["jd_text"]).jd_title)
        company, role = company or guess_c, role or guess_r
    if not company or not role:
        return jsonify({"error": "Provide 'company' and 'role' (or a jd_text they can be read from)"}), 400
    components = body.get("components")
    app_id = applog.log_application(
        company=company,
        role=role,
        ats_score=body.get("ats_score"),
        visibility_score=body.get("visibility_score"),
        recruiter_score=body.get("recruiter_score"),
        manager_score=body.get("manager_score"),
        jd_text=body.get("jd_text", ""),
        resume_version=body.get("resume_version", ""),
        days_after_posting=body.get("days_after_posting"),
        notes=body.get("notes", ""),
        db_path=DB_PATH,
        components=components if isinstance(components, dict) else None,
    )
    return jsonify({"id": app_id, "company": company, "role": role})


@app.post("/outcome")
def outcome():
    body = _json_body()
    app_id, status = body.get("id"), body.get("status")
    if app_id is None or not status:
        return jsonify({"error": "Provide 'id' and 'status'", "valid_status": applog.OUTCOMES}), 400
    try:
        ok = applog.set_outcome(int(app_id), status, notes=body.get("notes"), db_path=DB_PATH)
    except ValueError as e:
        return jsonify({"error": str(e), "valid_status": applog.OUTCOMES}), 400
    if not ok:
        return jsonify({"error": f"No application with id {app_id}"}), 404
    return jsonify({"id": int(app_id), "outcome": status})


@app.get("/applications")
def applications():
    limit = request.args.get("limit", default=50, type=int)
    apps = applog.list_applications(limit=limit, db_path=DB_PATH)
    return jsonify([a.__dict__ for a in apps])


@app.get("/stats")
def stats():
    min_resolved = request.args.get("min_resolved", default=20, type=int)
    # Read-only: a GET must not change the log (any page can make the browser
    # send one cross-site), so ghosts aren't reaped here — the result says how
    # many stale pendings were left out. `cli.py log stats` reaps first.
    return jsonify(applog.conversion_stats(db_path=DB_PATH, min_resolved=min_resolved))


# ------------------------------------------------------- first-run setup
#
# Everything here is JSON in, JSON out (the CSRF rule: no form posts, no
# text/plain), and writes only ever go to the paths the server was started
# with — a request can't name a file to overwrite.

def _not_json():
    """415 unless the request really is JSON. The setup endpoints write files,
    and an empty profile is a VALID profile — so a cross-site text/plain POST
    (which arrives as {}) must be refused outright, not read as 'all blank'."""
    if not request.is_json:
        return jsonify({"error": "Send Content-Type: application/json"}), 415
    return None


def _bank_payload() -> tuple[dict, int]:
    path = Path(MASTER_PATH)
    if not path.exists():
        return {"exists": False, "path": str(path)}, 404
    try:
        bank = gen_mod.load_bank(path)
    except Exception as e:  # noqa: BLE001 — a broken YAML file is reported, not fatal
        return {"exists": True, "path": str(path), "error": f"Could not read the bank: {e}"}, 422
    try:
        problems = bank.validate()
    except ValueError as e:
        problems = [str(e)]
    total = sum(len(r.bullets) for r in bank.roles)
    return {
        "exists": True,
        "path": str(path),
        "bank": gen_mod.bank_to_dict(bank),
        "bullets": total,
        "unreviewed": bank.unreviewed_count(),
        "problems": problems,
    }, 200


@app.get("/setup/status")
def setup_status():
    cfg = dict(ollama_client.current_config())
    if PROVIDER_OVERRIDE == "manual":
        cfg.update(provider="manual", model="manual", base_url="manual", api_key="")
    bank, code = _bank_payload()
    profile_exists = Path(PROFILE_PATH).exists()
    return jsonify({
        "first_run": not profile_exists or not bank.get("exists"),
        "profile": {"exists": profile_exists, "path": str(Path(PROFILE_PATH))},
        "bank": {k: bank.get(k) for k in ("exists", "path", "bullets", "unreviewed", "error")},
        "llm": {   # never the key itself — only whether one is set
            "provider": cfg["provider"], "model": cfg["model"], "base_url": cfg["base_url"],
            "api_key_set": bool(cfg["api_key"]),
        },
    })


@app.post("/setup/test-llm")
def setup_test_llm():
    if (refused := _not_json()):
        return refused
    kw = _llm_kwargs(_json_body())
    ok, message = ollama_client.test_connection(
        model=kw["model"], host=kw["host"], api_key=kw["api_key"], provider=kw["provider"])
    return jsonify({"ok": ok, "message": message})


@app.post("/manual/answers")
def manual_answers():
    """Copy-paste mode: the reply a chat AI gave to a prompt bundle. Only
    stored in the local answer cache; every answer still goes through the
    normal checks when the request is re-sent."""
    if (refused := _not_json()):
        return refused
    body = _json_body()
    ids = [str(i) for i in (body.get("ids") or []) if isinstance(i, str)]
    answers, problems = manual_llm.parse_answers(str(body.get("reply") or ""), ids)
    if not answers:
        return jsonify({"error": " ".join(problems) or "No answers found in the reply.",
                        "problems": problems}), 400
    for pid, ans in answers.items():
        manual_llm.store_answer(pid, ans)
    return jsonify({"stored": len(answers), "problems": problems})


@app.get("/profile")
def get_profile():
    exists = Path(PROFILE_PATH).exists()
    try:
        prof = profile_mod.load_profile(PROFILE_PATH)
    except Exception as e:  # noqa: BLE001 — surface a broken file to the wizard
        return jsonify({"exists": exists, "error": f"Could not read {PROFILE_PATH}: {e}"}), 422
    return jsonify({
        "exists": exists,
        "profile": profile_mod.profile_to_dict(prof),
        "education_levels": list(profile_mod.EDUCATION_LEVELS),
        "work_modes": list(profile_mod.WORK_MODES),
    })


@app.post("/profile")
def post_profile():
    if (refused := _not_json()):
        return refused
    body = _json_body()
    data = body.get("profile") if isinstance(body.get("profile"), dict) else body
    prof = profile_mod.profile_from_dict(data)
    errors = profile_mod.validate_profile(prof)
    if errors:
        return jsonify({"error": "; ".join(errors), "errors": errors}), 400
    path = profile_mod.save_profile(prof, PROFILE_PATH)
    return jsonify({"saved": path, "profile": profile_mod.profile_to_dict(prof)})


@app.get("/master")
def get_master():
    payload, code = _bank_payload()
    return jsonify(payload), code


@app.post("/master")
def save_master():
    if (refused := _not_json()):
        return refused
    body = _json_body()
    data = body.get("bank")
    if not isinstance(data, dict):
        return jsonify({"error": "Provide 'bank': {...} (the evidence bank as JSON)"}), 400
    bank = gen_mod.bank_from_dict(data, source_path=str(MASTER_PATH))
    gen_mod.save_bank(bank, MASTER_PATH)
    payload, code = _bank_payload()
    return jsonify(payload), code


@app.post("/master/import")
@_manual_aware
def import_master():
    """Resume upload -> the same LLM transcription as `init-master --from`.
    The file comes as base64 inside JSON (a multipart form would be a
    "simple" cross-site request). Every transcribed bullet starts unreviewed."""
    if (refused := _not_json()):
        return refused
    body = _json_body()
    if body.get("offline"):
        return jsonify({"error": "Importing a resume needs the LLM to transcribe it. Offline, "
                                 "start a blank bank in the editor instead."}), 400
    filename = str(body.get("filename") or "")
    suffix = Path(filename).suffix.lower()
    if suffix not in UPLOAD_TYPES:
        return jsonify({"error": f"Upload a {', '.join(UPLOAD_TYPES)} file (got {filename or 'no name'})."}), 400
    try:
        raw = base64.b64decode(str(body.get("content_b64") or ""), validate=True)
    except (binascii.Error, ValueError):
        return jsonify({"error": "'content_b64' is not valid base64."}), 400
    if not raw:
        return jsonify({"error": "The uploaded file is empty."}), 400
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": f"File is over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."}), 413
    if Path(MASTER_PATH).exists() and not body.get("overwrite"):
        return jsonify({"error": f"An evidence bank already exists at {MASTER_PATH}. Send "
                                 "'overwrite': true to replace it.", "exists": True}), 409

    with tempfile.TemporaryDirectory(prefix="ats_upload_") as td:
        resume_path = Path(td) / f"resume{suffix}"
        resume_path.write_bytes(raw)
        try:
            _path, notes = gen_mod.init_from_resume(
                resume_path=str(resume_path), out_path=MASTER_PATH, overwrite=True,
                **_llm_kwargs(body),
            )
        except RuntimeError as e:      # the LLM failed or transcribed nothing
            return jsonify({"error": str(e)}), 502
        except (ValueError, FileNotFoundError) as e:
            return jsonify({"error": str(e)}), 400
    payload, code = _bank_payload()
    payload["notes"] = notes
    return jsonify(payload), code


# --------------------------------------------------------------- web UI

UI_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ATS Score Checker</title>
<style>
:root{--bg:#0f1216;--panel:#171c22;--line:#2a323c;--txt:#e6ebf0;--dim:#8a94a0;
--green:#3fb96b;--yellow:#d9a53e;--red:#e05555;--blue:#4d9de0;--cyan:#3ec6c6;}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--txt)}
header{padding:18px 28px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px}
header h1{font-size:19px;margin:0;font-weight:650}
header .sub{color:var(--dim);font-size:13px}
main{max-width:1080px;margin:0 auto;padding:22px 20px 60px}
.tabs{display:flex;gap:8px;margin-bottom:18px}
.tabs button{background:var(--panel);border:1px solid var(--line);color:var(--dim);
padding:8px 18px;border-radius:8px;cursor:pointer;font-size:14px}
.tabs button.active{color:var(--txt);border-color:var(--blue);background:#1c2430}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:16px}
.panel h2{margin:0 0 12px;font-size:15px;font-weight:650;color:var(--txt)}
label{display:block;color:var(--dim);font-size:12.5px;margin:10px 0 4px}
textarea,input[type=text],input[type=url],input[type=number],select{
width:100%;background:#0d1116;color:var(--txt);border:1px solid var(--line);
border-radius:8px;padding:9px 11px;font:inherit;font-size:14px}
textarea{min-height:130px;resize:vertical;font-family:ui-monospace,Consolas,monospace;font-size:13px}
.row{display:flex;gap:14px;flex-wrap:wrap}
.row>div{flex:1;min-width:220px}
.chk{display:flex;align-items:center;gap:8px;margin:12px 2px;color:var(--dim);font-size:13.5px}
.chk input{width:auto}
button.run{background:var(--blue);color:#08121c;border:0;border-radius:8px;
padding:11px 26px;font-size:15px;font-weight:650;cursor:pointer;margin-top:14px}
button.run:disabled{opacity:.5;cursor:wait}
.scores{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:16px}
.card{flex:1;min-width:200px;background:var(--panel);border:1px solid var(--line);
border-radius:10px;padding:16px}
.card .num{font-size:34px;font-weight:700;line-height:1.1}
.card .lbl{color:var(--dim);font-size:12.5px;margin-top:2px}
.card .band{font-size:12.5px;margin-top:4px;color:var(--dim)}
.g{color:var(--green)}.y{color:var(--yellow)}.r{color:var(--red)}.d{color:var(--dim)}.c{color:var(--cyan)}
.tag{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px;margin:2px 4px 2px 0;border:1px solid var(--line)}
.tag.pass{border-color:var(--green);color:var(--green)}
.tag.warn{border-color:var(--yellow);color:var(--yellow)}
.tag.fail{border-color:var(--red);color:var(--red)}
.tag.skip{color:var(--dim)}
ul{margin:6px 0;padding-left:20px}
li{margin:4px 0}
.small{font-size:12.5px;color:var(--dim)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600}
pre{background:#0d1116;border:1px solid var(--line);border-radius:8px;padding:14px;
overflow:auto;font-size:12.5px;white-space:pre-wrap}
.blockers{background:#2a1518;border:1px solid var(--red);border-radius:10px;padding:14px 18px;margin-bottom:14px}
.blockers h3{margin:0 0 8px;color:var(--red)}
.note{color:var(--dim);font-size:13px;margin-top:10px}
.copy{background:var(--panel);border:1px solid var(--line);color:var(--txt);
padding:6px 14px;border-radius:7px;cursor:pointer;font-size:13px;margin-bottom:10px}
.hidden{display:none}
.obs{white-space:nowrap}
button.ob{background:var(--panel);border:1px solid var(--line);color:var(--dim);border-radius:6px;
  padding:2px 7px;margin:1px 2px;font-size:11.5px;cursor:pointer}
button.ob:hover{color:var(--txt);border-color:var(--blue)}
button.ob.on{color:var(--txt);border-color:var(--cyan);background:#16303a}
.err{background:#2a1518;border:1px solid var(--red);border-radius:10px;padding:12px 16px;color:#ffb0b0}
h3.sec{font-size:13.5px;margin:18px 0 6px;color:var(--txt)}
.banner{background:#14263a;border:1px solid var(--blue);border-radius:10px;padding:12px 16px;margin-bottom:16px}
.steps{display:flex;gap:10px;flex-wrap:wrap;margin:4px 0 2px}
.step{border:1px solid var(--line);border-radius:20px;padding:3px 12px;font-size:12.5px;color:var(--dim)}
.step.done{border-color:var(--green);color:var(--green)}
.step.todo{border-color:var(--yellow);color:var(--yellow)}
.tab-badge{display:inline-block;min-width:18px;padding:0 6px;margin-left:6px;border-radius:9px;
  background:var(--yellow);color:#14100a;font-size:11.5px;font-weight:700}
.role{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0}
.bullet{border-left:3px solid var(--line);padding:6px 0 6px 10px;margin:10px 0}
.bullet.unrev{border-left-color:var(--yellow)}
.bullet.rev{border-left-color:var(--green)}
.bullet textarea{min-height:58px}
.bullet .acts{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:6px}
button.mini{background:var(--panel);border:1px solid var(--line);color:var(--txt);border-radius:7px;
  padding:4px 11px;font-size:12.5px;cursor:pointer}
button.mini.ok{border-color:var(--green);color:var(--green)}
button.mini.del{color:var(--red)}
button.mini:disabled{opacity:.5;cursor:wait}
.badge{font-size:11.5px;border-radius:12px;padding:1px 9px;border:1px solid}
.badge.unrev{color:var(--yellow);border-color:var(--yellow)}
.badge.rev{color:var(--green);border-color:var(--green)}
.savebar{position:sticky;bottom:0;background:var(--panel);border-top:1px solid var(--line);
  padding:10px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.dirty{color:var(--yellow);font-size:13px}
.okmsg{color:var(--green);font-size:13px}
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);display:flex;align-items:flex-start;
  justify-content:center;padding:40px 16px;z-index:10;overflow:auto}
.overlay.hidden{display:none}
.modal{background:var(--panel);border:1px solid var(--blue);border-radius:10px;padding:18px;
  max-width:720px;width:100%}
.modal a{color:var(--blue)}
</style>
</head>
<body>
<header>
  <h1>ATS Score Checker</h1>
  <span class="sub">three layers, honestly measured &mdash; not a probability of passing</span>
</header>
<main>
  <div class="tabs">
    <button id="tab-score" class="active" onclick="showTab('score')">Score a resume</button>
    <button id="tab-tailor" onclick="showTab('tailor')">Tailor (generate)</button>
    <button id="tab-apps" onclick="showTab('apps');loadApps()">Applications</button>
    <button id="tab-setup" onclick="showTab('setup');loadSetup()">Setup<span id="setup_badge"></span></button>
  </div>
  <div id="first_run" class="banner hidden"></div>
  <div id="manual_box" class="overlay hidden"></div>

  <!-- ============ SCORE ============ -->
  <section id="sec-score">
    <div class="panel">
      <h2>Score a resume against a job description</h2>
      <label>Resume text (paste)</label>
      <textarea id="s_resume" placeholder="Paste the resume text here..."></textarea>
      <div class="row">
        <div>
          <label>... or resume path on the server</label>
          <input type="text" id="s_resume_path" placeholder="D:/resumes/cv.pdf (leave empty if pasting)">
        </div>
      </div>
      <label>Job description (paste)</label>
      <textarea id="s_jd" placeholder="Paste the JD text here..."></textarea>
      <div class="row">
        <div>
          <label>... or posting URL</label>
          <input type="url" id="s_jd_url" placeholder="https://.../job-posting (used when JD text is empty)">
        </div>
        <div>
          <label>Company</label>
          <input type="text" id="s_company" placeholder="For logging (blank = from the JD)">
        </div>
        <div>
          <label>Role</label>
          <input type="text" id="s_role" placeholder="For logging (blank = JD title)">
        </div>
      </div>
      <div class="chk"><input type="checkbox" id="s_offline"> Offline (skip both LLM layers &mdash; instant)</div>
      <div class="chk"><input type="checkbox" id="s_log"> Log this application (company/role filled from the JD if blank)</div>
      <button class="run" id="s_run" onclick="runScore()">Score</button>
      <button class="run" id="w_run" onclick="runWritingCheck(false)">Writing review (local)</button>
    </div>
    <div class="panel">
      <h2>Optional AI Writing Check &mdash; experimental</h2>
      <p class="small">Local writing feedback appears beside the scores and does not change them. GPTZero requires a separately configured API key;
      fees may apply. The full pasted resume text will be sent under GPTZero's data policies, including any personal details.
      Review and remove details from the paste box above before sending. File-path inputs cannot be sent from this button.</p>
      <div class="chk"><input type="checkbox" id="w_consent"> I consent to sending the exact text currently in the resume paste box to GPTZero.</div>
      <button class="run" id="w_external" onclick="runWritingCheck(true)">Send pasted text to GPTZero</button>
      <div id="w_out"></div>
    </div>
    <div id="s_out"></div>
  </section>

  <!-- ============ TAILOR ============ -->
  <section id="sec-tailor" class="hidden">
    <div class="panel">
      <h2>Generate a JD-tailored resume from the evidence bank</h2>
      <p class="small">Every line comes from <b>master_resume.yaml</b> (the evidence bank) &mdash; selected,
      ordered and reworded under fact verification. Nothing is invented; terms the bank can't cover
      are reported as gaps, not stuffed in.</p>
      <label>Job description (paste)</label>
      <textarea id="t_jd" placeholder="Paste the JD text here..."></textarea>
      <div class="row">
        <div>
          <label>... or posting URL</label>
          <input type="url" id="t_jd_url" placeholder="https://.../job-posting">
        </div>
        <div>
          <label>Company</label>
          <input type="text" id="t_company" placeholder="For logging (blank = from the JD)">
        </div>
        <div>
          <label>Role</label>
          <input type="text" id="t_role" placeholder="For logging (blank = JD title)">
        </div>
      </div>
      <div class="chk"><input type="checkbox" id="t_offline"> Offline (deterministic selection only, no LLM)</div>
      <div class="chk"><input type="checkbox" id="t_force"> Force generation despite candidacy blockers</div>
      <div class="chk"><input type="checkbox" id="t_log"> Log this application (company/role filled from the JD if blank)</div>
      <button class="run" id="t_run" onclick="runTailor()">Tailor</button>
    </div>
    <div id="t_out"></div>
  </section>

  <!-- ============ APPLICATIONS ============ -->
  <section id="sec-apps" class="hidden">
    <div class="panel">
      <h2>Logged applications</h2>
      <div id="apps_out"><p class="small">Loading...</p></div>
    </div>
  </section>
  <!-- ============ SETUP ============ -->
  <section id="sec-setup" class="hidden">
    <div class="panel">
      <h2>Setup</h2>
      <div class="steps" id="setup_steps"></div>
      <p class="small">Three steps, no YAML: your fixed facts (profile), your evidence bank, and a
      review of every bullet in it. Everything is saved on this machine only.</p>
    </div>

    <div class="panel">
      <h2>1. Your profile &mdash; the facts a recruiter screens on</h2>
      <p class="small">Leave anything blank that doesn't apply: its check is skipped, never guessed.</p>
      <div class="row">
        <div><label>Current / most recent title</label><input type="text" id="p_current_title"></div>
        <div><label>Years of relevant experience</label><input type="number" step="0.1" min="0" id="p_years_experience"></div>
        <div><label>Highest completed education</label><select id="p_education_level"></select></div>
      </div>
      <div class="row">
        <div><label>Location</label><input type="text" id="p_location" placeholder="City, Country"></div>
        <div><label>Open to relocation?</label>
          <select id="p_open_to_relocation"><option value="">(not set)</option>
          <option value="true">Yes</option><option value="false">No</option></select></div>
        <div><label>Notice period (days)</label><input type="number" min="0" id="p_notice_period_days"></div>
      </div>
      <label>Work modes you'd accept</label>
      <div class="chk" id="p_modes"></div>
      <div class="row">
        <div><label>Work authorisation (free text)</label>
          <input type="text" id="p_work_authorization" placeholder="e.g. Indian citizen; needs sponsorship for US roles"></div>
        <div><label>Countries you can work in without sponsorship (comma-separated)</label>
          <input type="text" id="p_work_authorized_in" placeholder="India, UAE"></div>
      </div>
      <div class="row">
        <div><label>Expected salary &mdash; min</label><input type="number" min="0" id="p_expected_salary_min"></div>
        <div><label>Expected salary &mdash; max</label><input type="number" min="0" id="p_expected_salary_max"></div>
        <div><label>Currency</label><input type="text" id="p_salary_currency" placeholder="INR, USD, AED..."></div>
      </div>
      <label>Certifications you actually hold (one per line)</label>
      <textarea id="p_certifications" style="min-height:70px"></textarea>
      <button class="run" id="p_save" onclick="saveProfile()">Save profile</button>
      <span id="p_msg"></span>
    </div>

    <div class="panel">
      <h2>2. Your evidence bank &mdash; every real bullet you've written</h2>
      <p class="small">Upload your current resume and the LLM transcribes it into the bank, verbatim.
      Every transcribed bullet starts <span class="badge unrev">unreviewed</span>: the tailor won't use it
      until you confirm it below. No LLM? Start a blank bank and type your bullets in.</p>
      <div class="row">
        <div><label>Resume file (.pdf, .docx or .txt)</label>
          <input type="file" id="m_file" accept=".pdf,.docx,.txt"></div>
      </div>
      <p class="small" id="llm_status"></p>
      <button class="run" id="m_import" onclick="importResume()">Transcribe with the LLM</button>
      <button class="copy" onclick="testLLM()">Test the LLM connection</button>
      <button class="copy" onclick="startBlank()">Start a blank bank</button>
      <div id="m_msg"></div>
    </div>

    <div class="panel" id="bank_panel">
      <h2>3. Review your evidence bank</h2>
      <div id="bank_out"><p class="small">No evidence bank yet &mdash; step 2 creates one.</p></div>
    </div>
  </section>
</main>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

function showTab(name){
  for (const t of ["score","tailor","apps","setup"]){
    $("sec-"+t).classList.toggle("hidden", t!==name);
    $("tab-"+t).classList.toggle("active", t===name);
  }
}

function bandColor(score){
  if (score===null||score===undefined) return "d";
  if (score>=80) return "g";
  if (score>=60) return "y";
  return "r";
}

async function post(url, payload){
  // copy-paste LLM mode: a 202 carries a prompt bundle; once its answer is
  // pasted, the same request goes again (later rounds may need more answers)
  for (let pass = 0; pass < 8; pass++){
    const r = await fetch(url, {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify(payload)});
    const data = await r.json().catch(()=>({}));
    if (r.status === 202 && data.manual_pending){
      await manualStep(data.manual_pending);
      continue;
    }
    if (!r.ok) throw new Error(data.error || ("HTTP "+r.status));
    return data;
  }
  throw new Error("Still waiting for answers after several copy-paste rounds.");
}

function manualStep(mp){
  return new Promise((resolve, reject) => {
    const box = $("manual_box");
    const links = (mp.links || []).map(u =>
      `<a href="${esc(u)}" target="_blank" rel="noopener noreferrer">${esc(u.replace("https://","").split("/")[0])}</a>`).join(" or ");
    box.innerHTML = `<div class="modal"><h2>Copy-paste LLM step</h2>
      <p class="small">No API key needed: ${mp.ids.length} prompt(s) for a chat AI.</p>
      <ol class="small"><li>Click <b>Copy prompt</b>.</li>
        <li>Paste it into ${links} and send it.</li>
        <li>Copy the AI's whole JSON reply, paste it below, and click <b>Submit answer</b>.</li></ol>
      <button class="copy" id="mp_copy">Copy prompt</button>
      <span id="mp_copied" class="small"></span>
      <textarea id="mp_reply" placeholder="Paste the AI's JSON reply here..."></textarea>
      <div id="mp_err"></div>
      <button class="run" id="mp_submit">Submit answer</button>
      <button class="copy" id="mp_cancel">Cancel</button></div>`;
    box.classList.remove("hidden");
    $("mp_copy").onclick = () => navigator.clipboard.writeText(mp.bundle).then(
      () => { $("mp_copied").textContent = " copied"; },
      () => { $("mp_copied").textContent = " copy failed - select the text below instead"; });
    $("mp_cancel").onclick = () => { box.classList.add("hidden"); reject(new Error("Cancelled the copy-paste step.")); };
    $("mp_submit").onclick = async () => {
      const r = await fetch("/manual/answers", {method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({reply: $("mp_reply").value, ids: mp.ids})});
      const d = await r.json().catch(()=>({}));
      if (!r.ok){ $("mp_err").innerHTML = `<div class="err">${esc(d.error || ("HTTP " + r.status))}</div>`; return; }
      box.classList.add("hidden");
      resolve(d);
    };
  });
}

function scoreCards(scores, bands){
  const items = [
    ["1. Search Visibility", scores?.search_visibility_pct, "recruiter searches this JD implies that would find you"],
    ["2. HR Screen", scores?.hr_screen_criteria_met_pct, "share of the JD's stated screening criteria you meet"],
    ["3. Manager Evidence", scores?.manager_evidence_strength_pct, "evidence-strength rubric score"],
  ];
  return `<div class="scores">` + items.map(([lbl,v,sub])=>{
    const c = bandColor(v);
    const val = (v===null||v===undefined) ? "&mdash;" : v+"/100";
    return `<div class="card"><div class="num ${c}">${val}</div>
      <div class="lbl">${lbl}</div><div class="band">${esc(sub)}</div></div>`;
  }).join("") + `</div>
  ${scores?.parse_safe === false ? `<div class="err">Parse gate FAILED &mdash; an ATS may not read this file well enough for any search to find it. Fix the parse issues first.</div>` : ""}
  <p class="small">Percentages of things measured &mdash; not probabilities of passing.</p>`;
}

async function runScore(){
  const btn = $("s_run"); btn.disabled = true; btn.textContent = "Scoring...";
  $("s_out").innerHTML = `<p class="small">Running the three layers...</p>`;
  try{
    const payload = {
      resume_text: $("s_resume").value.trim(),
      resume_path: $("s_resume_path").value.trim() || null,
      jd_text: $("s_jd").value.trim(),
      jd_url: $("s_jd_url").value.trim() || null,
      offline: $("s_offline").checked,
      log: $("s_log").checked,
      company: $("s_company").value.trim() || null,
      role: $("s_role").value.trim() || null,
    };
    const data = await post("/score", payload);
    $("s_out").innerHTML = renderScore(data);
  }catch(e){
    $("s_out").innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }finally{
    btn.disabled = false; btn.textContent = "Score";
  }
}

function renderScore(d){
  let h = scoreCards(d.scores, d.bands);
  if (d.logged_id) h += `<p class="note">Logged as application #${d.logged_id}.</p>`;

  const rec = d.recruiter_layer;
  if (rec){
    if (rec.blockers && rec.blockers.length)
      h += `<div class="blockers"><h3>Hard blockers &mdash; filtered on these first</h3>
        <ul>${rec.blockers.map(b=>`<li>${esc(b)}</li>`).join("")}</ul></div>`;
    h += `<div class="panel"><h2>HR screen &mdash; fixable by editing vs facts about you</h2>
      <p><span class="tag pass">Resume-fixable ${rec.resume_fixable_pct ?? "&mdash;"}%</span>
         <span class="tag warn">Candidacy fit ${rec.candidacy_fit_pct ?? "&mdash;"}%</span>
         <span class="tag">Top-third JD visibility ${rec.top_third_coverage ?? "&mdash;"}%</span></p>
      ${renderChecks(rec.checks)}</div>`;
    h += `<div class="panel"><h2>What HR will expect from you</h2>
      ${renderExpectations(rec.hr_expectations)}</div>`;
  }

  const vis = d.visibility_layer;
  if (vis){
    h += `<div class="panel"><h2>Layer 1 &mdash; would a recruiter's search find you?</h2>`;
    if (vis.searches && vis.searches.length)
      h += `<table><tr><th></th><th>Search</th><th>Query</th><th>Missing</th></tr>` +
        vis.searches.map(s=>`<tr><td><span class="tag ${s.matched?"pass":"fail"}">${s.matched?"HIT":"MISS"}</span></td>
          <td>${esc(s.name)}</td><td class="small">${esc(s.query)}</td>
          <td class="small">${esc(s.missing.join(", ")) || "&mdash;"}</td></tr>`).join("") + `</table>`;
    h += `<h3 class="sec">Parse gate &mdash; ${vis.parse_gate.safe ? '<span class="g">passed</span>' : '<span class="r">FAILED</span>'}</h3>
      <p>${vis.parse_gate.checks.map(c=>`<span class="tag ${c.status}" title="${esc(c.detail)}">${esc(c.name)}</span>`).join("")}</p>`;
    if (vis.notes && vis.notes.length)
      h += `<p class="small">${vis.notes.map(esc).join(" &middot; ")}</p>`;
    const llm = d.scores?.llm_fit_pct;
    h += `<p class="small">LLM fit read: ${llm ?? "not run"}${llm==null?"":"/100"} &mdash; a model's reading of meaning-level fit, shown beside visibility, never blended into it.
      Legacy composite &lsquo;ATS score&rsquo; (deprecated): ${d.scores?.ats_score ?? "&mdash;"}.</p></div>`;
  }

  const ats = d.ats_layer;
  if (ats){
    h += `<div class="panel"><h2>JD keyword coverage and parsing warnings</h2>
      <h3 class="sec">Matched (${ats.keywords.matched.length})</h3>
      <p>${ats.keywords.matched.map(k=>`<span class="tag pass">${esc(k)}</span>`).join("") || '<span class="d">none</span>'}</p>
      <h3 class="sec">Missing (${ats.keywords.missing.length})</h3>
      <p>${ats.keywords.missing.map(k=>`<span class="tag fail">${esc(k)}</span>`).join("") || '<span class="d">none</span>'}</p>
      ${ats.formatting.warnings && ats.formatting.warnings.length ?
        `<h3 class="sec">Parsing warnings</h3>
         <ul>${ats.formatting.warnings.map(w=>`<li>${esc(w)}</li>`).join("")}</ul>` : ""}`;
    if (ats.semantic && ats.semantic.available){
      if (ats.semantic.reworded_matches && ats.semantic.reworded_matches.length)
        h += `<h3 class="sec">Experience you have but word differently</h3>
          <ul>${ats.semantic.reworded_matches.map(s=>`<li>${esc(s)}</li>`).join("")}</ul>`;
      if (ats.semantic.gaps && ats.semantic.gaps.length)
        h += `<h3 class="sec">Semantic gaps</h3>
          <ul>${ats.semantic.gaps.map(s=>`<li>${esc(s)}</li>`).join("")}</ul>`;
      if (ats.semantic.recommendation)
        h += `<h3 class="sec">Recommendation</h3><p>${esc(ats.semantic.recommendation)}</p>`;
    }
    h += `</div>`;
  }

  const mgr = d.manager_layer;
  if (mgr && mgr.available){
    h += `<div class="panel"><h2>Layer 3 &mdash; manager evidence rubric</h2><table>
      <tr><th>Dimension</th><th>Score</th></tr>
      ${Object.entries(mgr.dimensions).map(([k,v])=>
        `<tr><td>${esc(k.replace(/_/g," "))}</td>
         <td class="${bandColor(v)}">${v === null ? "not scored" : v + "/100"}</td></tr>`).join("")}</table>`;
    if (mgr.weak_bullets && mgr.weak_bullets.length)
      h += `<h3 class="sec">Weakest bullets, with rewrites</h3><ul>` +
        mgr.weak_bullets.map(w=>`<li>&#10007; ${esc(w.bullet)}<br>
          <span class="small">${esc(w.problem)}</span><br>
          <span class="g">&rarr; ${esc(w.rewrite)}</span></li>`).join("") + `</ul>`;
    if (mgr.interview_risks && mgr.interview_risks.length)
      h += `<h3 class="sec">Claims a manager would probe</h3>
        <ul>${mgr.interview_risks.map(r=>`<li>${esc(r)}</li>`).join("")}</ul>`;
    if (mgr.verdict) h += `<p class="note">${esc(mgr.verdict)}</p>`;
    h += `</div>`;
  }

  h += renderWritingReview(d.writing_review);
  if (d.notes && d.notes.length)
    h += `<p class="note">${d.notes.map(esc).join(" &middot; ")}</p>`;
  return h;
}

function renderWritingReview(d){
  if (!d) return '';
  const findings = d.findings || [];
  const detector = d.detector || {};
  return `<div class="panel"><h2>Writing Review / AI Writing Check</h2>
    <p class="small">${esc(d.limitation || '')}</p>
    ${findings.length ? '<ul>'+findings.map(f=>`<li>Line ${esc(f.line)}: ${esc(f.excerpt)}<br><span class="small">${esc(f.explanation)}</span></li>`).join('')+'</ul>' : '<p>No issues found by local rules; this does not establish authorship.</p>'}
    <p>External detector: ${esc(detector.status || 'not_run')} &mdash; ${esc(detector.message || '')}</p>
    ${detector.vendor_label ? `<p>${esc(detector.provider)} reports: ${esc(detector.vendor_label)}. Model version: ${esc(detector.returned_model_version || 'unknown')}.</p>` : ''}
    ${detector.class_probabilities ? `<details><summary>Provider confidence details</summary><p>${esc(detector.score_semantics)}</p><pre>${esc(JSON.stringify(detector.class_probabilities,null,2))}</pre></details>` : ''}
    ${(d.extraction_warnings || []).map(w=>`<p class="small">${esc(w)}</p>`).join('')}
  </div>`;
}

async function runWritingCheck(external){
  const text = $('s_resume').value.trim();
  const path = $('s_resume_path').value.trim();
  if (external && (!text || !$('w_consent').checked)){
    $('w_out').textContent = 'Paste the text you wish to send and explicitly consent first.';
    return;
  }
  const button = $(external ? 'w_external' : 'w_run'); button.disabled = true;
  $('w_consent').checked = false;
  try{
    const result = await post('/writing-check', {resume_text:text || null,
      resume_path:external ? null : path || null, provider:external ? 'gptzero' : 'local', consent:external});
    $('w_out').innerHTML = renderWritingReview(result);
  }catch(e){ $('w_out').textContent = e.message; }
  finally{ button.disabled = false; }
}

function renderChecks(checks){
  if (!checks || !checks.length) return "";
  const mark = {pass:"PASS",warn:"WARN",fail:"FAIL",skipped:"SKIP"};
  return `<table><tr><th></th><th>Type</th><th>Check</th><th>Detail</th></tr>` +
    checks.map(c=>`<tr><td><span class="tag ${c.status}">${mark[c.status]||c.status}</span></td>
      <td class="small">${c.category==="resume"?"RESUME":"YOU"}</td>
      <td>${esc(c.name)}</td>
      <td class="small">${esc(c.detail)}${c.jd_citation?`<br><span class="d">JD: &ldquo;${esc(c.jd_citation.slice(0,110))}&rdquo;</span>`:""}</td></tr>`).join("") +
    `</table><p class="small">RESUME = fixable by editing &middot; YOU = a fact about you; go in knowing it.</p>`;
}

function renderExpectations(exps){
  if (!exps || !exps.length) return "";
  const derived = exps.filter(e=>e.source==="derived");
  const standard = exps.filter(e=>e.source!=="derived");
  let h = "";
  if (derived.length)
    h += `<h3 class="sec">Specific to your application</h3><ul>` +
      derived.map(e=>`<li><b class="y">${esc(e.topic)}</b> &mdash; <span class="small">${esc(e.why)}</span><br>
        <span class="g">Prepare:</span> ${esc(e.prepare)}</li>`).join("") + `</ul>`;
  if (standard.length)
    h += `<h3 class="sec">Asked on virtually every screen</h3><ul>` +
      standard.map(e=>`<li><b class="c">${esc(e.topic)}</b> &mdash; <span class="small">${esc(e.why)}</span><br>
        <span class="g">Prepare:</span> ${esc(e.prepare)}</li>`).join("") + `</ul>`;
  return h;
}

async function runTailor(){
  const btn = $("t_run"); btn.disabled = true; btn.textContent = "Tailoring...";
  $("t_out").innerHTML = `<p class="small">Selecting evidence, verifying rewordings...</p>`;
  try{
    const payload = {
      jd_text: $("t_jd").value.trim(),
      jd_url: $("t_jd_url").value.trim() || null,
      offline: $("t_offline").checked,
      force: $("t_force").checked,
      log: $("t_log").checked,
      company: $("t_company").value.trim() || null,
      role: $("t_role").value.trim() || null,
    };
    const data = await post("/tailor", payload);
    $("t_out").innerHTML = renderTailor(data);
  }catch(e){
    $("t_out").innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }finally{
    btn.disabled = false; btn.textContent = "Tailor";
  }
}

function renderTailor(d){
  let h = "";
  if (d.blocked){
    h += `<div class="blockers"><h3>No-apply gate: candidacy blockers</h3>
      <ul>${d.candidacy_blockers.map(b=>`<li>${esc(b)}</li>`).join("")}</ul>
      <p class="small">These are facts about you, not the resume &mdash; no rewrite fixes them.
      Re-run with &ldquo;Force&rdquo; to generate anyway (practice, or because you think the gate is wrong).</p></div>`;
  }
  if (d.scores) h += scoreCards(d.scores, null);
  if (d.honest_gaps && d.honest_gaps.length)
    h += `<div class="panel"><h2>Honest gaps &mdash; JD terms your evidence bank doesn't cover</h2>
      <p>${d.honest_gaps.map(g=>`<span class="tag fail">${esc(g)}</span>`).join("")}</p>
      <p class="small">These are NOT added &mdash; a keyword you can't defend is an interview trap.
      Gain the experience, then add the bullet to the bank.</p></div>`;
  if (d.rewordings && d.rewordings.length)
    h += `<div class="panel"><h2>Verified rewordings applied</h2><ul>` +
      d.rewordings.map(r=>`<li><span class="c">${esc(r.reason)}</span><br>
        <span class="d">&minus; ${esc(r.original)}</span><br>
        <span class="g">+ ${esc(r.rewrite)}</span></li>`).join("") + `</ul></div>`;
  if (d.suggestions && d.suggestions.length)
    h += `<div class="panel"><h2>Suggested rewrites &mdash; need your yes, NOT in the resume</h2><ul>` +
      d.suggestions.map(r=>`<li><span class="c">${esc(r.reason)}</span><br>
        <span class="d">&minus; ${esc(r.original)}</span><br>
        <span class="g">? ${esc(r.rewrite)}</span></li>`).join("") + `</ul>
      <p class="small">These can change what a bullet claims. If one is true, edit that bullet in
      the Setup tab's evidence bank.</p></div>`;
  if (d.rejected_rewrites && d.rejected_rewrites.length)
    h += `<div class="panel"><h2>Suggested but rejected (failed fact verification)</h2><ul>` +
      d.rejected_rewrites.map(r=>`<li><span class="d">${esc(r.suggested)}</span><br>
        <span class="r">rejected: ${esc(r.reason)}</span></li>`).join("") + `</ul></div>`;
  if (d.resume_text){
    h += `<div class="panel"><h2>Generated resume &mdash; every line traces to an evidence-bank bullet</h2>
      <button class="copy" onclick="copyResume()">Copy text</button>
      <button class="copy" onclick="downloadResume()">Download .txt</button>
      <pre id="t_resume_pre">${esc(d.resume_text)}</pre></div>`;
  }
  if (d.notes && d.notes.length)
    h += `<p class="note">${d.notes.map(esc).join(" &middot; ")}</p>`;
  if (d.logged_id) h += `<p class="note">Logged as application #${d.logged_id}.</p>`;
  return h;
}

function copyResume(){
  const text = $("t_resume_pre").textContent;
  navigator.clipboard.writeText(text).then(()=>alert("Resume copied to clipboard."));
}
function downloadResume(){
  const text = $("t_resume_pre").textContent;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], {type:"text/plain"}));
  a.download = "tailored_resume.txt"; a.click();
}

// One click records an outcome. Short labels; the full name is the tooltip.
const OUTCOME_BUTTONS = [
  ["recruiter_call","Call"], ["interview","Interview"], ["offer","Offer"],
  ["rejected_auto","Rej (auto)"], ["rejected_screen","Rej (screen)"], ["ghosted","Ghosted"],
];

async function setOutcome(id, status){
  try{
    await post("/outcome", {id: id, status: status});
    await loadApps();
  }catch(e){
    alert("Could not update #" + id + ": " + e.message);
  }
}

function outcomeButtons(a){
  return OUTCOME_BUTTONS.map(([st,lbl])=>
    `<button class="ob${a.outcome===st?" on":""}" title="${esc(st)}"
      onclick="setOutcome(${Number(a.id)}, '${st}')">${esc(lbl)}</button>`).join("");
}

async function loadApps(){
  try{
    const r = await fetch("/applications?limit=100");
    const apps = await r.json();
    if (!apps.length){
      $("apps_out").innerHTML = `<p class="small">No applications logged yet. Use the Score or Tailor tabs with &ldquo;log&rdquo; checked.</p>`;
      return;
    }
    $("apps_out").innerHTML = `<table><tr><th>#</th><th>Applied</th><th>Company</th><th>Role</th>
      <th>Visibility</th><th>HR</th><th>MGR</th><th>Outcome</th><th>Record what happened</th></tr>` +
      apps.map(a=>`<tr><td>${a.id}</td><td class="small">${esc(a.applied_date)}</td>
        <td>${esc(a.company)}</td><td>${esc(a.role)}</td>
        <td>${a.visibility_score ?? "&mdash;"}</td><td>${a.recruiter_score ?? "&mdash;"}</td>
        <td>${a.manager_score ?? "&mdash;"}</td><td>${esc(a.outcome)}</td>
        <td class="obs">${outcomeButtons(a)}</td></tr>`).join("") +
      `</table><p class="small">Click what happened when you hear back. Pending applications older than 45 days
      become ghosted when you run <code>cli.py log stats</code> or <code>log reap-ghosts</code>.</p>`;
  }catch(e){
    $("apps_out").innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
}

// ================================================================ setup
const NL = String.fromCharCode(10);
let BANK = null;          // the bank being edited (JSON shape of master_resume.yaml)
let BANK_DIRTY = false;
let ONLY_UNREVIEWED = false;
let PROFILE_META = {education_levels: [], work_modes: []};

window.addEventListener("beforeunload", (e) => {
  if (BANK_DIRTY){ e.preventDefault(); e.returnValue = ""; }
});

async function getJSON(url){
  const r = await fetch(url);
  const data = await r.json().catch(()=>({}));
  return {ok: r.ok, status: r.status, data};
}

function setBadge(n){
  $("setup_badge").innerHTML = n ? `<span class="tab-badge" title="unreviewed bullets">${n}</span>` : "";
}

async function refreshStatus(){
  const st = (await getJSON("/setup/status")).data;
  const steps = [
    ["1. Profile", st.profile && st.profile.exists],
    ["2. Evidence bank", st.bank && st.bank.exists],
    ["3. Bullets reviewed", st.bank && st.bank.exists && !st.bank.unreviewed && !st.bank.error],
  ];
  $("setup_steps").innerHTML = steps.map(([l, ok]) =>
    `<span class="step ${ok ? "done" : "todo"}">${ok ? "&#10003; " : ""}${esc(l)}</span>`).join("");
  const llm = st.llm || {};
  $("llm_status").innerHTML = `LLM: <b>${esc(llm.provider)}</b> &middot; ${esc(llm.model)} &middot; ${esc(llm.base_url)}
    &middot; API key ${llm.api_key_set ? "set" : "not set"} (configured in <code>.env</code>).`;
  setBadge(st.bank && st.bank.unreviewed);
  return st;
}

async function initFirstRun(){
  try{
    const st = await refreshStatus();
    if (st.first_run){
      const missing = [];
      if (!st.profile.exists) missing.push("your profile");
      if (!st.bank.exists) missing.push("your evidence bank");
      $("first_run").innerHTML = `<b>Welcome &mdash; first run.</b> Set up ${missing.join(" and ")} in a
        couple of minutes, no YAML editing. Scoring works without them, but the recruiter checks
        and the tailor need them.`;
      $("first_run").classList.remove("hidden");
      showTab("setup");
      loadSetup();
    }
  }catch(e){ /* status is a nicety; the rest of the UI works without it */ }
}

async function loadSetup(){
  await Promise.all([loadProfile(), loadBank(), refreshStatus()]);
}

// ---- profile form
async function loadProfile(){
  const {data} = await getJSON("/profile");
  PROFILE_META = data;
  const p = data.profile || {};
  $("p_education_level").innerHTML = `<option value="">(not set)</option>` +
    (data.education_levels || []).map(l => `<option value="${esc(l)}">${esc(l.replace("_", " "))}</option>`).join("");
  $("p_modes").innerHTML = (data.work_modes || []).map(m =>
    `<label style="display:inline-flex;gap:6px;margin:0 14px 0 0"><input type="checkbox" class="p_mode" value="${esc(m)}"
      ${(p.acceptable_work_modes || []).includes(m) ? "checked" : ""}> ${esc(m)}</label>`).join("");
  for (const k of ["current_title","years_experience","education_level","location","notice_period_days",
                   "work_authorization","expected_salary_min","expected_salary_max","salary_currency"])
    $("p_"+k).value = p[k] ?? "";
  $("p_open_to_relocation").value = p.open_to_relocation === true ? "true" : p.open_to_relocation === false ? "false" : "";
  $("p_work_authorized_in").value = (p.work_authorized_in || []).join(", ");
  $("p_certifications").value = (p.certifications || []).join(NL);
  if (data.error) $("p_msg").innerHTML = `<div class="err">${esc(data.error)}</div>`;
}

async function saveProfile(){
  const btn = $("p_save"); btn.disabled = true;
  const v = (k) => $("p_"+k).value.trim();
  const num = (k) => v(k) === "" ? null : Number(v(k));
  const profile = {
    current_title: v("current_title"), years_experience: num("years_experience"),
    education_level: v("education_level"), location: v("location"),
    open_to_relocation: v("open_to_relocation") === "" ? null : v("open_to_relocation") === "true",
    notice_period_days: num("notice_period_days"),
    acceptable_work_modes: [...document.querySelectorAll(".p_mode:checked")].map(x => x.value),
    work_authorization: v("work_authorization"),
    work_authorized_in: v("work_authorized_in").split(",").map(x => x.trim()).filter(Boolean),
    expected_salary_min: num("expected_salary_min"), expected_salary_max: num("expected_salary_max"),
    salary_currency: v("salary_currency"),
    certifications: $("p_certifications").value.split(NL).map(x => x.trim()).filter(Boolean),
  };
  try{
    const d = await post("/profile", {profile});
    $("p_msg").innerHTML = ` <span class="okmsg">Saved to ${esc(d.saved)}.</span>`;
    refreshStatus();
  }catch(e){
    $("p_msg").innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }finally{ btn.disabled = false; }
}

// ---- LLM + import
async function testLLM(){
  $("m_msg").innerHTML = `<p class="small">Sending one test request...</p>`;
  try{
    const d = await post("/setup/test-llm", {});
    $("m_msg").innerHTML = d.ok ? `<p class="okmsg">LLM OK &mdash; ${esc(d.message)}</p>`
      : `<div class="err">LLM not reachable: ${esc(d.message)}. You can still start a blank bank.</div>`;
  }catch(e){ $("m_msg").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}

function readFileB64(file){
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result).split(",").pop());
    fr.onerror = () => reject(new Error("Could not read the file."));
    fr.readAsDataURL(file);
  });
}

async function importResume(){
  const file = $("m_file").files[0];
  if (!file){ $("m_msg").innerHTML = `<div class="err">Choose a resume file first.</div>`; return; }
  const st = await refreshStatus();
  let overwrite = false;
  if (st.bank && st.bank.exists){
    if (!confirm("You already have an evidence bank. Replace it with a fresh transcription of this resume? " +
                 "Bullets you added or confirmed will be lost.")) return;
    overwrite = true;
  }
  const btn = $("m_import"); btn.disabled = true; btn.textContent = "Transcribing...";
  $("m_msg").innerHTML = `<p class="small">The LLM is transcribing your resume (this can take a minute)...</p>`;
  try{
    const content_b64 = await readFileB64(file);
    const d = await post("/master/import", {filename: file.name, content_b64, overwrite});
    $("m_msg").innerHTML = `<p class="okmsg">Transcribed ${d.bullets} bullet(s). Every one starts unreviewed &mdash;
      check each against what you actually did, then confirm it.</p>`;
    setBank(d);
  }catch(e){
    $("m_msg").innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }finally{
    btn.disabled = false; btn.textContent = "Transcribe with the LLM";
    refreshStatus();
  }
}

function startBlank(){
  if (BANK && !confirm("Discard the bank shown below and start blank? Nothing is saved until you press Save.")) return;
  BANK = {contact: {name:"", email:"", phone:"", location:"", linkedin:"", github:"", website:""},
          headline: "", roles: [{company:"", title:"", start:"", end:"present", location:"",
          bullets: [{text:"", skills:[]}]}], skills_extra: [], education: [], certifications: []};
  markDirty(); renderBank([]);
}

// ---- bank editor
async function loadBank(){
  if (BANK_DIRTY) return;    // don't clobber unsaved edits
  const {status, data} = await getJSON("/master");
  if (status === 404){ BANK = null; $("bank_out").innerHTML = `<p class="small">No evidence bank yet &mdash; step 2 creates one.</p>`; return; }
  if (data.error){ $("bank_out").innerHTML = `<div class="err">${esc(data.error)}</div>`; return; }
  setBank(data);
}

function setBank(d){
  BANK = d.bank; BANK_DIRTY = false;
  renderBank(d.problems || []);
}

function markDirty(){
  BANK_DIRTY = true;
  const m = $("bank_state");
  if (m) m.innerHTML = `<span class="dirty">Unsaved changes</span>`;
}

function setPath(path, value){
  let o = BANK;
  for (let i = 0; i < path.length - 1; i++) o = o[path[i]];
  o[path[path.length - 1]] = value;
  markDirty();
}

const csv = (s) => s.split(",").map(x => x.trim().toLowerCase()).filter(Boolean);

function field(label, path, value, attrs){
  return `<div><label>${esc(label)}</label><input type="text" value="${esc(value)}" ${attrs || ""}
    oninput='setPath(${JSON.stringify(path)}, this.value)'></div>`;
}

function counts(){
  let total = 0, unrev = 0;
  for (const r of BANK.roles) for (const b of r.bullets){ total++; if (b.reviewed === false) unrev++; }
  return {total, unrev};
}

function renderBank(problems){
  if (!BANK){ return; }
  const c = counts();
  const ct = BANK.contact;
  let h = `<p class="small">${c.total} bullet(s); <b class="${c.unrev ? "y" : "g"}">${c.unrev} unreviewed</b>
    &mdash; the tailor only uses bullets you've confirmed. Confirm a bullet only if it's true and you could
    defend it in an interview; edit or delete the rest.</p>
    <div class="chk"><input type="checkbox" id="only_unrev" ${ONLY_UNREVIEWED ? "checked" : ""}
      onchange="ONLY_UNREVIEWED=this.checked;renderBank([])"> Show unreviewed bullets only</div>`;
  if (problems && problems.length)
    h += `<p class="note">${problems.map(esc).join(" &middot; ")}</p>`;
  h += `<h3 class="sec">Contact &amp; headline</h3><div class="row">` +
    field("Name", ["contact","name"], ct.name) + field("Email", ["contact","email"], ct.email) +
    field("Phone", ["contact","phone"], ct.phone) + `</div><div class="row">` +
    field("Location", ["contact","location"], ct.location) + field("LinkedIn", ["contact","linkedin"], ct.linkedin) +
    field("Headline (your real title)", ["headline"], BANK.headline) + `</div>`;
  h += `<h3 class="sec">Roles</h3>`;
  BANK.roles.forEach((r, i) => {
    h += `<div class="role"><div class="row">` +
      field("Company", ["roles",i,"company"], r.company) + field("Title", ["roles",i,"title"], r.title) + `</div>
      <div class="row">` + field("Start (YYYY-MM)", ["roles",i,"start"], r.start) +
      field("End (YYYY-MM or present)", ["roles",i,"end"], r.end) +
      field("Location", ["roles",i,"location"], r.location) + `</div>`;
    r.bullets.forEach((b, j) => {
      const unrev = b.reviewed === false;
      if (ONLY_UNREVIEWED && !unrev) return;
      h += `<div class="bullet ${unrev ? "unrev" : "rev"}" id="b_${i}_${j}">
        <textarea oninput='setPath(["roles",${i},"bullets",${j},"text"], this.value)'>${esc(b.text)}</textarea>
        <div class="acts">
          <span class="badge ${unrev ? "unrev" : "rev"}">${unrev ? "unreviewed" : "confirmed"}</span>
          <input type="text" style="flex:1;min-width:200px" value="${esc((b.skills || []).join(", "))}"
            placeholder="skills it shows, comma-separated"
            oninput='setPath(["roles",${i},"bullets",${j},"skills"], csv(this.value))'>
          ${unrev ? `<button class="mini ok" onclick="confirmBullet(${i},${j})">Confirm &mdash; this is true</button>`
                  : `<button class="mini" onclick="unconfirmBullet(${i},${j})">Mark unreviewed</button>`}
          <button class="mini del" onclick="deleteBullet(${i},${j})">Delete</button>
        </div></div>`;
    });
    h += `<button class="mini" onclick="addBullet(${i})">+ Add bullet</button>
      <button class="mini del" onclick="deleteRole(${i})">Delete role</button></div>`;
  });
  h += `<button class="mini" onclick="addRole()">+ Add role</button>`;
  h += `<h3 class="sec">Other</h3><div class="row">
    <div><label>Skills with no specific bullet (comma-separated)</label>
      <input type="text" value="${esc((BANK.skills_extra || []).join(", "))}"
        oninput='setPath(["skills_extra"], csv(this.value))'></div>
    <div><label>Certifications (comma-separated)</label>
      <input type="text" value="${esc((BANK.certifications || []).join(", "))}"
        oninput='setPath(["certifications"], this.value.split(",").map(x=>x.trim()).filter(Boolean))'></div></div>`;
  (BANK.education || []).forEach((e, k) => {
    h += `<div class="row">` + field("Degree", ["education",k,"degree"], e.degree) +
      field("Institution", ["education",k,"institution"], e.institution) +
      field("Year", ["education",k,"year"], e.year) + `</div>`;
  });
  h += `<button class="mini" onclick="addEducation()">+ Add education</button>`;
  h += `<div class="savebar"><button class="run" id="bank_save" onclick="saveBank()">Save bank</button>
    <span id="bank_state">${BANK_DIRTY ? `<span class="dirty">Unsaved changes</span>` : ""}</span></div>`;
  $("bank_out").innerHTML = h;
  setBadge(c.unrev);
}

function confirmBullet(i, j){
  const b = BANK.roles[i].bullets[j];
  if (!b.text.trim()){ alert("An empty bullet can't be confirmed — write it or delete it."); return; }
  b.reviewed = true; markDirty(); renderBank([]);
}
function unconfirmBullet(i, j){ BANK.roles[i].bullets[j].reviewed = false; markDirty(); renderBank([]); }
function deleteBullet(i, j){ BANK.roles[i].bullets.splice(j, 1); markDirty(); renderBank([]); }
// a bullet you type yourself is your own words, so it starts confirmed
function addBullet(i){ BANK.roles[i].bullets.push({text:"", skills:[]}); markDirty(); renderBank([]); }
function addRole(){ BANK.roles.push({company:"", title:"", start:"", end:"", location:"", bullets:[{text:"", skills:[]}]}); markDirty(); renderBank([]); }
function deleteRole(i){
  if (!confirm("Delete this role and all its bullets?")) return;
  BANK.roles.splice(i, 1); markDirty(); renderBank([]);
}
function addEducation(){ (BANK.education = BANK.education || []).push({degree:"", institution:"", year:""}); markDirty(); renderBank([]); }

async function saveBank(){
  const btn = $("bank_save"); btn.disabled = true;
  try{
    const d = await post("/master", {bank: BANK});
    setBank(d);
    $("bank_state").innerHTML = `<span class="okmsg">Saved to ${esc(d.path)}.</span>`;
    refreshStatus();
  }catch(e){
    $("bank_state").innerHTML = `<span class="err">${esc(e.message)}</span>`;
  }finally{ const b2 = $("bank_save"); if (b2) b2.disabled = false; }
}

initFirstRun();
</script>
</body>
</html>
"""


@app.get("/")
def ui():
    return UI_PAGE


def main():
    global PROFILE_PATH, DB_PATH, MASTER_PATH, PROVIDER_OVERRIDE
    parser = argparse.ArgumentParser(description="Run the ATS checker as a local HTTP API")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--profile", default=profile_mod.DEFAULT_PROFILE_PATH)
    parser.add_argument("--master", default=gen_mod.DEFAULT_MASTER_PATH,
                        help="Default evidence bank path for /tailor")
    parser.add_argument("--db", default=applog.DEFAULT_DB)
    parser.add_argument("--llm", choices=["configured", "manual"], default="configured",
                        help="manual = copy-paste mode: no API key, paste prompts into Claude or "
                             "ChatGPT from the web UI")
    args = parser.parse_args()
    PROFILE_PATH, DB_PATH, MASTER_PATH = args.profile, args.db, args.master
    PROVIDER_OVERRIDE = "manual" if args.llm == "manual" else None
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
