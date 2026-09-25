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
    POST /log            record an application
    POST /outcome        update an application's outcome
    GET  /applications   list logged applications
    GET  /stats          conversion by score band + predictiveness (once enough outcomes)

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
      "log": true, "company": "...", "role": "..."
    }

Returns the same JSON shape as `cli.py score --json`: read
`scores.ats_score`, `scores.hr_screen_criteria_met_pct`,
`scores.manager_evidence_strength_pct`, plus `recruiter_layer.blockers` and
`manager_layer.weak_bullets` for the actionable parts.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

from ats_checker import applog
from ats_checker import generator as gen_mod
from ats_checker import llm_client as ollama_client
from ats_checker import profile as profile_mod
from ats_checker.jd_fetch import fetch_jd_url
from ats_checker.scorer import run_full_check

app = Flask(__name__)
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
        "provider": body.get("provider"),
    }


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


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/score")
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
        )
    except Exception as e:  # noqa: BLE001 — surface parsing/scoring errors to the caller
        return jsonify({"error": str(e)}), 400

    payload = result.to_dict()

    if body.get("log"):
        if not (body.get("company") and body.get("role")):
            return jsonify({"error": "'log': true needs 'company' and 'role'"}), 400
        payload["logged_id"] = applog.log_application(
            company=body.get("company", ""),
            role=body.get("role", ""),
            ats_score=result.ats_score,
            recruiter_score=result.recruiter_score,
            manager_score=result.manager_score,
            jd_text=jd_text,
            resume_version=body.get("resume_version", resume_path or "inline"),
            days_after_posting=body.get("days_after_posting"),
            db_path=DB_PATH,
        )

    return jsonify(payload)


@app.post("/tailor")
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
        "notes": result.notes,
        "used_llm": result.used_llm,
        "scores": (
            {
                "ats_score": rep.ats_score,
                "hr_screen_criteria_met_pct": rep.recruiter_score,
                "hr_resume_fixable_pct": rep.recruiter_result.resume_pct if rep.recruiter_result else None,
                "hr_candidacy_fit_pct": rep.recruiter_result.candidacy_pct if rep.recruiter_result else None,
                "manager_evidence_strength_pct": rep.manager_score,
            } if rep else None
        ),
    }

    if body.get("log") and not result.blocked and rep is not None:
        if not (body.get("company") and body.get("role")):
            return jsonify({"error": "'log': true needs 'company' and 'role'"}), 400
        payload["logged_id"] = applog.log_application(
            company=body.get("company", ""),
            role=body.get("role", ""),
            ats_score=rep.ats_score,
            recruiter_score=rep.recruiter_score,
            manager_score=rep.manager_score,
            jd_text=jd_text,
            resume_version="tailored:inline",
            days_after_posting=body.get("days_after_posting"),
            db_path=DB_PATH,
        )

    return jsonify(payload)


@app.post("/log")
def log():
    body = _json_body()
    if not body.get("company") or not body.get("role"):
        return jsonify({"error": "Provide 'company' and 'role'"}), 400
    app_id = applog.log_application(
        company=body["company"],
        role=body["role"],
        ats_score=body.get("ats_score"),
        recruiter_score=body.get("recruiter_score"),
        manager_score=body.get("manager_score"),
        jd_text=body.get("jd_text", ""),
        resume_version=body.get("resume_version", ""),
        days_after_posting=body.get("days_after_posting"),
        notes=body.get("notes", ""),
        db_path=DB_PATH,
    )
    return jsonify({"id": app_id})


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
    return jsonify(applog.conversion_stats(db_path=DB_PATH, min_resolved=min_resolved))


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
.err{background:#2a1518;border:1px solid var(--red);border-radius:10px;padding:12px 16px;color:#ffb0b0}
h3.sec{font-size:13.5px;margin:18px 0 6px;color:var(--txt)}
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
  </div>

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
          <input type="text" id="s_company" placeholder="For logging (optional)">
        </div>
        <div>
          <label>Role</label>
          <input type="text" id="s_role" placeholder="For logging (optional)">
        </div>
      </div>
      <div class="chk"><input type="checkbox" id="s_offline"> Offline (skip both LLM layers &mdash; instant)</div>
      <div class="chk"><input type="checkbox" id="s_log"> Log this application (needs company + role)</div>
      <button class="run" id="s_run" onclick="runScore()">Score</button>
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
          <input type="text" id="t_company" placeholder="For logging (optional)">
        </div>
        <div>
          <label>Role</label>
          <input type="text" id="t_role" placeholder="For logging (optional)">
        </div>
      </div>
      <div class="chk"><input type="checkbox" id="t_offline"> Offline (deterministic selection only, no LLM)</div>
      <div class="chk"><input type="checkbox" id="t_force"> Force generation despite candidacy blockers</div>
      <div class="chk"><input type="checkbox" id="t_log"> Log this application (needs company + role)</div>
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
</main>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

function showTab(name){
  for (const t of ["score","tailor","apps"]){
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
  const r = await fetch(url, {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify(payload)});
  const data = await r.json().catch(()=>({}));
  if (!r.ok) throw new Error(data.error || ("HTTP "+r.status));
  return data;
}

function scoreCards(scores, bands){
  const items = [
    ["1. ATS", scores?.ats_score, "machine-filter criteria met (parse+keyword+semantic)"],
    ["2. HR Screen", scores?.hr_screen_criteria_met_pct, "share of the JD's stated screening criteria you meet"],
    ["3. Manager Evidence", scores?.manager_evidence_strength_pct, "evidence-strength rubric score"],
  ];
  return `<div class="scores">` + items.map(([lbl,v,sub])=>{
    const c = bandColor(v);
    const val = (v===null||v===undefined) ? "&mdash;" : v+"/100";
    return `<div class="card"><div class="num ${c}">${val}</div>
      <div class="lbl">${lbl}</div><div class="band">${esc(sub)}</div></div>`;
  }).join("") + `</div>
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
    if (payload.log && (!payload.company || !payload.role))
      throw new Error("Logging needs company and role filled in.");
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

  const ats = d.ats_layer;
  if (ats){
    h += `<div class="panel"><h2>Layer 1 &mdash; keywords and parsing</h2>
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
         <td class="${bandColor(v)}">${v}/100</td></tr>`).join("")}</table>`;
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

  if (d.notes && d.notes.length)
    h += `<p class="note">${d.notes.map(esc).join(" &middot; ")}</p>`;
  return h;
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
    if (payload.log && (!payload.company || !payload.role))
      throw new Error("Logging needs company and role filled in.");
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

async function loadApps(){
  try{
    const r = await fetch("/applications?limit=100");
    const apps = await r.json();
    if (!apps.length){
      $("apps_out").innerHTML = `<p class="small">No applications logged yet. Use the Score or Tailor tabs with &ldquo;log&rdquo; checked.</p>`;
      return;
    }
    $("apps_out").innerHTML = `<table><tr><th>#</th><th>Applied</th><th>Company</th><th>Role</th>
      <th>ATS</th><th>HR</th><th>MGR</th><th>Outcome</th></tr>` +
      apps.map(a=>`<tr><td>${a.id}</td><td class="small">${esc(a.applied_date)}</td>
        <td>${esc(a.company)}</td><td>${esc(a.role)}</td>
        <td>${a.ats_score ?? "&mdash;"}</td><td>${a.recruiter_score ?? "&mdash;"}</td>
        <td>${a.manager_score ?? "&mdash;"}</td><td>${esc(a.outcome)}</td></tr>`).join("") +
      `</table><p class="small">Update outcomes with the API: POST /outcome {"id": N, "status": "recruiter_call|interview|offer|..."}</p>`;
  }catch(e){
    $("apps_out").innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
}
</script>
</body>
</html>
"""


@app.get("/")
def ui():
    return UI_PAGE


def main():
    global PROFILE_PATH, DB_PATH, MASTER_PATH
    parser = argparse.ArgumentParser(description="Run the ATS checker as a local HTTP API")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--profile", default=profile_mod.DEFAULT_PROFILE_PATH)
    parser.add_argument("--master", default=gen_mod.DEFAULT_MASTER_PATH,
                        help="Default evidence bank path for /tailor")
    parser.add_argument("--db", default=applog.DEFAULT_DB)
    args = parser.parse_args()
    PROFILE_PATH, DB_PATH, MASTER_PATH = args.profile, args.db, args.master
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()