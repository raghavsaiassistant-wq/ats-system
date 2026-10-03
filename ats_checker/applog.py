"""Application log — the piece that turns heuristic scores into real ones.

Right now the three layers are heuristics. They become calibrated
probabilities only when there's labelled outcome data to fit against:
application -> what actually happened. So every scored application can be
logged here, and you update the outcome when you hear back (or don't).

After ~50-100 logged applications with outcomes, `stats` starts showing
real conversion rates by score band — that's the honest version of the
"passing %" you wanted, earned from your own data rather than invented.
"""
from __future__ import annotations

import csv
import difflib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_DB = "applications.db"

OUTCOMES = [
    "pending",          # applied, no response yet
    "ghosted",          # no response after a long wait
    "rejected_auto",    # rejection with no human contact (likely ATS/recruiter screen)
    "rejected_screen",  # rejected after a recruiter call
    "recruiter_call",   # got a recruiter screen
    "interview",        # got to a hiring-manager/technical interview
    "offer",
]

# Which outcomes count as "a human engaged with me" for conversion stats
POSITIVE_OUTCOMES = {"recruiter_call", "interview", "offer"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    applied_date TEXT NOT NULL,
    company TEXT,
    role TEXT,
    resume_version TEXT,
    jd_text TEXT,
    ats_score REAL,
    recruiter_score REAL,
    manager_score REAL,
    days_after_posting INTEGER,
    outcome TEXT NOT NULL DEFAULT 'pending',
    outcome_date TEXT,
    notes TEXT,
    visibility_score REAL,
    components TEXT
);
"""

# Columns added after the first release, applied to existing databases.
MIGRATIONS = {
    "visibility_score": "ALTER TABLE applications ADD COLUMN visibility_score REAL",
    # JSON: the component scores behind the headlines (see components_from_report)
    "components": "ALTER TABLE applications ADD COLUMN components TEXT",
}


@dataclass
class Application:
    id: int
    applied_date: str
    company: str
    role: str
    resume_version: str
    ats_score: float | None
    recruiter_score: float | None
    manager_score: float | None
    days_after_posting: int | None
    outcome: str
    outcome_date: str | None
    notes: str | None
    visibility_score: float | None = None
    components: dict | None = None


def _load_components(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        val = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return val if isinstance(val, dict) else None


def components_from_report(report) -> dict:
    """The component scores behind a FullReport's headlines, as plain JSON.

    Stored per application so `stats` can later say WHICH part of the score
    tracks callbacks — the headline alone can't tell a parse problem from a
    missing-skill problem from a salary mismatch."""
    vis = getattr(report, "visibility", None)
    rec = getattr(report, "recruiter_result", None)
    mgr = getattr(report, "manager_result", None)
    sem = getattr(report, "semantic_result", None)
    kw = getattr(report, "keyword_result", None)
    extraction = getattr(report, "jd_extraction", None) or {}
    return {
        "visibility": {
            "searches_matched": vis.matched_count,
            "searches_total": len(vis.searches),
            "parse_safe": bool(vis.parse_safe),
        } if vis else None,
        "recruiter_checks": {c.name: c.status for c in rec.checks} if rec else {},
        "manager_dimensions": dict(mgr.dimensions) if mgr is not None and mgr.available else {},
        "semantic_score": sem.semantic_score if sem is not None and sem.available else None,
        "keyword_score": kw.score if kw is not None else None,
        "jd_source": extraction.get("source", "rules"),
    }


# Recruiter check status -> number for correlation. "skipped" is missing
# data (the check didn't apply), not a zero.
_CHECK_VALUE = {"pass": 1.0, "warn": 0.5, "fail": 0.0}


def numeric_components(components: dict | None) -> dict[str, float]:
    """Flatten a stored components dict into {label: number} for analysis."""
    out: dict[str, float] = {}
    if not components:
        return out
    vis = components.get("visibility") or {}
    total = vis.get("searches_total")
    if total:
        out["visibility: searches matched %"] = 100.0 * (vis.get("searches_matched") or 0) / total
    if vis.get("parse_safe") is not None:
        out["visibility: parse gate passed"] = 1.0 if vis["parse_safe"] else 0.0
    for name, status in (components.get("recruiter_checks") or {}).items():
        if status in _CHECK_VALUE:
            out[f"recruiter: {name}"] = _CHECK_VALUE[status]
    for dim, val in (components.get("manager_dimensions") or {}).items():
        if isinstance(val, (int, float)):
            out[f"manager: {dim}"] = float(val)
    for key, label in (("semantic_score", "llm fit (semantic)"), ("keyword_score", "keyword match")):
        val = components.get(key)
        if isinstance(val, (int, float)):
            out[label] = float(val)
    return out


# ------------------------------------------------- company / role defaults

# ATS hosts whose first path segment is the employer's slug
_PATH_SLUG_HOSTS = ("greenhouse.io", "lever.co", "ashbyhq.com", "workable.com",
                    "smartrecruiters.com", "jobvite.com")
# ATS hosts whose leftmost subdomain is the employer's slug
_SUBDOMAIN_SLUG_HOSTS = ("myworkdayjobs.com", "bamboohr.com", "recruitee.com",
                         "teamtailor.com", "personio.de", "personio.com", "breezy.hr")
# Job boards: the host says nothing about the employer
_BOARD_HOSTS = ("linkedin.com", "indeed.com", "naukri.com", "glassdoor.com",
                "monster.com", "ziprecruiter.com", "wellfound.com", "angel.co",
                "instahyre.com", "foundit.in", "dice.com", "bayt.com")
_GENERIC_LABELS = {"www", "jobs", "careers", "career", "apply", "boards", "job-boards",
                   "hire", "work", "join", "en", "us", "uk", "eu", "recruiting"}
_ABOUT_SKIP = {"the", "this", "us", "you", "our", "your", "role", "team", "position",
               "job", "company", "opportunity"}


def _slug_name(slug: str) -> str | None:
    slug = re.sub(r"[^A-Za-z0-9-_]", "", slug or "")
    if not slug or slug.lower() in _GENERIC_LABELS or slug.isdigit():
        return None
    words = re.split(r"[-_]+", slug)
    return " ".join(w if w.isupper() else w.capitalize() for w in words if w) or None


def company_from_url(url: str | None) -> str | None:
    """Best-effort employer name from a posting URL (None when the host is a
    job board or says nothing useful)."""
    if not url:
        return None
    try:
        parsed = urlparse(url if "://" in url else "https://" + url)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if not host or any(host == b or host.endswith("." + b) for b in _BOARD_HOSTS):
        return None
    parts = [p for p in parsed.path.split("/") if p]
    labels = host.split(".")
    if any(host == h or host.endswith("." + h) for h in _PATH_SLUG_HOSTS) and parts:
        return _slug_name(parts[0])
    if any(host.endswith("." + h) for h in _SUBDOMAIN_SLUG_HOSTS):
        return _slug_name(labels[0])
    # an employer's own careers site: careers.acme.com, acme.com/careers
    core = labels[-3] if len(labels) >= 3 and len(labels[-1]) == 2 and len(labels[-2]) <= 3 \
        else (labels[-2] if len(labels) >= 2 else labels[0])
    return _slug_name(core)


_ABOUT_RE = re.compile(r"^\W*about\s+(?:us\s*[:\-\u2013\u2014]\s*)?(?P<name>[^\n:.,;!?]{2,60})", re.I | re.M)
_COMPANY_LINE_RE = re.compile(r"^\W*(?:company|employer|organi[sz]ation)\s*(?:name)?\s*[:\-]\s*(?P<name>[^\n]{2,60})$",
                              re.I | re.M)


def company_from_jd(jd_text: str) -> str | None:
    """Employer name from an explicit 'Company: X' line or an 'About X' heading."""
    text = jd_text or ""
    m = _COMPANY_LINE_RE.search(text)
    if m:
        return m.group("name").strip(" -*\t") or None
    for m in _ABOUT_RE.finditer(text):
        name = m.group("name").strip(" -*\t")
        first = name.split()[0].lower() if name.split() else ""
        if first in _ABOUT_SKIP or not name[:1].isupper():
            continue
        # a heading, not a sentence: "About Acme Corp" but not "About Acme, we build..."
        words = name.split()
        if len(words) <= 5:
            return name
    return None


def default_company_role(jd_text: str = "", jd_url: str | None = None,
                         jd_title: str | None = None) -> tuple[str | None, str | None]:
    """(company, role) guessed from the JD, for logging without typing them.
    The role is the extracted JD title; the company comes from a 'Company:'
    or 'About X' line, else the posting URL's host. Either may be None."""
    company = company_from_jd(jd_text) or company_from_url(jd_url)
    role = (jd_title or "").strip() or None
    return company, role


def _connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    have = {r["name"] for r in conn.execute("PRAGMA table_info(applications)")}
    for col, ddl in MIGRATIONS.items():
        if col not in have:
            conn.execute(ddl)
    return conn


def log_application(
    company: str,
    role: str,
    ats_score: float | None,
    recruiter_score: float | None,
    manager_score: float | None,
    jd_text: str = "",
    resume_version: str = "",
    days_after_posting: int | None = None,
    applied_date: str | None = None,
    notes: str = "",
    db_path: str = DEFAULT_DB,
    visibility_score: float | None = None,
    report=None,
    components: dict | None = None,
) -> int:
    """Record an application. Pass the FullReport as `report` to also store
    its component scores (or a ready `components` dict)."""
    if components is None and report is not None:
        components = components_from_report(report)
    conn = _connect(db_path)
    with conn:
        cur = conn.execute(
            """INSERT INTO applications
               (applied_date, company, role, resume_version, jd_text,
                ats_score, recruiter_score, manager_score, days_after_posting,
                outcome, notes, visibility_score, components)
               VALUES (?,?,?,?,?,?,?,?,?,'pending',?,?,?)""",
            (
                applied_date or date.today().isoformat(),
                company, role, resume_version, jd_text,
                ats_score, recruiter_score, manager_score, days_after_posting, notes,
                visibility_score,
                json.dumps(components) if components is not None else None,
            ),
        )
    app_id = cur.lastrowid
    conn.close()
    return app_id


def set_outcome(app_id: int, outcome: str, notes: str | None = None, db_path: str = DEFAULT_DB) -> bool:
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of: {', '.join(OUTCOMES)}")
    conn = _connect(db_path)
    with conn:
        cur = conn.execute(
            """UPDATE applications
               SET outcome = ?, outcome_date = ?,
                   notes = COALESCE(?, notes)
               WHERE id = ?""",
            (outcome, date.today().isoformat(), notes, app_id),
        )
    changed = cur.rowcount > 0
    conn.close()
    return changed


def list_applications(limit: int = 50, db_path: str = DEFAULT_DB) -> list[Application]:
    conn = _connect(db_path)
    rows = conn.execute(
        """SELECT id, applied_date, company, role, resume_version, ats_score,
                  recruiter_score, manager_score, days_after_posting, outcome,
                  outcome_date, notes, visibility_score, components
           FROM applications ORDER BY applied_date DESC, id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        Application(
            id=r["id"], applied_date=r["applied_date"], company=r["company"] or "",
            role=r["role"] or "", resume_version=r["resume_version"] or "",
            ats_score=r["ats_score"], recruiter_score=r["recruiter_score"],
            manager_score=r["manager_score"], days_after_posting=r["days_after_posting"],
            outcome=r["outcome"], outcome_date=r["outcome_date"], notes=r["notes"],
            visibility_score=r["visibility_score"],
            components=_load_components(r["components"]),
        )
        for r in rows
    ]


def last_application_id(db_path: str = DEFAULT_DB) -> int | None:
    """The most recently logged application's id (by insertion)."""
    conn = _connect(db_path)
    row = conn.execute("SELECT MAX(id) AS m FROM applications").fetchone()
    conn.close()
    return row["m"]


def find_applications(company: str, db_path: str = DEFAULT_DB) -> list[Application]:
    """Applications whose company matches `company`: case-insensitive
    substring first, then close spellings ("acme" finds "ACME Corp",
    "gogle" finds "Google"). Newest first."""
    needle = (company or "").strip().lower()
    if not needle:
        return []
    apps = list_applications(limit=1_000_000, db_path=db_path)
    hits = [a for a in apps if needle in a.company.lower()]
    if hits:
        return hits
    names = {a.company.lower() for a in apps if a.company}
    close = set(difflib.get_close_matches(needle, names, n=5, cutoff=0.75))
    return [a for a in apps if a.company.lower() in close]


def export_csv(out_path: str, db_path: str = DEFAULT_DB) -> tuple[str, int]:
    """Flat export (JD text excluded — it bloats the file and Power BI won't
    want it). Ready to point a Power BI funnel dashboard at."""
    conn = _connect(db_path)
    rows = conn.execute(
        """SELECT id, applied_date, company, role, resume_version, ats_score,
                  recruiter_score, manager_score, days_after_posting, outcome,
                  outcome_date, notes, visibility_score, components
           FROM applications ORDER BY applied_date, id"""
    ).fetchall()
    conn.close()

    p = Path(out_path)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "applied_date", "company", "role", "resume_version",
            "ats_score", "recruiter_score", "manager_score", "days_after_posting",
            "outcome", "outcome_date", "reached_human", "notes", "visibility_score",
            "components_json",
        ])
        for r in rows:
            writer.writerow([
                r["id"], r["applied_date"], r["company"], r["role"], r["resume_version"],
                r["ats_score"], r["recruiter_score"], r["manager_score"],
                r["days_after_posting"], r["outcome"], r["outcome_date"],
                1 if r["outcome"] in POSITIVE_OUTCOMES else 0,
                r["notes"], r["visibility_score"], r["components"] or "",
            ])
    return str(p.resolve()), len(rows)


# (label, column) — "visibility" is the Layer 1 headline; "ats" is the legacy
# composite, kept so older logged applications still analyse.
LAYERS = (("visibility", "visibility_score"), ("ats", "ats_score"),
          ("recruiter", "recruiter_score"), ("manager", "manager_score"))


def _band(score: float | None) -> str | None:
    """Aligned with the report's bands (Strong/Workable/Weak/Very weak,
    cut at 80/60/40) so the stats tables and the score reports speak the
    same language — they used to disagree, which made cross-reading them
    a quiet trap."""
    if score is None:
        return None
    if score >= 80:
        return "80-100 (Strong)"
    if score >= 60:
        return "60-79 (Workable)"
    if score >= 40:
        return "40-59 (Weak)"
    return "0-39 (Very weak)"


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation in pure Python (unrounded). None when there's
    nothing to correlate (fewer than 3 points or zero variance)."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return max(-1.0, min(1.0, cov / (vx * vy) ** 0.5))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n, as fractions in [0, 1].

    Unlike the textbook p ± z·SE interval it stays inside [0, 1] and is
    honest at small n and at 0% / 100% — exactly where a job-search log
    lives. n == 0 gives the uninformative (0, 1)."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def format_rate(k: int, n: int) -> str:
    """'12/40 = 30% (95% CI 18–45%)' — a rate never travels without its n
    and its interval."""
    lo, hi = wilson_interval(k, n)
    pct = round(k / n * 100) if n else 0
    return f"{k}/{n} = {pct}% (95% CI {lo * 100:.0f}\u2013{hi * 100:.0f}%)"


def _rate_bucket(bucket: dict) -> dict:
    """Fill conversion_pct, ci_low/ci_high (percent) and a printable rate."""
    k, n = bucket["reached_human"], bucket["n"]
    lo, hi = wilson_interval(k, n)
    bucket["conversion_pct"] = round(k / n * 100, 1)
    bucket["ci_low"] = round(lo * 100, 1)
    bucket["ci_high"] = round(hi * 100, 1)
    bucket["rate"] = format_rate(k, n)
    return bucket


def fisher_interval(r: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a correlation via the Fisher z-transform. With n <= 3 there
    is no standard error, so the interval is the whole range (-1, 1)."""
    if n <= 3:
        return -1.0, 1.0
    r = max(-0.999999, min(0.999999, r))
    fz = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    return math.tanh(fz - z * se), math.tanh(fz + z * se)


# Below this many points a correlation's size isn't graded at all — a
# "strong" r on 12 applications is mostly luck.
MIN_N_FOR_STRENGTH = 30


def correlation_summary(xs: list[float], ys: list[float]) -> dict | None:
    """Correlation of a score with reaching a human, with its 95% CI and a
    plain reading. None when it can't be computed."""
    r = _pearson(xs, ys)
    if r is None:
        return None
    n = len(xs)
    lo, hi = fisher_interval(r, n)
    distinguishable = lo > 0 or hi < 0
    ci_text = f"95% CI {lo:+.2f} to {hi:+.2f}, n={n}"
    if not distinguishable:
        strength = "not distinguishable from zero"
        reading = (f"r = {r:+.2f} ({ci_text}): not distinguishable from zero — "
                   f"no evidence yet that this score tracks your callbacks")
    else:
        direction = ("higher scores -> more callbacks" if r > 0
                     else "higher scores -> FEWER callbacks")
        if n < MIN_N_FOR_STRENGTH:
            strength = f"{'positive' if r > 0 else 'negative'} (too few to grade, n<{MIN_N_FOR_STRENGTH})"
        else:
            size = abs(r)
            strength = "strong" if size >= 0.5 else "moderate" if size >= 0.3 else "weak"
        reading = f"{strength} correlation, r = {r:+.2f} ({ci_text}); {direction}"
    return {
        "correlation": round(r, 3),
        "ci_low": round(lo, 3),
        "ci_high": round(hi, 3),
        "n": n,
        "distinguishable_from_zero": distinguishable,
        "strength": strength,
        "reading": reading,
    }


# Apply-timing buckets in time order: (label, last day included)
TIMING_BUCKETS = (("0-2 days", 2), ("3-7 days", 7), ("8-14 days", 14), ("15+ days", None))


def _timing_bucket(days: int) -> str:
    for label, upto in TIMING_BUCKETS:
        if upto is None or days <= upto:
            return label
    return TIMING_BUCKETS[-1][0]


def count_stale_pending(db_path: str = DEFAULT_DB, days: int = 45) -> int:
    """How many pending applications are older than `days` (ghost candidates)."""
    conn = _connect(db_path)
    row = conn.execute(
        """SELECT COUNT(*) AS c FROM applications
           WHERE outcome = 'pending' AND date(applied_date) <= date('now', ?)""",
        (f"-{days} days",),
    ).fetchone()
    conn.close()
    return row["c"]


def reap_ghosts(db_path: str = DEFAULT_DB, days: int = 45) -> int:
    """Mark long-pending applications as ghosted. A 'pending' from two months
    ago is a ghost by any realistic reading; leaving it pending quietly
    corrupts your resolved-outcome counts."""
    conn = _connect(db_path)
    with conn:
        cur = conn.execute(
            """UPDATE applications
               SET outcome = 'ghosted', outcome_date = ?
               WHERE outcome = 'pending'
                 AND date(applied_date) <= date('now', ?)""",
            (date.today().isoformat(), f"-{days} days"),
        )
    changed = cur.rowcount
    conn.close()
    return changed


def conversion_stats(db_path: str = DEFAULT_DB, min_resolved: int = 20,
                     reap_days: int | None = None) -> dict:
    """Conversion rate by score band, once enough outcomes are recorded.

    Deliberately refuses to report rates below `min_resolved` resolved
    applications — a 2-of-3 sample is noise, and presenting it as a
    percentage is exactly the fabricated precision this tool avoids. Above
    it, every rate carries its n and a Wilson 95% interval, and every
    correlation a Fisher-z 95% interval.

    `reap_days`: first mark pending applications older than this as ghosted
    (reap_ghosts). None leaves them alone, and the result says how many
    stale pendings were left out of the resolved counts.
    """
    reap_note = ""
    reaped = 0
    if reap_days is not None:
        reaped = reap_ghosts(db_path=db_path, days=reap_days)
        reap_note = (f"Marked {reaped} pending application(s) older than {reap_days} days "
                     f"as ghosted first. ")
    else:
        stale = count_stale_pending(db_path=db_path)
        if stale:
            reap_note = (f"Ghosts NOT reaped: {stale} pending application(s) are older than "
                         f"45 days and are left out of these counts (run `log reap-ghosts`). ")

    conn = _connect(db_path)
    rows = conn.execute(
        """SELECT ats_score, recruiter_score, manager_score, outcome, days_after_posting,
                  visibility_score, components
           FROM applications WHERE outcome != 'pending'"""
    ).fetchall()
    total_logged = conn.execute("SELECT COUNT(*) AS c FROM applications").fetchone()["c"]
    conn.close()

    resolved = len(rows)
    reached = sum(1 for r in rows if r["outcome"] in POSITIVE_OUTCOMES)
    result = {
        "total_logged": total_logged,
        "resolved": resolved,
        "min_resolved_needed": min_resolved,
        "calibrated": resolved >= min_resolved,
        "reaped": reaped if reap_days is not None else None,
        "overall": _rate_bucket({"n": resolved, "reached_human": reached}) if resolved else None,
        "bands": {},
        "predictiveness": {},
        "component_predictiveness": [],
        "apply_timing": {},
        "message": "",
    }

    if resolved < min_resolved:
        result["message"] = reap_note + (
            f"{resolved} resolved outcome(s) logged. Need at least {min_resolved} before "
            f"conversion rates mean anything — until then the three layers stay heuristic "
            f"scores, not probabilities."
        )
        return result

    def outcome01(r) -> float:
        return 1.0 if r["outcome"] in POSITIVE_OUTCOMES else 0.0

    for layer, key in LAYERS:
        bands: dict[str, dict] = {}
        for r in rows:
            band = _band(r[key])
            if band is None:
                continue
            bucket = bands.setdefault(band, {"n": 0, "reached_human": 0})
            bucket["n"] += 1
            if r["outcome"] in POSITIVE_OUTCOMES:
                bucket["reached_human"] += 1
        for bucket in bands.values():
            _rate_bucket(bucket)
        result["bands"][layer] = dict(sorted(bands.items(), reverse=True))

    # ---- which layer actually PREDICTS your outcomes: point-biserial
    # correlation (score vs reached-human 0/1) per layer, with a Fisher-z
    # interval. If manager scores turn out indistinguishable from zero while
    # recruiter scores track tightly, that's real information about where
    # your applications are dying.
    predictiveness: dict[str, dict] = {}
    for layer, key in LAYERS:
        pts = [(float(r[key]), outcome01(r)) for r in rows if r[key] is not None]
        summary = correlation_summary([x for x, _ in pts], [y for _, y in pts])
        if summary is not None:
            predictiveness[layer] = summary
    result["predictiveness"] = predictiveness

    # ---- the same, one row per stored component (searches, each recruiter
    # check, each manager dimension...), ranked: evidence first, then size.
    series: dict[str, tuple[list[float], list[float]]] = {}
    for r in rows:
        for label, val in numeric_components(_load_components(r["components"])).items():
            xs, ys = series.setdefault(label, ([], []))
            xs.append(val)
            ys.append(outcome01(r))
    comp_rows = []
    for label, (xs, ys) in series.items():
        summary = correlation_summary(xs, ys)
        if summary is not None:
            comp_rows.append({"component": label, **summary})
    comp_rows.sort(key=lambda c: (not c["distinguishable_from_zero"], -abs(c["correlation"]),
                                  c["component"]))
    result["component_predictiveness"] = comp_rows

    # ---- apply timing: does applying early actually matter for you?
    timing: dict[str, dict] = {}
    for r in rows:
        d = r["days_after_posting"]
        if d is None:
            continue
        bucket = timing.setdefault(_timing_bucket(d), {"n": 0, "reached_human": 0})
        bucket["n"] += 1
        if r["outcome"] in POSITIVE_OUTCOMES:
            bucket["reached_human"] += 1
    order = [label for label, _ in TIMING_BUCKETS]
    result["apply_timing"] = {label: _rate_bucket(timing[label]) for label in order if label in timing}

    result["message"] = reap_note + (
        f"Based on {resolved} resolved applications. These ARE real rates from your own "
        f"data — but they describe your history, not a guarantee for the next application. "
        f"Intervals are 95%: wide ones mean you need more outcomes before reading much into "
        f"the difference."
    )
    return result
