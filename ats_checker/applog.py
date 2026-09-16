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
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

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
    notes TEXT
);
"""


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


def _connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
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
) -> int:
    conn = _connect(db_path)
    with conn:
        cur = conn.execute(
            """INSERT INTO applications
               (applied_date, company, role, resume_version, jd_text,
                ats_score, recruiter_score, manager_score, days_after_posting,
                outcome, notes)
               VALUES (?,?,?,?,?,?,?,?,?,'pending',?)""",
            (
                applied_date or date.today().isoformat(),
                company, role, resume_version, jd_text,
                ats_score, recruiter_score, manager_score, days_after_posting, notes,
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
                  outcome_date, notes
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
        )
        for r in rows
    ]


def export_csv(out_path: str, db_path: str = DEFAULT_DB) -> tuple[str, int]:
    """Flat export (JD text excluded — it bloats the file and Power BI won't
    want it). Ready to point a Power BI funnel dashboard at."""
    conn = _connect(db_path)
    rows = conn.execute(
        """SELECT id, applied_date, company, role, resume_version, ats_score,
                  recruiter_score, manager_score, days_after_posting, outcome,
                  outcome_date, notes
           FROM applications ORDER BY applied_date, id"""
    ).fetchall()
    conn.close()

    p = Path(out_path)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "applied_date", "company", "role", "resume_version",
            "ats_score", "recruiter_score", "manager_score", "days_after_posting",
            "outcome", "outcome_date", "reached_human", "notes",
        ])
        for r in rows:
            writer.writerow([
                r["id"], r["applied_date"], r["company"], r["role"], r["resume_version"],
                r["ats_score"], r["recruiter_score"], r["manager_score"],
                r["days_after_posting"], r["outcome"], r["outcome_date"],
                1 if r["outcome"] in POSITIVE_OUTCOMES else 0,
                r["notes"],
            ])
    return str(p.resolve()), len(rows)


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
    """Pearson correlation in pure Python. None when there's nothing to
    correlate (fewer than 3 points or zero variance)."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return round(cov / (vx * vy) ** 0.5, 3)


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


def conversion_stats(db_path: str = DEFAULT_DB, min_resolved: int = 20) -> dict:
    """Conversion rate by score band, once enough outcomes are recorded.

    Deliberately refuses to report rates below `min_resolved` resolved
    applications — a 2-of-3 sample is noise, and presenting it as a
    percentage is exactly the fabricated precision this tool avoids.
    """
    conn = _connect(db_path)
    rows = conn.execute(
        """SELECT ats_score, recruiter_score, manager_score, outcome, days_after_posting
           FROM applications WHERE outcome != 'pending'"""
    ).fetchall()
    total_logged = conn.execute("SELECT COUNT(*) AS c FROM applications").fetchone()["c"]
    conn.close()

    resolved = len(rows)
    result = {
        "total_logged": total_logged,
        "resolved": resolved,
        "min_resolved_needed": min_resolved,
        "calibrated": resolved >= min_resolved,
        "bands": {},
        "predictiveness": {},
        "apply_timing": {},
        "message": "",
    }

    if resolved < min_resolved:
        result["message"] = (
            f"{resolved} resolved outcome(s) logged. Need at least {min_resolved} before "
            f"conversion rates mean anything — until then the three layers stay heuristic "
            f"scores, not probabilities."
        )
        return result

    for layer, key in (("ats", "ats_score"), ("recruiter", "recruiter_score"), ("manager", "manager_score")):
        bands: dict[str, dict] = {}
        for r in rows:
            band = _band(r[key])
            if band is None:
                continue
            bucket = bands.setdefault(band, {"n": 0, "reached_human": 0})
            bucket["n"] += 1
            if r["outcome"] in POSITIVE_OUTCOMES:
                bucket["reached_human"] += 1
        for band, bucket in bands.items():
            bucket["conversion_pct"] = round(bucket["reached_human"] / bucket["n"] * 100, 1)
        result["bands"][layer] = dict(sorted(bands.items(), reverse=True))

    # ---- which layer actually PREDICTS your outcomes: point-biserial
    # correlation (score vs reached-human 0/1) per layer. This is the number
    # the README promised — if manager scores turn out uncorrelated with
    # callbacks while recruiter scores track tightly, that's real
    # information about where your applications are dying.
    predictiveness: dict[str, dict] = {}
    for layer, key in (("ats", "ats_score"), ("recruiter", "recruiter_score"), ("manager", "manager_score")):
        xs = [float(r[key]) for r in rows if r[key] is not None]
        ys = [1.0 if r["outcome"] in POSITIVE_OUTCOMES else 0.0
              for r in rows if r[key] is not None]
        r_val = _pearson(xs, ys)
        if r_val is not None:
            strength = ("strong" if abs(r_val) >= 0.5 else
                        "moderate" if abs(r_val) >= 0.3 else "weak")
            direction = "higher scores -> more callbacks" if r_val > 0 else "higher scores -> FEWER callbacks"
            predictiveness[layer] = {
                "correlation": r_val,
                "n": len(xs),
                "strength": strength,
                "reading": f"{strength} correlation ({r_val:+.2f}); {direction}",
            }
    result["predictiveness"] = predictiveness

    # ---- apply timing: does applying early actually matter for you? Uses
    # the days_after_posting column that was logged but never analysed.
    timing: dict[str, dict] = {}
    for r in rows:
        d = r["days_after_posting"]
        if d is None:
            continue
        bucket_key = ("0-2 days" if d <= 2 else
                      "3-7 days" if d <= 7 else
                      "8-14 days" if d <= 14 else "15+ days")
        bucket = timing.setdefault(bucket_key, {"n": 0, "reached_human": 0})
        bucket["n"] += 1
        if r["outcome"] in POSITIVE_OUTCOMES:
            bucket["reached_human"] += 1
    for bucket in timing.values():
        bucket["conversion_pct"] = round(bucket["reached_human"] / bucket["n"] * 100, 1)
    result["apply_timing"] = dict(sorted(timing.items()))

    result["message"] = (
        f"Based on {resolved} resolved applications. These ARE real rates from your own "
        f"data — but they describe your history, not a guarantee for the next application."
    )
    return result
