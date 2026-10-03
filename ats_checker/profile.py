"""Candidate profile — the fixed facts a recruiter screens against.

These can't be inferred from a JD and are only partly inferable from a
resume (nobody puts their notice period or salary expectation on a CV), so
you fill this in once and every run checks it against what each JD demands.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROFILE_PATH = "profile.yaml"

TEMPLATE = """# Your fixed profile — the things a recruiter screens for in the first
# 30 seconds. Fill this in once. Leave a field blank/null if it doesn't
# apply and the matching check will be skipped rather than guessed.
#
# The values below are a FICTIONAL EXAMPLE. Replace every one with yours —
# a check run against example values is a wrong check, not a skipped one.
# (Easier: run `python server.py` and use the Setup tab's form instead.)

# Total years of relevant professional experience (decimals fine: 1.5)
years_experience: 3

# Your current/most recent job title
current_title: "Data Analyst"

# Highest completed education level.
# One of: high_school | diploma | bachelors | masters | mba | phd
education_level: bachelors

# Certifications you actually hold (exact names help matching)
certifications:
  - "Example Certification Name"

# Where you are, and whether you'd move
location: "Pune, India"
open_to_relocation: true
# Which work modes you'd accept: remote | hybrid | onsite
acceptable_work_modes:
  - remote
  - hybrid
  - onsite

# Right-to-work. Used to flag JDs that say sponsorship isn't available.
# Free text, e.g. "Indian citizen, no sponsorship needed for India roles"
work_authorization: "Indian citizen — needs sponsorship outside India"
# Countries/regions where you can work without sponsorship
work_authorized_in:
  - "India"

# Notice period in days (0 if immediately available)
notice_period_days: 30

# Expected annual salary. Currency is free text; leave null to skip the check.
expected_salary_min: null
expected_salary_max: null
salary_currency: "INR"
"""


@dataclass
class CandidateProfile:
    years_experience: float | None = None
    current_title: str = ""
    education_level: str = ""
    certifications: list[str] = field(default_factory=list)
    location: str = ""
    open_to_relocation: bool | None = None
    acceptable_work_modes: list[str] = field(default_factory=list)
    work_authorization: str = ""
    work_authorized_in: list[str] = field(default_factory=list)
    notice_period_days: int | None = None
    expected_salary_min: float | None = None
    expected_salary_max: float | None = None
    salary_currency: str = ""

    @property
    def is_empty(self) -> bool:
        return self.years_experience is None and not self.current_title

    def education_rank(self) -> int:
        """Ordinal so we can compare against a JD's minimum degree."""
        return EDUCATION_RANK.get((self.education_level or "").lower().strip(), 0)


EDUCATION_RANK = {
    "": 0,
    "high_school": 1,
    "diploma": 2,
    "bachelors": 3,
    "bachelor": 3,
    "masters": 4,
    "master": 4,
    "mba": 4,
    "phd": 5,
    "doctorate": 5,
}


def write_template(path: str = DEFAULT_PROFILE_PATH, overwrite: bool = False) -> str:
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists — pass --force to overwrite it.")
    p.write_text(TEMPLATE, encoding="utf-8")
    return str(p.resolve())


EDUCATION_LEVELS = ("high_school", "diploma", "bachelors", "masters", "mba", "phd")
WORK_MODES = ("remote", "hybrid", "onsite")


def profile_from_dict(data: dict) -> CandidateProfile:
    """Build a profile from a parsed mapping (profile.yaml or a web form),
    coercing loosely: a blank or unparseable number is None, a list may
    arrive as one comma/newline-separated string."""

    def _list(key: str) -> list[str]:
        val = data.get(key) or []
        if isinstance(val, str):
            val = re.split(r"[\n,;]+", val) if ("\n" in val or "," in val or ";" in val) else [val]
        return [str(v).strip() for v in val if str(v).strip()]

    return CandidateProfile(
        years_experience=_maybe_float(data.get("years_experience")),
        current_title=str(data.get("current_title") or "").strip(),
        education_level=str(data.get("education_level") or "").strip(),
        certifications=_list("certifications"),
        location=str(data.get("location") or "").strip(),
        open_to_relocation=_maybe_bool(data.get("open_to_relocation")),
        acceptable_work_modes=[m.lower() for m in _list("acceptable_work_modes")],
        work_authorization=str(data.get("work_authorization") or "").strip(),
        work_authorized_in=_list("work_authorized_in"),
        notice_period_days=_maybe_int(data.get("notice_period_days")),
        expected_salary_min=_maybe_float(data.get("expected_salary_min")),
        expected_salary_max=_maybe_float(data.get("expected_salary_max")),
        salary_currency=str(data.get("salary_currency") or "").strip(),
    )


def profile_to_dict(prof: CandidateProfile) -> dict:
    return {
        "years_experience": prof.years_experience,
        "current_title": prof.current_title,
        "education_level": prof.education_level,
        "certifications": list(prof.certifications),
        "location": prof.location,
        "open_to_relocation": prof.open_to_relocation,
        "acceptable_work_modes": list(prof.acceptable_work_modes),
        "work_authorization": prof.work_authorization,
        "work_authorized_in": list(prof.work_authorized_in),
        "notice_period_days": prof.notice_period_days,
        "expected_salary_min": prof.expected_salary_min,
        "expected_salary_max": prof.expected_salary_max,
        "salary_currency": prof.salary_currency,
    }


def validate_profile(prof: CandidateProfile) -> list[str]:
    """Problems that would make a check misfire (empty = fine). Blank fields
    are allowed: their checks are skipped, never guessed."""
    errors = []
    if prof.education_level and prof.education_level.lower() not in EDUCATION_LEVELS:
        errors.append(f"education_level must be one of: {', '.join(EDUCATION_LEVELS)}")
    bad = [m for m in prof.acceptable_work_modes if m not in WORK_MODES]
    if bad:
        errors.append(f"acceptable_work_modes may only contain {', '.join(WORK_MODES)} (got {', '.join(bad)})")
    if prof.years_experience is not None and not 0 <= prof.years_experience <= 60:
        errors.append("years_experience must be between 0 and 60")
    if prof.notice_period_days is not None and not 0 <= prof.notice_period_days <= 365:
        errors.append("notice_period_days must be between 0 and 365")
    lo, hi = prof.expected_salary_min, prof.expected_salary_max
    if (lo is not None and lo < 0) or (hi is not None and hi < 0):
        errors.append("expected salary can't be negative")
    if lo is not None and hi is not None and lo > hi:
        errors.append("expected_salary_min is above expected_salary_max")
    return errors


SAVED_HEADER = """# Your fixed profile — the things a recruiter screens for in the first
# 30 seconds. Written by the web UI's setup wizard; edit it there or here.
# A blank/null field skips its check rather than guessing.
"""


def save_profile(prof: CandidateProfile, path: str = DEFAULT_PROFILE_PATH) -> str:
    """Write profile.yaml (atomically: a crash mid-write never leaves half a
    file). Raises ValueError when validate_profile finds problems."""
    import yaml

    errors = validate_profile(prof)
    if errors:
        raise ValueError("; ".join(errors))
    target = Path(path)
    body = yaml.safe_dump(profile_to_dict(prof), sort_keys=False, allow_unicode=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(SAVED_HEADER + "\n" + body, encoding="utf-8")
    tmp.replace(target)
    return str(target.resolve())


def load_profile(path: str | None = None) -> CandidateProfile:
    """Load profile.yaml. Returns an empty profile if the file is missing —
    the recruiter layer will then skip the checks it can't make rather than
    inventing answers."""
    target = Path(path or DEFAULT_PROFILE_PATH)
    if not target.exists():
        return CandidateProfile()

    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError("pyyaml is required to read profile.yaml — pip install pyyaml") from e

    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{target} did not parse into a mapping of fields.")
    return profile_from_dict(data)


def _maybe_bool(v) -> bool | None:
    """YAML gives real booleans, but a quoted "no" (or a form's "false") is a
    string — and bool("no") is True. Unknown text is None (check skipped)."""
    if v is None or isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "yes", "y", "1", "on"}:
        return True
    if s in {"false", "no", "n", "0", "off"}:
        return False
    return None


def _maybe_float(v) -> float | None:
    if isinstance(v, str) and not v.strip():
        return None
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _maybe_int(v) -> int | None:
    if isinstance(v, str) and not v.strip():
        return None
    try:
        return int(float(v)) if v is not None else None
    except (TypeError, ValueError):
        return None
