"""Candidate profile — the fixed facts a recruiter screens against.

These can't be inferred from a JD and are only partly inferable from a
resume (nobody puts their notice period or salary expectation on a CV), so
you fill this in once and every run checks it against what each JD demands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROFILE_PATH = "profile.yaml"

TEMPLATE = """# Your fixed profile — the things a recruiter screens for in the first
# 30 seconds. Fill this in once. Leave a field blank/null if it doesn't
# apply and the matching check will be skipped rather than guessed.

# Total years of relevant professional experience (decimals fine: 1.5)
years_experience: 1.2

# Your current/most recent job title
current_title: "BI Analyst"

# Highest completed education level.
# One of: high_school | diploma | bachelors | masters | mba | phd
education_level: bachelors

# Certifications you actually hold (exact names help matching)
certifications:
  - "Microsoft Power BI Data Analyst (PL-300)"

# Where you are, and whether you'd move
location: "Vadodara, India"
open_to_relocation: true
# Which work modes you'd accept: remote | hybrid | onsite
acceptable_work_modes:
  - remote
  - hybrid
  - onsite

# Right-to-work. Used to flag JDs that say sponsorship isn't available.
# Free text, e.g. "Indian citizen, no sponsorship needed for India roles"
work_authorization: "Indian citizen — requires sponsorship for UAE/US roles"
# Countries/regions where you can work without sponsorship
work_authorized_in:
  - "India"

# Notice period in days (0 if immediately available)
notice_period_days: 60

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

    def _list(key: str) -> list[str]:
        val = data.get(key) or []
        if isinstance(val, str):
            return [val]
        return [str(v) for v in val]

    return CandidateProfile(
        years_experience=_maybe_float(data.get("years_experience")),
        current_title=str(data.get("current_title") or ""),
        education_level=str(data.get("education_level") or ""),
        certifications=_list("certifications"),
        location=str(data.get("location") or ""),
        open_to_relocation=data.get("open_to_relocation"),
        acceptable_work_modes=[m.lower() for m in _list("acceptable_work_modes")],
        work_authorization=str(data.get("work_authorization") or ""),
        work_authorized_in=_list("work_authorized_in"),
        notice_period_days=_maybe_int(data.get("notice_period_days")),
        expected_salary_min=_maybe_float(data.get("expected_salary_min")),
        expected_salary_max=_maybe_float(data.get("expected_salary_max")),
        salary_currency=str(data.get("salary_currency") or ""),
    )


def _maybe_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _maybe_int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
