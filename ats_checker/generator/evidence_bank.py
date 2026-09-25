"""Master resume — the evidence bank the tailor generates from.

The bank is the truth constraint of the whole generator: every bullet a
tailored resume contains is SELECTED from here, never written fresh. That
makes "never invent a fact" a structural property of the data model — the
assembler can only render what the bank contains — instead of a promise
we'd have to audit after the fact.

Keep the bank richer than any single resume needs: every real bullet you've
ever written belongs here, tagged with the skills it demonstrates. The
selector picks the subset that covers this JD best; the rest stays in the
bank for the next application. Rewordings produced by the LLM layer are
verified against the original bullet (numbers/dates preserved, none added)
and every rendered line keeps a pointer back to its atom.

Schema (master_resume.yaml):

    contact:
      name / email / phone / location
    headline: "BI Analyst"            # used for the Summary line
    roles:
      - company: ...
        title: ...
        start: "2025-07"               # YYYY-MM, or "Jul 2025"
        end: "present"                # YYYY-MM, "present", or "" (= present)
        location: ...                 # optional
        bullets:
          - text: "..."               # the real bullet, verbatim
            skills: [power bi, dax]    # lowercase tags of what it demonstrates
    skills_extra: [...]               # skills you hold with no specific bullet
    education:
      - degree / institution / year
    certifications: [...]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import llm_client
from .. import parsing

DEFAULT_MASTER_PATH = "master_resume.yaml"

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ------------------------------------------------------------------ dates

def parse_ym(value: str) -> tuple[int, int] | None:
    """Normalise a role date to (year, month). Accepts YYYY-MM, YYYY/MM,
    'Mon YYYY', 'YYYY'. Returns None for empty/unparseable input."""
    s = (value or "").strip().lower()
    if not s or s in ("present", "current", "now", "ongoing"):
        return None
    m = re.fullmatch(r"((?:19|20)\d{2})[-/](\d{1,2})", s)
    if m:
        return int(m.group(1)), max(1, min(12, int(m.group(2))))
    m = re.fullmatch(r"([a-z]{3,9})\.?\s+((?:19|20)\d{2})", s)
    if m:
        mon = MONTHS.get(m.group(1)[:4].rstrip("."), None) or MONTHS.get(m.group(1)[:3], None)
        if mon:
            return int(m.group(2)), mon
    m = re.fullmatch(r"((?:19|20)\d{2})", s)
    if m:
        return int(m.group(1)), 1
    return None


# ------------------------------------------------------------------ model

@dataclass
class Bullet:
    text: str
    skills: list[str] = field(default_factory=list)


@dataclass
class Role:
    company: str
    title: str
    start: str = ""
    end: str = ""
    location: str = ""
    bullets: list[Bullet] = field(default_factory=list)

    @property
    def is_current(self) -> bool:
        return (self.end or "").strip().lower() in ("", "present", "current", "now", "ongoing")

    def start_key(self) -> tuple[int, int]:
        return parse_ym(self.start) or (1900, 1)


@dataclass
class Education:
    degree: str = ""
    institution: str = ""
    year: str = ""


@dataclass
class EvidenceBank:
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    website: str = ""
    headline: str = ""
    roles: list[Role] = field(default_factory=list)
    skills_extra: list[str] = field(default_factory=list)
    education: list[Education] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    source_path: str = ""

    @property
    def is_empty(self) -> bool:
        return not any(r.bullets for r in self.roles)

    def validate(self) -> list[str]:
        """Non-fatal problems the user should fix. Fatal ones raise."""
        if not self.roles:
            raise ValueError("Evidence bank has no roles — the tailor has nothing to generate from.")
        if not any(r.bullets for r in self.roles):
            raise ValueError("Evidence bank has no bullets — the tailor has nothing to generate from.")
        problems = []
        if not self.email:
            problems.append("No email in contact — the generated resume will lose formatting points.")
        if not self.phone:
            problems.append("No phone in contact — the generated resume will lose formatting points.")
        if not self.headline and self.roles:
            problems.append("No headline set — the Summary line will fall back to your current title.")
        for r in self.roles:
            if r.is_current and not r.bullets:
                problems.append(f"Current role '{r.company}' has no bullets.")
        return problems

    def all_skills(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.roles:
            for b in r.bullets:
                for s in b.skills:
                    seen.setdefault(s.strip().lower(), None)
        for s in self.skills_extra:
            seen.setdefault(s.strip().lower(), None)
        return list(seen)


# ------------------------------------------------------------------ yaml IO

def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


def load_bank(path: str | Path = DEFAULT_MASTER_PATH) -> EvidenceBank:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"No evidence bank at {target}. Run `python cli.py init-master` to create one "
            "(add `--from your_resume.pdf` to bootstrap it from an existing resume), then "
            "review every bullet — you are the truth gate."
        )
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError("pyyaml is required to read master_resume.yaml — pip install pyyaml") from e

    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{target} did not parse into a mapping.")

    contact = data.get("contact") or {}
    bank = EvidenceBank(
        name=str(contact.get("name") or data.get("name") or ""),
        email=str(contact.get("email") or data.get("email") or ""),
        phone=str(contact.get("phone") or data.get("phone") or ""),
        location=str(contact.get("location") or data.get("location") or ""),
        linkedin=str(contact.get("linkedin") or data.get("linkedin") or ""),
        github=str(contact.get("github") or data.get("github") or ""),
        website=str(contact.get("website") or data.get("website") or ""),
        headline=str(data.get("headline") or ""),
        skills_extra=[str(s).strip().lower() for s in _as_list(data.get("skills_extra"))],
        certifications=[str(c) for c in _as_list(data.get("certifications"))],
        source_path=str(target),
    )

    for r in _as_list(data.get("roles")):
        if not isinstance(r, dict):
            continue
        role = Role(
            company=str(r.get("company") or ""),
            title=str(r.get("title") or ""),
            start=str(r.get("start") or ""),
            end=str(r.get("end") or ""),
            location=str(r.get("location") or ""),
        )
        for b in _as_list(r.get("bullets")):
            if isinstance(b, dict):
                text = str(b.get("text") or "").strip()
                skills = [str(s).strip().lower() for s in _as_list(b.get("skills"))]
            elif isinstance(b, str):
                text, skills = b.strip(), []
            else:
                continue  # stray list/number: skip, don't reuse the last bullet's text
            if text:
                role.bullets.append(Bullet(text=text, skills=skills))
        if role.company or role.bullets:
            bank.roles.append(role)

    for e in _as_list(data.get("education")):
        if isinstance(e, dict):
            bank.education.append(Education(
                degree=str(e.get("degree") or ""),
                institution=str(e.get("institution") or ""),
                year=str(e.get("year") or ""),
            ))

    bank.roles.sort(key=lambda r: (not r.is_current, tuple(-x for x in r.start_key())))
    return bank


def bank_to_yaml(bank: EvidenceBank) -> str:
    import yaml

    data = {
        "contact": {
            "name": bank.name, "email": bank.email,
            "phone": bank.phone, "location": bank.location,
            "linkedin": bank.linkedin, "github": bank.github, "website": bank.website,
        },
        "headline": bank.headline,
        "roles": [
            {
                "company": r.company, "title": r.title,
                "start": r.start, "end": r.end, "location": r.location,
                "bullets": [{"text": b.text, "skills": b.skills} for b in r.bullets],
            }
            for r in bank.roles
        ],
        "skills_extra": bank.skills_extra,
        "education": [
            {"degree": e.degree, "institution": e.institution, "year": e.year}
            for e in bank.education
        ],
        "certifications": bank.certifications,
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


TEMPLATE = """\
# Evidence bank — your master resume. Every real bullet you've ever written
# belongs here; the tailor SELECTS from it and never invents. You are the
# truth gate: review every bullet, delete anything you couldn't defend in an
# interview, and keep adding real bullets over time.
#
# Generate a tailored resume with:
#   python cli.py tailor --master master_resume.yaml --jd jd.txt

contact:
  name: "Your Name"
  email: "you@example.com"
  phone: "+91 98765 43210"
  location: "City, Country"
  linkedin: "https://www.linkedin.com/in/your-handle"   # optional
  github: "https://github.com/your-handle"              # optional
  website: "https://your-portfolio.example.com"         # optional

# Used as the opening of the Summary line. Your real title — never a JD's.
headline: "Your Current Title"

roles:
  - company: "Company"
    title: "Your Real Title"
    start: "2024-01"          # YYYY-MM (or "Jan 2024")
    end: "present"            # YYYY-MM, or "present"
    location: "City"           # optional
    bullets:
      - text: "What you actually did, with the real numbers kept in"
        skills: [tool one, tool two, competency]   # lowercase tags it demonstrates
      - text: "Another real bullet — outcomes over duties where you can"
        skills: [tool one]

  - company: "Earlier Company"
    title: "Earlier Title"
    start: "2022-06"
    end: "2023-12"
    bullets:
      - text: "A bullet from an earlier role"
        skills: [tool]

# Skills you hold that no single bullet demonstrates — they go in the
# Skills section and still match JD keywords.
skills_extra:
  - "sql"
  - "excel"

education:
  - degree: "Degree in Field"
    institution: "University"
    year: "2024"

certifications:
  - "Certification Name"
"""


def write_template(path: str | Path = DEFAULT_MASTER_PATH, overwrite: bool = False) -> str:
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists — pass --force to overwrite it.")
    p.write_text(TEMPLATE, encoding="utf-8")
    return str(p.resolve())


# ------------------------------------------------------------------ LLM atomizer

ATOMIZE_SYSTEM_PROMPT = """You convert a raw resume into a structured evidence bank. You are a
transcriber, NOT a writer: copy bullet text VERBATIM — never rewrite, never
embellish, never add a number, tool, or outcome that isn't in the original.

Respond with STRICT JSON ONLY, no markdown fences, matching this schema:
{
  "name": "", "email": "", "phone": "", "location": "",
  "linkedin": "", "github": "", "website": "",
  "headline": "",
  "roles": [
    {"company": "", "title": "", "start": "YYYY-MM or empty",
     "end": "YYYY-MM or 'present'", "location": "",
     "bullets": [{"text": "verbatim bullet", "skills": ["lowercase", "tags"]}]}
  ],
  "skills_extra": ["lowercase skills listed on the resume with no bullet"],
  "education": [{"degree": "", "institution": "", "year": ""}],
  "certifications": [""]
}

Rules:
- Every experience bullet and responsibility line from the resume becomes a
  bank bullet, split one achievement per bullet.
- skills tags: only tools/technologies/methods the bullet itself clearly
  demonstrates. Lowercase. No padding.
- start/end from the resume's dates (YYYY-MM); "present" stays "present".
- The Skills section of the resume goes into skills_extra.
- linkedin/github/website: the profile/portfolio URLs from the resume header,
  exactly as written (empty string if absent).
- headline: the person's current/most recent job title, exactly as written.
- If a field isn't in the resume, use "" or [] — never guess."""


def init_from_resume(
    resume_path: str,
    out_path: str | Path = DEFAULT_MASTER_PATH,
    model: str | None = None,
    host: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    overwrite: bool = False,
) -> tuple[str, list[str]]:
    """Bootstrap an evidence bank from an existing resume via the LLM.

    The LLM only transcribes — the output must still be reviewed by the
    human, who is the truth gate. Returns (written_path, notes).
    """
    p = Path(out_path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists — pass --force to overwrite it.")

    raw_text, _file_type = parsing.extract_text(resume_path)
    if not raw_text.strip():
        raise ValueError(f"Could not extract any text from {resume_path}.")

    parsed, error = llm_client.call_json(
        ATOMIZE_SYSTEM_PROMPT,
        f"RESUME TO TRANSCRIBE:\n{raw_text.strip()}",
        model=model, host=host, api_key=api_key, provider=provider,
        temperature=0.0,
    )
    if error or not isinstance(parsed, dict):
        raise RuntimeError(
            f"LLM atomization failed ({error or 'unexpected payload'}). "
            "Fix the provider with `python cli.py test-llm` and retry, or create "
            "the bank manually from the template."
        )

    contact = parsed.get("contact") or {}
    bank = EvidenceBank(
        name=str(parsed.get("name") or contact.get("name") or ""),
        email=str(parsed.get("email") or contact.get("email") or ""),
        phone=str(parsed.get("phone") or contact.get("phone") or ""),
        location=str(parsed.get("location") or contact.get("location") or ""),
        linkedin=str(parsed.get("linkedin") or contact.get("linkedin") or ""),
        github=str(parsed.get("github") or contact.get("github") or ""),
        website=str(parsed.get("website") or contact.get("website") or ""),
        headline=str(parsed.get("headline") or ""),
        source_path=str(resume_path),
    )
    for r in _as_list(parsed.get("roles")):
        if not isinstance(r, dict):
            continue
        role = Role(
            company=str(r.get("company") or ""),
            title=str(r.get("title") or ""),
            start=str(r.get("start") or ""),
            end=str(r.get("end") or ""),
            location=str(r.get("location") or ""),
        )
        for b in _as_list(r.get("bullets")):
            if isinstance(b, dict):
                text, skills = str(b.get("text") or "").strip(), [
                    str(s).strip().lower() for s in _as_list(b.get("skills"))]
            elif isinstance(b, str):
                text, skills = b.strip(), []
            else:
                continue  # stray list/number: skip, don't reuse the last bullet's text
            if text:
                role.bullets.append(Bullet(text=text, skills=skills))
        if role.company or role.bullets:
            bank.roles.append(role)
    bank.skills_extra = [str(s).strip().lower() for s in _as_list(parsed.get("skills_extra"))]
    for e in _as_list(parsed.get("education")):
        if isinstance(e, dict):
            bank.education.append(Education(
                degree=str(e.get("degree") or ""),
                institution=str(e.get("institution") or ""),
                year=str(e.get("year") or ""),
            ))
    bank.certifications = [str(c) for c in _as_list(parsed.get("certifications")) if str(c).strip()]

    notes = []
    if not bank.roles or not any(r.bullets for r in bank.roles):
        raise RuntimeError("The LLM transcribed no experience bullets — create the bank manually "
                           "from the template instead: `python cli.py init-master` (without --from).")
    notes.extend(bank.validate())
    notes.append("LLM-transcribed from your resume — YOU must now review every bullet for "
                 "accuracy. The bank is the truth constraint; anything wrong here propagates "
                 "into every tailored resume.")

    bank.roles.sort(key=lambda r: (not r.is_current, tuple(-x for x in r.start_key())))
    p.write_text(bank_to_yaml(bank), encoding="utf-8")
    return str(p.resolve()), notes