"""Deterministic assembly — bank atoms + selection -> resume text.

No LLM here, on purpose. The output structure is fixed: single column,
standard section headers ('Summary', 'Experience', 'Skills', 'Education'),
parseable 'Mon YYYY - Mon YYYY' dates, email and phone in the header. That
means layer 1's formatting/parseability score is maxed BY CONSTRUCTION —
it's the one component the generator can guarantee rather than measure.

Bullet order within a role is the selector's pick order (strongest JD
coverage first), which directly optimises the recruiter layer's top-third
visibility check: highest-weight evidence lands where recruiters actually
read.
"""
from __future__ import annotations

import re
from collections import defaultdict

from ..keywords import JDKeyword
from .evidence_bank import EvidenceBank
from .selector import Selection, build_jd_map, canonical

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# How each canonical skill term renders in the Skills line / summary.
# Terms not listed fall back to title-case (or upper for short acronyms).
DISPLAY = {
    "sql": "SQL", "power bi": "Power BI", "dax": "DAX", "python": "Python",
    "excel": "Excel", "n8n": "n8n", "api": "API", "rest api": "REST API",
    "graphql": "GraphQL", "etl": "ETL", "elt": "ELT", "kpi": "KPI",
    "aws": "AWS", "azure": "Azure", "gcp": "GCP", "erp": "ERP", "crm": "CRM",
    "sap": "SAP", "ml": "ML", "ai": "AI", "nlp": "NLP", "llm": "LLM",
    "gpt": "GPT", "git": "Git", "github": "GitHub", "docker": "Docker",
    "kubernetes": "Kubernetes", "jira": "Jira", "agile": "Agile",
    "scrum": "Scrum", "pmp": "PMP", "cpa": "CPA", "cfa": "CFA", "mba": "MBA",
    "dax queries": "DAX Queries", "snowflake": "Snowflake",
    "bigquery": "BigQuery", "redshift": "Redshift", "tableau": "Tableau",
    "looker": "Looker", "spark": "Spark", "airflow": "Airflow",
    "flask": "Flask", "django": "Django", "fastapi": "FastAPI",
    "javascript": "JavaScript", "typescript": "TypeScript", "react": "React",
    "node.js": "Node.js", "html": "HTML", "css": "CSS", "java": "Java",
    "c++": "C++", "c#": "C#", "r": "R", "sas": "SAS", "spss": "SPSS",
    "machine learning": "Machine Learning", "statistics": "Statistics",
    "a/b testing": "A/B Testing", "forecasting": "Forecasting",
    "data warehouse": "Data Warehouse", "data modeling": "Data Modeling",
    "data visualization": "Data Visualization",
    "dashboard": "Dashboard", "reporting": "Reporting",
    "automation": "Automation", "powerpoint": "PowerPoint",
    "stakeholder management": "Stakeholder Management",
    "project management": "Project Management",
    "microsoft copilot studio": "Microsoft Copilot Studio",
    "claude": "Claude", "openai": "OpenAI", "chatgpt": "ChatGPT",
    "ollama": "Ollama", "salesforce": "Salesforce", "workday": "Workday",
    "six sigma": "Six Sigma", "nosql": "NoSQL", "mongodb": "MongoDB",
    "postgresql": "PostgreSQL", "mysql": "MySQL",
    "generative ai": "Generative AI", "copilot studio": "Copilot Studio",
    "data analysis": "Data Analysis", "analytics": "Analytics",
    "callminer": "CallMiner", "eql": "EQL",
}


def display_case(term: str) -> str:
    t = canonical(term)
    if t in DISPLAY:
        return DISPLAY[t]
    if len(t) <= 3 and t.isalpha():
        return t.upper()
    return " ".join(w.capitalize() for w in t.split())


def _fmt_month(ym: str) -> str:
    from .evidence_bank import parse_ym
    parsed = parse_ym(ym)
    if not parsed:
        return ""
    year, month = parsed
    return f"{MONTH_NAMES[month - 1]} {year}"


def _role_dates(role) -> str:
    start = _fmt_month(role.start) or "?"
    if role.is_current:
        end = "Present"
    else:
        end = _fmt_month(role.end) or "?"
    return f"{start} - {end}"


def _join_and(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _skills_ordered(bank: EvidenceBank, selection: Selection,
                    jd_map: dict[str, float], max_skills: int = 15) -> list[str]:
    """Skills the resume can stand behind, in priority order, bounded.

    Two rules:
      - Only skills demonstrated by SELECTED bullets, or declared in
        skills_extra (which the user vouches for by putting it in the bank).
        Skills whose only backing bullet wasn't selected don't appear — a
        listed-but-never-shown skill is exactly the 'evidence_backing' risk
        the manager layer penalises.
      - Capped at max_skills total. An unbounded skills line is a keyword
        dump; the JD-covered ones lead, ordered by weight.
    """
    bullet_skills: dict[str, None] = {}
    for cb in selection.chosen:
        for s in cb.bullet.skills:
            bullet_skills.setdefault(canonical(s), None)
    extras = {canonical(s) for s in bank.skills_extra} - set(bullet_skills)

    covered = [s for s in bullet_skills if s in jd_map] + [s for s in extras if s in jd_map]
    covered.sort(key=lambda s: (-jd_map.get(s, 0.0), s))
    rest_bullet = sorted(set(bullet_skills) - set(covered))
    rest_extra = sorted(extras - set(covered))

    ordered = (covered + rest_bullet + rest_extra)[:max_skills]
    return [display_case(s) for s in ordered]


def assemble(
    bank: EvidenceBank,
    selection: Selection,
    jd_keywords: list[JDKeyword],
    headline: str | None = None,
) -> str:
    """Render the tailored resume as plain text. Structure is fixed; the
    only variable parts are WHICH bullets appear and their order."""
    jd_map, _reps = build_jd_map(jd_keywords)

    head = (headline or bank.headline or (bank.roles[0].title if bank.roles else "")).strip()

    lines: list[str] = []

    # ---- header (email + phone + profile links here: layer-1 parseability
    # and recruiter expectations both read the contact line)
    lines.append(bank.name)
    contact_bits = [
        x for x in (bank.email, bank.phone, bank.location,
                    bank.linkedin, bank.github, bank.website) if x
    ]
    if contact_bits:
        lines.append(" | ".join(contact_bits))

    # ---- summary: only claims the SHOWN EVIDENCE backs. Terms covered solely
    # via skills_extra are skills you hold, not demonstrated experience —
    # "hands-on experience in X" for those would be an overclaim, and this
    # module never lets the deterministic path lie.
    top_covered = sorted(selection.covered_by_bullets, key=lambda t: -jd_map.get(t, 0.0))[:5]
    if top_covered:
        claims = [display_case(t) for t in top_covered]
    else:
        # No bullet-backed JD terms — fall back to the selected bullets' own
        # skill tags, which are backed by definition, in pick (strongest) order.
        seen: dict[str, None] = {}
        for cb in selection.chosen:
            for s in cb.bullet.skills:
                seen.setdefault(canonical(s), None)
        claims = [display_case(s) for s in list(seen)[:5]]
    summary = f"{head} with hands-on experience in {_join_and(claims)}." if claims \
        else f"{head} with professional experience delivering reporting and analysis."
    lines += ["", "Summary", summary]

    # ---- skills
    lines += ["", "Skills", ", ".join(_skills_ordered(bank, selection, jd_map))]

    # ---- experience: roles in selection order, bullets strongest-first
    lines += ["", "Experience"]
    by_role: dict[int, list] = {}
    role_order: list[int] = []
    for cb in selection.chosen:
        key = id(cb.role)
        if key not in by_role:
            by_role[key] = []
            role_order.append(key)
        by_role[key].append(cb)
    for key in role_order:
        group = by_role[key]
        role = group[0].role
        where = f", {role.location}" if role.location else ""
        lines.append(f"{role.title} - {role.company}{where} ({_role_dates(role)})")
        for cb in group:
            lines.append(f"- {cb.text}")
        lines.append("")

    # ---- education
    lines.append("Education")
    for e in bank.education:
        bits = " - ".join(x for x in (e.degree, e.institution) if x)
        if e.year:
            bits += f" ({e.year})"
        lines.append(bits)

    # ---- certifications
    if bank.certifications:
        lines += ["", "Certifications"]
        lines.extend(bank.certifications)

    return "\n".join(lines).rstrip() + "\n"


def write_docx(text: str, path: str) -> str:
    """Write the assembled resume as a plain .docx — no tables, no columns,
    no text boxes: exactly what ATS parsers read cleanly."""
    import docx

    doc = docx.Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(path)
    return path