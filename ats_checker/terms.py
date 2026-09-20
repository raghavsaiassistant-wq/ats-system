"""Shared term machinery — alias canonicalization used by BOTH the scoring
layers and the generator.

Why this module exists: the scoring layer and the generator used to disagree
about synonyms. `score_keywords` did literal string matching while the
generator's selector canonicalized aliases — so a resume containing
"PostgreSQL" scored as MISSING a JD asking for "Postgres", even though the
tool's own alias table knows they are identical. With one shared home for the
alias rules, the measurement and the optimisation can never disagree again.

This module deliberately imports nothing from the rest of the package —
`keywords.py` imports it, and everything that needs aliasing imports
`keywords` or this module, so there is no cycle.
"""
from __future__ import annotations

import re

# variant -> canonical. Matched case-insensitively with word boundaries.
ALIASES: dict[str, str] = {
    "powerbi": "power bi", "power-bi": "power bi", "power_bi": "power bi",
    "ms power bi": "power bi", "powerbi desktop": "power bi",
    "postgres": "postgresql", "postgre sql": "postgresql", "postgressql": "postgresql",
    "ml": "machine learning", "machine-learning": "machine learning",
    "js": "javascript", "nodejs": "node.js", "node js": "node.js", "node": "node.js",
    "restful api": "rest api", "rest apis": "rest api", "restful apis": "rest api",
    "rest endpoints": "rest api", "restful": "rest api",
    "data-warehouse": "data warehouse", "data warehouses": "data warehouse",
    "dataviz": "data visualization", "data visualisation": "data visualization",
    "data modelling": "data modeling", "data-modeling": "data modeling",
    "ab testing": "a/b testing", "ab tests": "a/b testing", "a/b tests": "a/b testing",
    "powerpoint": "powerpoint", "ms excel": "excel", "microsoft excel": "excel",
    "ms sql": "sql", "t-sql": "sql", "tsql": "sql", "mssql": "sql",
    "genai": "generative ai", "power platform": "power platform",
    "stakeholder communications": "stakeholder management",
    "dashboarding": "dashboard",
    # ---- added: common spellings the scorer and generator both need ----
    "kubernetes": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci/cd", "cicd": "ci/cd", "ci cd": "ci/cd", "continuous integration": "ci/cd",
    "continuous delivery": "ci/cd", "continuous deployment": "ci/cd",
    "devops": "devops",
    "ms azure": "azure", "microsoft azure": "azure", "amazon web services": "aws",
    "google cloud": "gcp", "google cloud platform": "gcp",
    "business intelligence": "business intelligence", "bi analyst": "business intelligence analyst",
    "data analytics": "data analysis",
    "qa": "quality assurance",
    "data engineer": "data engineering", "data engineers": "data engineering",
    "user experience": "ux", "user interface": "ui",
    "ecommerce": "ecommerce", "e-commerce": "ecommerce",
    "artificial intelligence": "ai", "gen ai": "generative ai",
    "ms sql server": "sql server",
    "excel vba": "vba",
    "m365": "microsoft 365", "office 365": "microsoft 365",
    "ms teams": "microsoft teams",
    # soft-skill phrases canonicalize to their skill word so a resume that
    # lists "Communication" matches a JD asking for "communication skills"
    "communication skills": "communication",
    "presentation skills": "presentation",
    "interpersonal skills": "interpersonal skills",
    # "BA" in business-analyst JDs means business analysis — canonicalize so
    # a resume's real "business analysis" work matches the JD's "BA s" token,
    # on both the scoring and the generation side (shared table).
    "ba": "business analysis",
}

# Aliases sorted longest-first so multiword variants apply before their substrings.
_ALIAS_ORDER = sorted(ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True)
_ALIAS_RES = [
    (re.compile(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", re.I), canonical_form)
    for variant, canonical_form in _ALIAS_ORDER
]


def canonical(term: str) -> str:
    """Lowercase + alias-resolve one term."""
    t = (term or "").strip().lower()
    return ALIASES.get(t, t)


def alias_normalize(text: str) -> str:
    """Rewrite alias variants in a text to their canonical form, so that
    'PowerBI' and 'power bi' compare equal during matching."""
    out = text
    for pattern, canonical_form in _ALIAS_RES:
        out = pattern.sub(canonical_form, out)
    return out


def term_pattern(term: str) -> re.Pattern:
    """Word-boundary pattern for a canonical term, tolerant of a simple
    plural 's' ("dashboard" matches "dashboards") so the scorer and the
    generator agree on near-identical spellings without full stemming."""
    t = canonical(term)
    base = re.escape(t)
    if t.endswith("s") or len(t) <= 2:
        return re.compile(rf"(?<![a-z0-9]){base}(?![a-z0-9])")
    return re.compile(rf"(?<![a-z0-9]){base}s?(?![a-z0-9])")