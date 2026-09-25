"""Layer 1 — Search Visibility: would you come up when a recruiter searches?

Why this replaced the old "ATS score": mainstream ATS platforms (Workday,
Greenhouse, Lever, iCIMS) mostly do not auto-reject on a keyword percentage.
A recruiter opens the applicant pool and SEARCHES it — a job title, plus a
few must-have skills, with OR for tools they'd accept interchangeably. A
resume that doesn't match those searches is never opened, even though no
machine "rejected" it. So the honest question for this layer is:

    "If a recruiter ran the searches this JD implies, how many would I show up in?"

The layer has three parts, reported separately and never blended:

  1. Parse safety   — a GATE, not a score. If the ATS can't read the file
                      (text extraction failed, two columns, no contact info),
                      no search will find you, so this is pass/fail with the
                      reasons listed.
  2. Search visibility — the headline number: share of the simulated
                      recruiter searches your resume matches. Every search is
                      printed with what it hit and what it missed, so the
                      score is auditable rather than a black-box percentage.
  3. LLM fit read   — the semantic layer, kept but labelled for what it is:
                      a model's reading of meaning-level fit, NOT something an
                      ATS computes.

The searches are deterministic and built only from the JD itself: its title,
and its highest-weighted requirement terms. Tools are OR-ed only when the JD
names them side by side on one line ("Power BI or Tableau") — the tool never
assumes a recruiter would accept a substitute the JD didn't offer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .keywords import JDKeyword
from .parsing import ParseResult
from .terms import alias_normalize, canonical, term_pattern

# Terms recruiters don't type into a search box: soft skills and words too
# generic to narrow an applicant pool. They still count in the keyword
# coverage diagnostics — just not as searches.
NOT_SEARCHED = {
    "communication", "presentation", "problem solving", "critical thinking",
    "attention to detail", "time management", "public speaking", "mentoring",
    "interpersonal skills", "cross-functional collaboration", "client-facing",
    "documentation", "report writing", "insights", "analytics", "cloud",
    "consulting", "specification", "stakeholder management", "negotiation",
    "training delivery", "domain expertise", "delivery team",
}

# Interchangeable tools. Two terms are OR-ed only when they're in the same
# family AND the JD names both on the same line.
FAMILIES = [
    {"power bi", "tableau", "looker", "looker studio", "qlik", "qliksense", "microstrategy"},
    {"aws", "azure", "gcp"},
    {"snowflake", "bigquery", "redshift", "databricks", "synapse", "data warehouse"},
    {"postgresql", "mysql", "sql server", "oracle", "db2", "sqlite"},
    {"python", "r", "sas", "spss", "stata"},
    {"java", "c#", "golang", "scala", "rust"},
    {"react", "angular", "vue", "next.js"},
    {"jira", "asana", "trello", "monday.com"},
    {"salesforce", "hubspot", "zoho", "dynamics 365"},
    {"airflow", "dbt", "matillion", "fivetran", "airbyte", "ssis"},
    {"docker", "kubernetes"},
    {"terraform", "ansible"},
    {"jenkins", "github actions", "gitlab ci", "azure devops"},
]

MAX_SKILL_TERMS = 6


@dataclass
class ParseCheck:
    name: str
    status: str   # pass | warn | fail
    detail: str


@dataclass
class Search:
    name: str
    groups: list[list[str]]            # AND of OR-groups (canonical terms)
    matched: bool = False
    missing: list[str] = field(default_factory=list)   # groups that missed, rendered

    @property
    def query(self) -> str:
        return " AND ".join(_render_group(g) for g in self.groups)


@dataclass
class VisibilityResult:
    score: float | None                     # % of searches matched; None = nothing to search
    searches: list[Search] = field(default_factory=list)
    parse_safe: bool = True
    parse_checks: list[ParseCheck] = field(default_factory=list)
    title: str = ""
    title_found: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def matched_count(self) -> int:
        return sum(1 for s in self.searches if s.matched)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "searches_matched": self.matched_count,
            "searches_total": len(self.searches),
            "searches": [
                {"name": s.name, "query": s.query, "matched": s.matched, "missing": s.missing}
                for s in self.searches
            ],
            "jd_title": self.title,
            "title_found": self.title_found,
            "parse_gate": {
                "safe": self.parse_safe,
                "checks": [{"name": c.name, "status": c.status, "detail": c.detail}
                           for c in self.parse_checks],
            },
            "notes": self.notes,
        }


# ------------------------------------------------------------ parse gate

def parse_gate(pr: ParseResult) -> tuple[bool, list[ParseCheck]]:
    """Can an ATS read this file at all? Fails are the things that stop a
    parser outright; warns degrade it. 'Summary' is optional on a resume, so
    its absence isn't checked here."""
    checks: list[ParseCheck] = []

    def add(name, ok, detail_ok, detail_bad, bad="fail"):
        checks.append(ParseCheck(name, "pass" if ok else bad, detail_ok if ok else detail_bad))

    add("Text extraction", pr.word_count >= 100,
        f"{pr.word_count} words extracted.",
        f"Only {pr.word_count} words extracted — likely a scanned image, text-as-graphics "
        "or an encoding problem. An ATS would index almost nothing.")
    add("Single-column layout", not pr.multi_column_risk,
        "No multi-column layout detected.",
        "Two-column layout detected — parsers read across the page and scramble it.")
    add("No tables", not pr.has_tables,
        "No tables.", "Tables found — some parsers drop table content.", bad="warn")
    add("Email", pr.has_email, "Email found.", "No email found — contact fields won't populate.")
    add("Phone", pr.has_phone, "Phone found.", "No phone number found.", bad="warn")
    core_missing = [s for s in pr.sections_missing if s in ("experience", "education", "skills")]
    add("Standard section headers", not core_missing,
        "Experience / Education / Skills headers found.",
        "Missing header(s): " + ", ".join(core_missing)
        + " — parsers use these to split the resume into fields.", bad="warn")

    return all(c.status != "fail" for c in checks), checks


# ------------------------------------------------------------ searches

def _render_group(group: list[str]) -> str:
    parts = [f'"{t}"' if " " in t else t for t in group]
    return parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"


def _family(term: str) -> int | None:
    for i, fam in enumerate(FAMILIES):
        if term in fam:
            return i
    return None


def _terms_by_line(jd_text: str, terms: list[str]) -> list[set[str]]:
    lines = []
    for line in jd_text.splitlines():
        hay = alias_normalize(line.lower())
        found = {t for t in terms if term_pattern(t).search(hay)}
        if found:
            lines.append(found)
    return lines


def _skill_groups(jd_text: str, jd_keywords: list[JDKeyword], title: str) -> list[list[str]]:
    """The JD's top searchable requirement terms as OR-groups, strongest first."""
    # "BI" in a Business Intelligence JD is the title again, not a skill
    words = title.split()
    initials = {"".join(w[0] for w in words[i:j])
                for i in range(len(words)) for j in range(i + 2, len(words) + 1)}
    ranked: list[tuple[str, float]] = []
    seen: set[str] = set()
    for k in sorted(jd_keywords, key=lambda k: -k.weight):
        c = canonical(k.term)
        if (c in seen or c in NOT_SEARCHED or k.section not in ("hard", "skills")
                or c in initials or (title and term_pattern(c).search(title))):
            continue
        seen.add(c)
        ranked.append((c, k.weight))
    ranked = ranked[:MAX_SKILL_TERMS]
    if not ranked:
        return []

    # OR two same-family terms only when one JD line names both
    co_lines = _terms_by_line(jd_text, [t for t, _ in ranked])
    groups: list[list[str]] = []
    placed: dict[str, int] = {}
    for term, _w in ranked:
        fam = _family(term)
        target = None
        if fam is not None:
            for other, gi in placed.items():
                if _family(other) == fam and any({term, other} <= ln for ln in co_lines):
                    target = gi
                    break
        if target is None:
            placed[term] = len(groups)
            groups.append([term])
        else:
            groups[target].append(term)
            placed[term] = target
    return groups


def _matches(group: list[str], hay: str) -> bool:
    return any(term_pattern(t).search(hay) for t in group)


def build_searches(title: str, groups: list[list[str]]) -> list[Search]:
    """Recruiter-style searches, loosest to strictest. Built from the JD's
    title and requirement groups only."""
    searches: list[Search] = []
    t = [[title]] if title else []
    g = groups
    if t:
        searches.append(Search("Job title", t))
    if t and g:
        searches.append(Search("Title + top requirement", t + g[:1]))
    if len(g) >= 2:
        searches.append(Search("Top two requirements", g[:2]))
    if len(g) >= 3:
        searches.append(Search("Core stack (top three)", g[:3]))
    if t and len(g) >= 3:
        searches.append(Search("Title + core stack", t + g[:3]))
    if len(g) >= 4:
        searches.append(Search(f"All top requirements ({len(g)})", g))
    return searches


def score_visibility(
    resume_text: str,
    jd_text: str,
    jd_keywords: list[JDKeyword],
    jd_title: str,
    parse_result: ParseResult,
) -> VisibilityResult:
    safe, checks = parse_gate(parse_result)
    hay = alias_normalize((resume_text or "").lower())
    title = alias_normalize(jd_title.lower()) if jd_title else ""

    groups = _skill_groups(jd_text, jd_keywords, title)
    searches = build_searches(title, groups)
    for s in searches:
        missed = [grp for grp in s.groups if not _matches(grp, hay)]
        s.matched = not missed
        s.missing = [_render_group(grp) for grp in missed]

    result = VisibilityResult(
        score=round(sum(s.matched for s in searches) / len(searches) * 100, 1) if searches else None,
        searches=searches, parse_safe=safe, parse_checks=checks, title=title,
        title_found=_matches([title], hay) if title else None,
    )
    if not title:
        result.notes.append("No role title found in the JD's first lines — title searches skipped.")
    if not groups:
        result.notes.append("No searchable requirement terms found under a Requirements/Skills "
                            "heading — skill searches skipped.")
    if not safe:
        result.notes.append("Parse gate FAILED: until the file parses, an ATS may index too little "
                            "for any search to find you — fix the parse issues first.")
    return result
