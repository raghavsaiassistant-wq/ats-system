"""Evidence selection — budgeted maximum weighted coverage, with saturation.

Given the JD's weighted keywords and the evidence bank, choose which bullets
appear in this resume. A real resume has finite *space*, not a finite
bullet count, so the budget is in estimated lines and every bullet pays its
own line cost — a 40-word bullet costs more than an 8-word one.

Method: greedy on marginal value-per-line, run JOINTLY across roles against
one shared covered-set and one shared budget:

  - Marginal, not static: each pick is scored against what nothing already
    chosen covers, and the scores are recomputed between picks. Two bullets
    that both cover "SQL" don't both earn it twice.
  - Saturation, not binary: the first mention of a JD term earns its full
    weight; a second mention — SQL across two roles reads to a recruiter as
    *sustained use*, which is evidence — earns 0.3x; a third 0.1x; beyond
    that 0. Reinforcement with space to spare is kept; genuine stuffing
    earns nothing.
  - Structure, not just coverage: a mandatory minimum per role (4 for the
    current role, 2 for earlier ones, when available) runs FIRST — a
    one-bullet role reads as a red flag to a human, and the recruiter
    layer's tenure/legibility checks parse the timeline those bullets make.
    Each role's first bullet always survives (structure); further
    mandatory picks must fit the shared line budget like everything else.
  - Per-role caps (5 current / 3 earlier) bound any single role's share of
    the budget.

No approximation guarantee is claimed: plain greedy under a knapsack-style
budget has no constant factor, and the bound that DOES exist for greedy
max-coverage ((1-1/e)) applies to cardinality-constrained selection, which
this is not. At this scale (dozens of bullets, ~40 terms) greedy is
near-optimal in practice, and the quality lever is saturation + a real
budget, not a smarter search — do not optimise the loop.

Alias canonicalization (PowerBI == power-bi == Power BI, PostgreSQL ==
postgres) runs before matching so the generator never "adds" a keyword the
resume already carries under a different spelling — accidental keyword
stuffing is exactly what the manager layer penalises.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..keywords import JDKeyword
from ..terms import ALIASES, alias_normalize, canonical, term_pattern
from .evidence_bank import Bullet, EvidenceBank, Role

# Alias canonicalization (PowerBI == power-bi == Power BI, PostgreSQL ==
# postgres) lives in ats_checker.terms and is SHARED with the scoring layer,
# so the generator can never "add" a keyword the resume already carries
# under a different spelling — and the scorer can never report a term
# missing that the generator knows is covered. Accidental keyword stuffing
# is exactly what the manager layer penalises.

# Value multiplier by how many times the term is already covered.
SATURATION = (1.0, 0.3, 0.1)   # 1st mention full, 2nd 0.3x, 3rd 0.1x, 4th+ 0

# Rough wrap estimate for the line budget: a resume line holds ~12 words.
WORDS_PER_LINE = 12


def lines_cost(bullet: Bullet) -> int:
    """Estimated resume lines this bullet occupies (>= 1)."""
    return max(1, math.ceil(len(bullet.text.split()) / WORDS_PER_LINE))


def text_covers(text: str, jd_map: dict[str, float]) -> set[str]:
    """Which canonical JD terms does this text demonstrate?"""
    haystack = alias_normalize((text or "").lower())
    return {term for term in jd_map if term_pattern(term).search(haystack)}


def bullet_covers(bullet: Bullet, jd_map: dict[str, float]) -> set[str]:
    covered = text_covers(bullet.text, jd_map)
    for skill in bullet.skills:
        c = canonical(skill)
        if c in jd_map:
            covered.add(c)
    return covered


def build_jd_map(jd_keywords: list[JDKeyword]) -> tuple[dict[str, float], dict[str, JDKeyword]]:
    """canonical term -> weight, and canonical term -> a representative
    JDKeyword (for reporting). Terms that canonicalize together merge with
    MAX, not sum: a requirement written three ways in one JD is still one
    requirement — repetition doesn't make it more required."""
    weights: dict[str, float] = {}
    reps: dict[str, JDKeyword] = {}
    for k in jd_keywords:
        c = canonical(k.term)
        weights[c] = max(weights.get(c, 0.0), k.weight)
        reps.setdefault(c, k)
    return weights, reps


@dataclass
class ChosenBullet:
    """A bank bullet selected for this resume. `text` starts verbatim and may
    be replaced by a VERIFIED rewording during optimisation — the atom it
    traces back to is `bullet`, which is the provenance record."""
    bullet: Bullet
    role: Role
    text: str
    covered: set[str] = field(default_factory=set)


@dataclass
class Selection:
    chosen: list[ChosenBullet] = field(default_factory=list)   # render order: strongest first
    covered: set[str] = field(default_factory=set)            # JD terms covered anywhere in the resume
    covered_by_bullets: set[str] = field(default_factory=set)  # terms backed by shown evidence
    uncovered: list[JDKeyword] = field(default_factory=list)   # honest gaps, weight-sorted
    covered_by_skills_section: set[str] = field(default_factory=set)
    total_lines: int = 0                                       # estimated bullet lines used


def select(
    bank: EvidenceBank,
    jd_keywords: list[JDKeyword],
    max_current: int = 5,
    max_other: int = 3,
    max_lines: int = 26,
) -> Selection:
    """Mandatory-minimum picks first (timeline credibility), then joint
    greedy on saturated marginal value-per-line under a shared line budget
    and per-role caps. All picks within a role are ordered strongest-first."""
    jd_map, reps = build_jd_map(jd_keywords)

    roles = sorted(bank.roles, key=lambda r: (not r.is_current, tuple(-x for x in r.start_key())))
    covers = {
        id(role): {id(b): bullet_covers(b, jd_map) for b in role.bullets}
        for role in roles
    }
    remaining = {id(role): list(role.bullets) for role in roles}
    picked: dict[int, list[Bullet]] = {id(role): [] for role in roles}

    covered_count: dict[str, int] = {}

    def value(b: Bullet) -> float:
        """Saturated marginal value: full weight for a term nothing covers
        yet, 0.3x for the second mention, 0.1x for the third, 0 after."""
        v = 0.0
        for t in _covers_of(b):
            n = covered_count.get(t, 0)
            if n < len(SATURATION):
                v += jd_map[t] * SATURATION[n]
        return v

    def _covers_of(b: Bullet) -> set[str]:
        for role in roles:
            if id(b) in covers[id(role)]:
                return covers[id(role)][id(b)]
        return set()

    def record(b: Bullet) -> None:
        for t in _covers_of(b):
            covered_count[t] = covered_count.get(t, 0) + 1

    def pick_key(b: Bullet):
        return (value(b), len(b.skills), -len(b.text))

    # ---- Phase A: mandatory minimums. Structure first: every role shows a
    # real timeline; the current role is never a two-bullet skeleton.
    # Budget discipline: the FIRST bullet of each role is always kept (a
    # role header with no bullets reads as a red flag), but every further
    # mandatory pick must fit the shared line budget — with a long career
    # history the budget is a real constraint, and blowing past it silently
    # produced multi-page resumes nobody warned about.
    used_lines = 0
    for role in roles:
        need = min(4 if role.is_current else 2, len(role.bullets))
        taken = 0
        while taken < need and remaining[id(role)]:
            affordable = [
                b for b in remaining[id(role)]
                if used_lines + lines_cost(b) <= max_lines
            ]
            if not affordable:
                if taken == 0:
                    # structural minimum: never render a role with zero bullets
                    affordable = list(remaining[id(role)])
                else:
                    break
            best = max(affordable, key=pick_key)
            remaining[id(role)].remove(best)
            picked[id(role)].append(best)
            record(best)
            used_lines += lines_cost(best)
            taken += 1

    # ---- Phase B: joint greedy under the shared budget and per-role caps.
    while True:
        best: tuple | None = None
        best_ratio = 0.0
        for role in roles:
            cap = max_current if role.is_current else max_other
            if len(picked[id(role)]) >= cap:
                continue
            for b in remaining[id(role)]:
                cost = lines_cost(b)
                if used_lines + cost > max_lines:
                    continue
                v = value(b)
                ratio = v / cost
                if ratio > best_ratio or (ratio == best_ratio and best and v > best[2]):
                    best = (role, b, v, cost)
                    best_ratio = ratio
        if best is None:
            break
        role, b, v, cost = best
        remaining[id(role)].remove(b)
        picked[id(role)].append(b)
        record(b)
        used_lines += cost

    chosen: list[ChosenBullet] = []
    for role in roles:
        for b in picked[id(role)]:
            chosen.append(ChosenBullet(bullet=b, role=role, text=b.text, covered=covers[id(role)][id(b)]))

    # skills the Skills section covers for free (they're always listed)
    skills_covered = {canonical(s) for s in bank.all_skills()} & set(jd_map)
    bullet_covered = {t for t, n in covered_count.items() if n > 0}

    covered = bullet_covered | skills_covered
    uncovered = sorted(
        (reps[t] for t in set(jd_map) - covered),
        key=lambda k: k.weight, reverse=True,
    )
    return Selection(
        chosen=chosen,
        covered=covered,
        covered_by_bullets=bullet_covered,
        uncovered=uncovered,
        covered_by_skills_section=skills_covered,
        total_lines=used_lines,
    )