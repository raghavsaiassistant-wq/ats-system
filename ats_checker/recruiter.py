"""HR / Recruiter Screen layer — the ~30-second human filter after the ATS,
plus what HR will actually expect from you on the call.

Two kinds of finding live here and they are NOT the same thing:

  RESUME    — fixable by editing the document (evidence buried on page two,
              dates a recruiter can't read, title not visible up top)
  CANDIDACY — facts about you (notice period, sponsorship, years, work mode).
              No rewrite changes these. They're either compatible with the
              role or they aren't, and mostly they get captured by the
              application form's knockout questions or the screening call,
              not by your resume at all.

Both are reported, separately scored, and never blended into one
undifferentiated number — because "70 because your evidence is buried" and
"70 because your notice period is long" need completely different responses.

Scores here are "% of stated criteria met", which is a measurement. They are
not probabilities of passing, which would need outcome data to claim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .jd_requirements import (
    COUNTRY_ALIASES, JDRequirements, detect_countries, normalize_country, seniority_rank,
)
from .profile import EDUCATION_RANK, CandidateProfile
from .terms import alias_normalize

RESUME, CANDIDACY = "resume", "candidacy"

# Importance weights — hard gates a recruiter filters on instantly vs
# softer signals that shape the conversation.
GATE, STRONG, SOFT = 3.0, 2.0, 1.0

# Credit earned toward "criteria met" by status
CREDIT = {"pass": 1.0, "warn": 0.5, "fail": 0.0}

_COUNTRY_WORDS = {w for name in COUNTRY_ALIASES for w in name.split()}

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

EDUCATION_LINE = re.compile(
    r"\b(university|college|institute|school|b\.?tech|bachelor|master|mba|ph\.?d|"
    r"b\.?sc|m\.?sc|b\.?b\.?a|degree|gpa|cgpa)\b",
    re.I,
)

SENIORITY_WORD = re.compile(
    r"\b(intern|trainee|junior|jr\.?|associate|senior|sr\.?|staff|principal|lead|head of|director|vp|manager)\b",
    re.I,
)

DATE_RANGE_RE = re.compile(
    r"(?P<start>"
    r"(?:(?P<smon>jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+)?"
    r"(?P<syear>(?:19|20)\d{2})"
    r"|(?P<smm>0?[1-9]|1[0-2])/(?P<syear2>(?:19|20)\d{2})"
    r")"
    r"\s*(?:-|–|—|to|until|through)\s*"
    r"(?P<end>present|current|now|"
    r"(?:(?P<emon>jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+)?"
    r"(?P<eyear>(?:19|20)\d{2})"
    r"|(?P<emm>0?[1-9]|1[0-2])/(?P<eyear2>(?:19|20)\d{2})"
    r")",
    re.I,
)


@dataclass
class Check:
    name: str
    status: str            # pass | warn | fail | skipped
    category: str          # resume | candidacy
    detail: str
    importance: float = SOFT
    jd_citation: str = ""


@dataclass
class Expectation:
    """Something HR will raise on the screening call, and how to be ready."""
    topic: str
    why: str
    prepare: str
    source: str = "derived"   # "derived" from a check, or "standard" screen item


@dataclass
class RecruiterResult:
    criteria_met_pct: float             # headline: % of applicable criteria met
    resume_pct: float | None            # % met among resume-fixable checks
    candidacy_pct: float | None         # % met among candidacy-fact checks
    checks: list[Check] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    expectations: list[Expectation] = field(default_factory=list)
    top_third_coverage: float | None = None
    employment_gaps: list[str] = field(default_factory=list)
    average_tenure_months: float | None = None

    @property
    def score(self) -> float:
        """Kept for compatibility — the headline criteria-met percentage."""
        return self.criteria_met_pct

    def to_dict(self) -> dict:
        return {
            "criteria_met_pct": self.criteria_met_pct,
            "resume_fixable_pct": self.resume_pct,
            "candidacy_fit_pct": self.candidacy_pct,
            "blockers": self.blockers,
            "top_third_coverage": self.top_third_coverage,
            "employment_gaps": self.employment_gaps,
            "average_tenure_months": self.average_tenure_months,
            "checks": [
                {
                    "name": c.name, "status": c.status, "category": c.category,
                    "detail": c.detail, "importance": c.importance,
                    "jd_citation": c.jd_citation,
                }
                for c in self.checks
            ],
            "hr_expectations": [
                {"topic": e.topic, "why": e.why, "prepare": e.prepare, "source": e.source}
                for e in self.expectations
            ],
        }


# ---------------------------------------------------------------- timeline


def _month_year(mon, year, mm, yy2) -> tuple[int, int] | None:
    if year:
        m = MONTHS.get((mon or "").lower()[:4].rstrip("."), 1) if mon else 1
        return int(year), m
    if yy2 and mm:
        return int(yy2), int(mm)
    return None


def extract_date_ranges(resume_text: str) -> list[tuple[tuple[int, int], tuple[int, int] | None]]:
    """Employment-looking date ranges. None end == 'Present'. Education lines
    are skipped so a degree overlapping a job doesn't read as a fake gap."""
    ranges = []
    for line in resume_text.splitlines():
        if EDUCATION_LINE.search(line):
            continue
        for m in DATE_RANGE_RE.finditer(line):
            start = _month_year(m.group("smon"), m.group("syear"), m.group("smm"), m.group("syear2"))
            if not start:
                continue
            end_raw = (m.group("end") or "").lower()
            if end_raw in ("present", "current", "now"):
                end = None
            else:
                end = _month_year(m.group("emon"), m.group("eyear"), m.group("emm"), m.group("eyear2"))
                if not end:
                    continue
            ranges.append((start, end))
    return ranges


def _months_between(a: tuple[int, int], b: tuple[int, int]) -> int:
    return (b[0] - a[0]) * 12 + (b[1] - a[1])


def analyze_timeline(resume_text: str) -> tuple[list[str], float | None, int]:
    """Return (gap descriptions, average tenure months, number of dated roles)."""
    ranges = extract_date_ranges(resume_text)
    if not ranges:
        return [], None, 0

    today = date.today()
    normalized = [(s, e or (today.year, today.month)) for s, e in ranges]
    normalized.sort(key=lambda r: r[0])

    durations = [max(0, _months_between(s, e)) for s, e in normalized]
    avg_tenure = round(sum(durations) / len(durations), 1) if durations else None

    gaps = []
    furthest_end = normalized[0][1]
    for start, end in normalized[1:]:
        gap_months = _months_between(furthest_end, start)
        if gap_months > 6:
            gaps.append(
                f"~{gap_months} month gap between {furthest_end[1]:02d}/{furthest_end[0]} "
                f"and {start[1]:02d}/{start[0]}"
            )
        if _months_between(furthest_end, end) > 0:
            furthest_end = end

    return gaps, avg_tenure, len(normalized)


def top_third_coverage(resume_text: str, must_have_terms: list[str]) -> float | None:
    """Recruiters read top-down and stop early. What share of the JD's
    highest-weighted terms appear in the first third of the resume?

    The head is alias-normalized (so the shared alias table applies here too)
    and hyphens/spaces compare equal: a resume phrasing 'system-capabilities'
    or 'change-request' still counts for a JD demanding 'system capabilities'
    or 'change request' — the recruiter layer must not disagree with the
    other layers about whether the same evidence covers the same term."""
    if not must_have_terms:
        return None
    words = resume_text.split()
    if len(words) < 30:
        return None
    head = alias_normalize(" ".join(words[: max(30, len(words) // 3)]).lower())
    head_folded = head.replace("-", " ")
    hits = 0
    for term in must_have_terms:
        t = term.lower().replace("-", " ")
        # plural tolerance matching terms.term_pattern: 'change request'
        # matches 'change requests'
        base = re.escape(t) + ("s?" if not t.endswith("s") and len(t) > 2 else "")
        if re.search(rf"(?<![a-z0-9]){base}(?![a-z0-9])", head_folded):
            hits += 1
    return round(hits / len(must_have_terms) * 100, 1)


def _in_top_third(resume_text: str, needle: str) -> bool:
    words = resume_text.split()
    head = " ".join(words[: max(30, len(words) // 3)]).lower()
    return needle.lower() in head


# ------------------------------------------------------------------ scoring


def score_recruiter_screen(
    resume_text: str,
    profile: CandidateProfile,
    reqs: JDRequirements,
    must_have_terms: list[str] | None = None,
) -> RecruiterResult:
    checks: list[Check] = []
    blockers: list[str] = []
    expectations: list[Expectation] = []

    def add(name, status, category, detail, importance=SOFT, citation=""):
        checks.append(Check(name, status, category, detail, importance, citation))
        if status == "fail" and importance >= GATE:
            blockers.append(detail)

    def expect(topic, why, prepare, source="derived"):
        expectations.append(Expectation(topic, why, prepare, source))

    # ============================ CANDIDACY: the hard filters ==============

    # --- Years of experience
    if reqs.min_years is None:
        add("Years of experience", "skipped", CANDIDACY, "JD states no years threshold.")
    elif profile.years_experience is None:
        add("Years of experience", "skipped", CANDIDACY,
            "Set years_experience in profile.yaml to enable this check.")
    else:
        gap = reqs.min_years - profile.years_experience
        cite = reqs.cite("min_years")
        if gap <= 0:
            add("Years of experience", "pass", CANDIDACY,
                f"You have {profile.years_experience:g}y vs {reqs.min_years:g}y required.", GATE, cite)
        elif gap <= 0.5:
            add("Years of experience", "warn", CANDIDACY,
                f"Marginally under: {profile.years_experience:g}y vs {reqs.min_years:g}y required.",
                GATE, cite)
            expect("Experience level",
                   f"You're just under the {reqs.min_years:g}y they asked for.",
                   "Lead with depth, not duration — name the specific projects that gave you "
                   "senior-level exposure earlier than the year count suggests.")
        else:
            status = "warn" if gap <= 2 else "fail"
            add("Years of experience", status, CANDIDACY,
                f"Under by {gap:g}y ({profile.years_experience:g}y vs {reqs.min_years:g}y required)"
                + (" — many recruiters filter on this outright." if status == "fail" else "."),
                GATE, cite)
            expect("Experience level",
                   f"You're {gap:g} years under their stated floor — they will ask about this.",
                   "Have a direct answer for why you're ready anyway: scope handled, systems owned, "
                   "results delivered. Don't apologise for the number, replace it with evidence.")

        if profile.years_experience > reqs.min_years + 6:
            add("Overqualification", "warn", CANDIDACY,
                f"You're {profile.years_experience - reqs.min_years:g}y above the stated floor — "
                "some recruiters screen this out.", SOFT, cite)
            expect("Overqualification",
                   "You're well above the experience they asked for.",
                   "Expect 'why would you take this role?' — have a genuine reason "
                   "(domain, product, scope, stability) that isn't 'I need a job'.")

    # --- Degree
    if not reqs.min_degree:
        add("Education", "skipped", CANDIDACY, "JD states no degree requirement.")
    elif not profile.education_level:
        add("Education", "skipped", CANDIDACY, "Set education_level in profile.yaml.")
    else:
        need, have = EDUCATION_RANK.get(reqs.min_degree, 0), profile.education_rank()
        cite = reqs.cite("min_degree")
        if have >= need:
            add("Education", "pass", CANDIDACY,
                f"Your {profile.education_level} meets the stated {reqs.min_degree}.", GATE, cite)
        elif not reqs.degree_mandatory:
            add("Education", "warn", CANDIDACY,
                f"JD asks for {reqs.min_degree} but accepts equivalent experience.", STRONG, cite)
            expect("Education",
                   f"They ask for {reqs.min_degree}; you have {profile.education_level}, "
                   "but the JD allows equivalent experience.",
                   "Be ready to state plainly what stands in for the degree — years in the "
                   "function, certifications, delivered work.")
        else:
            add("Education", "fail", CANDIDACY,
                f"JD requires {reqs.min_degree}; profile says {profile.education_level}.", GATE, cite)
            expect("Education",
                   f"They list {reqs.min_degree} as a requirement and you don't have it.",
                   "This may be a hard gate on the application form. If you apply, lead so strongly "
                   "on delivered work that a human wants to make an exception.")

    # --- Certifications
    if not reqs.certifications:
        add("Certifications", "skipped", CANDIDACY, "JD names no specific certification.")
    else:
        held = (" ".join(profile.certifications) + " " + resume_text).lower()
        for cert in reqs.certifications:
            cite, mandatory = "", True
            for req in reqs.evidence:
                if req.kind == "certification" and str(req.value).lower() == cert.lower():
                    cite, mandatory = req.source_line, req.mandatory
                    break
            if cert.lower() in held:
                add(f"Certification: {cert}", "pass", CANDIDACY, "Present in your profile/resume.",
                    STRONG, cite)
            elif mandatory:
                add(f"Certification: {cert}", "fail", CANDIDACY,
                    f"{cert} is named as required and isn't in your profile or resume.", STRONG, cite)
                expect(f"Certification: {cert}",
                       f"{cert} is listed as required and you don't hold it.",
                       f"Expect to be asked directly. Best answer is a concrete plan — "
                       f"booked exam date beats 'I'm planning to'.")
            else:
                add(f"Certification: {cert}", "warn", CANDIDACY,
                    f"{cert} listed as preferred — missing, but not a gate.", SOFT, cite)

    # --- Work mode
    if not reqs.work_modes:
        add("Work mode", "skipped", CANDIDACY, "JD doesn't state remote/hybrid/onsite.")
    elif not profile.acceptable_work_modes:
        add("Work mode", "skipped", CANDIDACY, "Set acceptable_work_modes in profile.yaml.")
    else:
        overlap = set(reqs.work_modes) & set(profile.acceptable_work_modes)
        cite = reqs.cite("work_mode")
        if overlap:
            add("Work mode", "pass", CANDIDACY,
                f"JD offers {'/'.join(reqs.work_modes)}; you accept {'/'.join(sorted(overlap))}.",
                GATE, cite)
        else:
            add("Work mode", "fail", CANDIDACY,
                f"JD is {'/'.join(reqs.work_modes)}; your profile accepts only "
                f"{'/'.join(profile.acceptable_work_modes)}.", GATE, cite)
            expect("Work arrangement",
                   f"The role is {'/'.join(reqs.work_modes)} and that's outside what you've said "
                   "you'd accept.",
                   "This is usually a form question and a fast no. Decide before applying whether "
                   "you'd actually flex — don't discover it mid-call.")

    # --- Salary compatibility. The profile's expected_salary fields were
    # collected but never checked before; a JD that states a range is a
    # real filter, and money mismatch kills applications late enough to
    # waste everyone's time.
    if reqs.salary_min is None and reqs.salary_max is None:
        add("Salary", "skipped", CANDIDACY, "JD states no salary range.")
    elif profile.expected_salary_min is None and profile.expected_salary_max is None:
        add("Salary", "skipped", CANDIDACY,
            "Set expected_salary_min/max in profile.yaml to enable this check.")
    else:
        cite = reqs.cite("salary")
        jcur = (reqs.salary_currency or "").upper()
        pcur = (profile.salary_currency or "").upper()
        if jcur and pcur and jcur != pcur:
            add("Salary", "skipped", CANDIDACY,
                f"Currency mismatch — the JD salary range is in {jcur}, your expectation "
                f"in {pcur}; the tool won't compare across currencies.",
                SOFT, cite)
        else:
            cur_txt = pcur or jcur
            jd_min, jd_max = reqs.salary_min, reqs.salary_max
            p_min, p_max = profile.expected_salary_min, profile.expected_salary_max
            if p_min is not None and jd_max is not None and p_min > jd_max:
                add("Salary", "fail", CANDIDACY,
                    f"Your stated minimum ({p_min:g} {cur_txt}) is above their stated maximum "
                    f"({jd_max:g} {cur_txt}).", GATE, cite)
                expect("Salary expectations",
                       "Your floor is above the top of their stated range.",
                       "Either the role is genuinely below your floor — then don't apply — or your "
                       "floor is negotiable for the right role. Decide before the first call, not during it.")
            elif p_max is not None and jd_min is not None and p_max < jd_min:
                add("Salary", "warn", CANDIDACY,
                    f"Their stated minimum ({jd_min:g} {cur_txt}) is above your stated maximum "
                    f"({p_max:g} {cur_txt}) — either out of range or you're underpriced.", STRONG, cite)
            elif p_min is not None and jd_min is not None and p_min > jd_min:
                add("Salary", "warn", CANDIDACY,
                    f"Your minimum ({p_min:g} {cur_txt}) sits in the upper half of their stated range"
                    + (f" ({jd_min:g}–{jd_max:g} {cur_txt})." if jd_max is not None else "."),
                    SOFT, cite)
                expect("Salary expectations",
                       "Your minimum is in the upper part of their range.",
                       "Expect an early money conversation. Have a researched justification for the "
                       "number and know where you can flex — total comp, not just base.")
            else:
                add("Salary", "pass", CANDIDACY,
                    "Your expectations fit inside their stated range.", SOFT, cite)

    # --- Location & relocation. profile.location and open_to_relocation
    # were collected but never checked before — a JD that names a city and
    # a candidate who won't move is a filter, not a formatting note.
    if not reqs.jd_location and not reqs.countries:
        add("Location", "skipped", CANDIDACY, "JD doesn't state a location.")
    elif not profile.location and not profile.work_authorized_in:
        add("Location", "skipped", CANDIDACY, "Set location in profile.yaml.")
    else:
        cite = reqs.cite("location")
        # Countries compare as canonical names, cities as words. Country-name
        # fragments are kept out of the word overlap: "United Arab Emirates"
        # and "United States" share "united", which used to read as a match.
        prof_words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", profile.location)}
        jd_words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", reqs.jd_location or "")}
        city_match = bool((prof_words & jd_words) - _COUNTRY_WORDS)
        authorized = {normalize_country(c) for c in profile.work_authorized_in}
        here = set(detect_countries(profile.location)) | authorized
        same_place = city_match or bool(here & set(reqs.countries))
        where = reqs.jd_location or "/".join(reqs.countries)
        if same_place:
            add("Location", "pass", CANDIDACY,
                f"Role location ({where}) matches where you are or are authorised to work.",
                SOFT, cite)
        elif profile.open_to_relocation is None:
            add("Location", "skipped", CANDIDACY, "Set open_to_relocation in profile.yaml.")
        elif profile.open_to_relocation:
            add("Location", "warn", CANDIDACY,
                f"Role is in {where}; you're in {profile.location} and open to relocating.",
                SOFT, cite)
            expect("Relocation",
                   f"The role is in {where} and you'd be relocating from {profile.location}.",
                   "Say unprompted that you've thought it through: timeline, cost, whether "
                   "you've worked there before. Unrehearsed relocation answers read as risk.")
        else:
            if "onsite" in reqs.work_modes:
                add("Location", "fail", CANDIDACY,
                    f"Role is onsite in {where}; you're in {profile.location} and not open to relocation.",
                    GATE, cite)
                expect("Location",
                       "The role is onsite somewhere you'd need to move to, and you're not open to it.",
                       "This is usually a fast no on the form. Decide whether that's actually "
                       "your answer before applying — don't discover it mid-call.")
            elif "remote" in reqs.work_modes:
                add("Location", "warn", CANDIDACY,
                    f"Role is remote but anchored in {where}; location mostly matters for time zone.",
                    SOFT, cite)
            else:
                add("Location", "warn", CANDIDACY,
                    f"Role appears to be in {where}; you're in {profile.location} and not open to relocation.",
                    STRONG, cite)

    # --- Sponsorship / right to work — country-aware now. When the JD names
    # a country, the check uses your work_authorized_in list instead of
    # guessing from free text; free text remains the fallback.
    if reqs.sponsorship_unavailable:
        cite = reqs.cite("sponsorship_unavailable")
        authorized = {normalize_country(c) for c in profile.work_authorized_in}
        jd_countries = set(reqs.countries)
        if jd_countries and authorized and not (jd_countries & authorized):
            add("Work authorisation", "fail", CANDIDACY,
                f"JD (for {'/'.join(sorted(jd_countries))}) states sponsorship isn't available, and that "
                "isn't among your work_authorized_in countries.", GATE, cite)
            expect("Work authorisation",
                   f"The role is in {'/'.join(sorted(jd_countries))} and the JD says no sponsorship.",
                   "This is almost always a knockout question on the application form — answering "
                   "honestly may auto-reject. Confirm the country reading is right before applying.")
        elif jd_countries and (jd_countries & authorized):
            add("Work authorisation", "pass", CANDIDACY,
                f"You're authorised to work in {'/'.join(sorted(jd_countries & authorized))} "
                "without sponsorship — the no-sponsorship clause doesn't apply to you.",
                GATE, cite)
        else:
            needs_sponsorship = "sponsor" in (profile.work_authorization or "").lower()
            if needs_sponsorship:
                add("Work authorisation", "fail", CANDIDACY,
                    "JD states sponsorship isn't available and your profile says you'd need it. "
                    "Verify against the role's country before applying.", GATE, cite)
                expect("Work authorisation",
                       "The JD says no sponsorship, and your profile indicates you'd need it.",
                       "This is almost always a knockout question on the application form — answering "
                       "honestly may auto-reject. Confirm the role's country first; the tool can't "
                       "infer it reliably.")
            else:
                add("Work authorisation", "warn", CANDIDACY,
                    "JD states sponsorship isn't available — confirm you can work in the role's "
                    "country without it.", STRONG, cite)
    elif reqs.work_auth_required:
        add("Work authorisation", "warn", CANDIDACY,
            "JD requires proof of right to work for that location.", SOFT,
            reqs.cite("work_auth_required"))
    else:
        add("Work authorisation", "skipped", CANDIDACY, "JD says nothing about sponsorship.")

    # --- Notice period
    if profile.notice_period_days is None:
        add("Notice period", "skipped", CANDIDACY, "Set notice_period_days in profile.yaml.")
    elif profile.notice_period_days > 90:
        add("Notice period", "warn", CANDIDACY,
            f"{profile.notice_period_days}-day notice is long enough that some recruiters "
            "deprioritise you.", STRONG)
        expect("Notice period",
               f"Your {profile.notice_period_days}-day notice is long; urgent roles screen it out.",
               "Know in advance whether you can buy it out or negotiate it down, and say so "
               "unprompted — it removes the objection before it forms.")
    elif profile.notice_period_days > 60:
        add("Notice period", "warn", CANDIDACY,
            f"{profile.notice_period_days}-day notice — fine for most roles, a problem for "
            "urgent backfills.", SOFT)
    else:
        add("Notice period", "pass", CANDIDACY,
            f"{profile.notice_period_days}-day notice is unlikely to be an issue.", SOFT)

    # --- Title / seniority alignment
    if not reqs.seniority:
        add("Title alignment", "skipped", CANDIDACY, "No seniority marker in the JD title.")
    elif not profile.current_title:
        add("Title alignment", "skipped", CANDIDACY, "Set current_title in profile.yaml.")
    else:
        m = SENIORITY_WORD.search(profile.current_title)
        delta = seniority_rank(reqs.seniority) - seniority_rank(m.group(1) if m else "")
        cite = reqs.cite("seniority")
        if delta <= 0:
            add("Title alignment", "pass", CANDIDACY,
                f"Your title ({profile.current_title}) is at or above the JD's level "
                f"({reqs.seniority}).", STRONG, cite)
        elif delta == 1:
            add("Title alignment", "warn", CANDIDACY,
                f"JD is one level up ({reqs.seniority}) from {profile.current_title} — "
                "a normal stretch.", SOFT, cite)
        else:
            add("Title alignment", "warn", CANDIDACY,
                f"JD is {delta} levels above your current title "
                f"({profile.current_title} → {reqs.seniority}).", STRONG, cite)
            expect("Step up in level",
                   f"You'd be jumping {delta} levels ({profile.current_title} → {reqs.seniority}).",
                   "Expect 'have you done this scope before?' — have one example where you "
                   "operated at the higher level regardless of your title.")

    # --- Tenure & gaps (facts, not formatting)
    gaps, avg_tenure, dated_roles = analyze_timeline(resume_text)
    if gaps:
        add("Employment gaps", "warn", CANDIDACY,
            "Possible gap(s): " + "; ".join(gaps[:3]), STRONG)
        expect("Employment gap",
               f"Detected: {gaps[0]}.",
               "Recruiters always ask. Short, unapologetic, factual — what you did, what you "
               "learned, why you're back. Don't over-explain.")
    else:
        add("Employment gaps", "pass", CANDIDACY, "No gaps over 6 months in parsed dates.", SOFT)

    if avg_tenure is None:
        add("Tenure pattern", "skipped", CANDIDACY, "Not enough dated roles to assess.")
    elif avg_tenure < 18:
        add("Tenure pattern", "warn", CANDIDACY,
            f"Average parsed tenure ~{avg_tenure} months — short tenure draws scrutiny.", STRONG)
        expect("Short tenure",
               f"Your average role length parses to ~{avg_tenure} months.",
               "Expect 'why did you leave so quickly?' for each move. Frame as deliberate "
               "progression, not escape — and never criticise a past employer.")
    else:
        add("Tenure pattern", "pass", CANDIDACY, f"Average parsed tenure ~{avg_tenure} months.", SOFT)

    # ============================ RESUME: fixable by editing ===============

    coverage = top_third_coverage(resume_text, must_have_terms or [])
    if coverage is None:
        add("Top-third visibility", "skipped", RESUME, "Not enough resume text or JD terms.")
    elif coverage >= 60:
        add("Top-third visibility", "pass", RESUME,
            f"{coverage}% of the JD's top terms appear in the first third of your resume.", GATE)
    elif coverage >= 35:
        add("Top-third visibility", "warn", RESUME,
            f"Only {coverage}% of the JD's top terms are in the first third — recruiters often "
            "stop reading before page one ends.", GATE)
    else:
        add("Top-third visibility", "fail", RESUME,
            f"Just {coverage}% of the JD's top terms are in the first third. Move your most "
            "relevant evidence up.", GATE)

    # Can a recruiter compute your experience in seconds?
    if dated_roles == 0:
        add("Experience legibility", "fail", RESUME,
            "No parseable date ranges found on your roles. A recruiter can't verify your years "
            "at a glance — use a consistent 'Mon YYYY – Mon YYYY' format.", STRONG)
    elif dated_roles == 1:
        add("Experience legibility", "warn", RESUME,
            "Only one dated role parsed — make sure every position carries explicit dates.", STRONG)
    else:
        add("Experience legibility", "pass", RESUME,
            f"{dated_roles} roles carry parseable dates.", STRONG)

    # Is your current title visible where they actually look?
    if not profile.current_title:
        add("Title visibility", "skipped", RESUME, "Set current_title in profile.yaml.")
    elif _in_top_third(resume_text, profile.current_title):
        add("Title visibility", "pass", RESUME,
            f"'{profile.current_title}' appears in the top third of your resume.", STRONG)
    else:
        add("Title visibility", "warn", RESUME,
            f"'{profile.current_title}' doesn't appear in the top third — recruiters look for a "
            "title match immediately.", STRONG)

    # ============================ baseline screen items ====================
    # Asked on essentially every screening call regardless of your scores.
    expect("Current & expected salary",
           "Asked on almost every screen, and often on the application form itself.",
           "Have a researched range ready, not a single number, and know your floor. "
           "In India/GCC this is usually asked directly as current CTC and expected CTC.",
           source="standard")
    expect("Why are you leaving",
           "Standard opener; the answer sets the tone for the rest of the call.",
           "One sentence, forward-looking. What you're moving toward, not what you're escaping.",
           source="standard")
    expect("Why this role / company",
           "Filters out mass applicants immediately.",
           "Name something specific about this company or team that you couldn't say about "
           "three competitors.",
           source="standard")
    if profile.location or reqs.work_modes:
        expect("Location & relocation",
               "Logistics get confirmed early because they're cheap to disqualify on.",
               f"Be clear on whether you'd relocate"
               + (f" from {profile.location}" if profile.location else "")
               + " and from when.",
               source="standard")

    # ============================ scoring ==================================

    def pct(subset: list[Check]) -> float | None:
        scored = [c for c in subset if c.status in CREDIT]
        if not scored:
            return None
        earned = sum(CREDIT[c.status] * c.importance for c in scored)
        total = sum(c.importance for c in scored)
        return round(earned / total * 100, 1) if total else None

    resume_pct = pct([c for c in checks if c.category == RESUME])
    candidacy_pct = pct([c for c in checks if c.category == CANDIDACY])
    overall = pct(checks) or 0.0

    return RecruiterResult(
        criteria_met_pct=overall,
        resume_pct=resume_pct,
        candidacy_pct=candidacy_pct,
        checks=checks,
        blockers=blockers,
        expectations=expectations,
        top_third_coverage=coverage,
        employment_gaps=gaps,
        average_tenure_months=avg_tenure,
    )
