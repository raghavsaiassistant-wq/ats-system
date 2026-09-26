"""Deterministic extraction of a JD's hard screening filters.

This is kept rule-based (not LLM) on purpose: every requirement it pulls out
carries the exact JD line that produced it, so when the recruiter layer
penalises you, you can see the sentence responsible and judge for yourself
whether the extraction was right. An LLM would be more robust on weird
phrasing but you'd lose that audit trail.

Extracted filters:
  - years-of-experience floor (section-aware: a floor stated under
    Requirements/Qualifications gates; nice-to-have years no longer lower
    the bar for the whole role)
  - degree requirement, with softener detection ("or equivalent experience")
  - named certifications (broad vendor-neutral + vendor-specific patterns)
  - work mode (remote / hybrid / onsite)
  - sponsorship availability + work-authorisation demands
  - seniority marker from the title
  - stated salary range (with currency), when the JD includes one
  - stated location, and any country mentions (checked against your
    work_authorized_in list — the profile field is live now)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

YOE_PATTERNS = [
    # "3-5 years", "3 to 5 years"  -> floor is the first number
    (re.compile(r"(\d{1,2})\s*(?:-|–|—|to)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.I), "range"),
    # "5+ years", "5 + years"
    (re.compile(r"(\d{1,2})\s*\+\s*(?:years?|yrs?)", re.I), "min"),
    # "minimum of 3 years", "at least 3 years"
    (re.compile(r"(?:minimum|at least|min\.?)\s*(?:of\s*)?(\d{1,2})\s*(?:years?|yrs?)", re.I), "min"),
    # bare "3 years of experience"
    (re.compile(r"(\d{1,2})\s*(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+|professional\s+|hands-on\s+)?experience", re.I), "min"),
]

# Phrasing that means the years figure is a nice-to-have, not the role's gate
YEARS_SOFTENERS = re.compile(
    r"\bpreferred\b|\bnice to have\b|\ba plus\b|\bbonus\b|\bor equivalent\b",
    re.I,
)

DEGREE_PATTERNS = [
    (re.compile(r"\bph\.?d\b|\bdoctorate\b", re.I), "phd"),
    (re.compile(r"\bmba\b", re.I), "mba"),
    # "MS"/"M.Sc" means a master's degree, but "MS Excel", "MS Office", "MS SQL"
    # etc. name Microsoft products — don't read those as a degree requirement.
    (re.compile(
        r"\bmaster'?s?\b|\bm\.?s\.?c?\b(?!\w)"
        r"(?!\s*[-/]?\s*(?:excel|office|word|sql|access|power\s*point|ppt|outlook|teams|"
        r"project|visio|dynamics|azure|fabric|power\s*bi|sharepoint|365|dos|paint)\b)",
        re.I), "masters"),
    (re.compile(r"\bbachelor'?s?\b|\bb\.?tech\b|\bb\.?sc\b|\bb\.?a\b|\bb\.?b\.?a\b|\bundergraduate degree\b", re.I), "bachelors"),
    (re.compile(r"\bdiploma\b", re.I), "diploma"),
]

DEGREE_RANK = {"diploma": 2, "bachelors": 3, "masters": 4, "mba": 4, "phd": 5}

# Phrasing that means the degree is a nice-to-have, not a gate
DEGREE_SOFTENERS = re.compile(
    r"\bor equivalent\b|\bpreferred\b|\bnice to have\b|\ba plus\b|\bor relevant experience\b",
    re.I,
)

WORK_MODE_PATTERNS = [
    (re.compile(r"\b(?:fully\s+)?remote\b", re.I), "remote"),
    (re.compile(r"\bhybrid\b", re.I), "hybrid"),
    (re.compile(r"\bon-?site\b|\bin-?office\b|\bin person\b", re.I), "onsite"),
]

SPONSORSHIP_BLOCKED = re.compile(
    r"(?:sponsorship|visa)[^.\n]{0,60}(?:not (?:be )?available|cannot|can not|unable|do(?:es)? not (?:provide|offer|sponsor))"
    r"|(?:we do not|will not)[^.\n]{0,30}sponsor"
    r"|no (?:visa )?sponsorship"
    # "must currently be based in the UAE with a valid visa": the candidate
    # needs the right to work there already, the same gate as no-sponsorship.
    r"|must (?:currently |already )?(?:be )?(?:based|residing|located|living|reside) in[^.\n]{0,40}valid[^.\n]{0,20}visa"
    r"|(?:valid|own|existing) (?:\w+ )?(?:residence |residency |employment |work )?visa (?:is )?(?:required|mandatory|a must)"
    r"|candidates? (?:must|should) (?:hold|have|possess) (?:a )?valid (?:\w+ )?(?:residence |employment |work )?visa",
    re.I,
)
WORK_AUTH_REQUIRED = re.compile(
    r"must be (?:legally )?(?:authori[sz]ed|eligible) to work|right to work|work authori[sz]ation required",
    re.I,
)

# Broad certification detection: fixed acronyms, vendor exam codes, and the
# "X Certified Y" / "Google Z" families. Every hit still cites its line, so
# a false positive is visible and judgeable. The tail uses (?![A-Za-z0-9])
# rather than \b because \b doesn't exist between '+' and a space — that
# silently dropped every CompTIA-style cert ending in '+'.
CERT_PATTERN = re.compile(
    r"\b("
    r"PMP|PRINCE2|CPA|CFA|CMA|CIMA|ACCA|FRM|CISA|CISM|CISSP|CIPP|CCNA|CCNP|CCIE|"
    r"CSM|CSPO|PSM\s?I{1,3}|SAFe(?:\s+\w+)?|ITIL(?:\s+\w+)?|CBAP|"
    r"PHR|SPHR|SHRM-?(?:CP|SCP)|CIPD|CAMS|"
    r"(?:PL|DP|AZ|AI|MB|MS)-\d{3}|"
    r"AWS Certified[\w\s]{0,30}|"
    r"Google (?:Professional|Associate|Data Analytics|Cloud|Project Management|UX|IT Support|Digital Marketing)[\w\s]{0,25}|"
    r"Microsoft Certified[\w\s]{0,40}|"
    r"CompTIA\s+(?:A\+|Network\+|Security\+|Cloud\+|CySA\+|PenTest\+)|"
    r"(?:Lean\s+)?Six Sigma(?:\s+(?:Green|Black|Yellow|White)\s+Belt)?|"
    r"Tableau(?:\s+Desktop)?\s+(?:Specialist|Certified[\w\s]{0,20})|"
    r"Qlik[\w\s]{0,15}Certified[\w\s]{0,20}|"
    r"Databricks Certified[\w\s]{0,30}|"
    r"Snowflake[\w\s]{0,15}Certified[\w\s]{0,30}|"
    r"Oracle Certified[\w\s]{0,30}|"
    r"SAP Certified[\w\s]{0,40}|"
    r"Salesforce Certified[\w\s]{0,40}|"
    r"Teradata Certified[\w\s]{0,30}|"
    r"Alteryx Certified[\w\s]{0,30}|"
    r"Certified \w[\w\s]{0,30}"
    r")(?![A-Za-z0-9])",
    re.I,
)

SENIORITY_PATTERN = re.compile(
    r"\b(intern|trainee|junior|jr\.?|associate|senior|sr\.?|staff|principal|lead|head of|director|vp|manager)\b",
    re.I,
)

SENIORITY_RANK = {
    "intern": 0, "trainee": 0,
    "junior": 1, "jr": 1, "jr.": 1,
    "associate": 2,
    "": 3,  # unmarked == mid
    "senior": 4, "sr": 4, "sr.": 4,
    "staff": 5, "lead": 5,
    "principal": 6, "manager": 6, "head of": 6,
    "director": 7, "vp": 8,
}

# ------------------------------------------------------------- salary

SALARY_CONTEXT = re.compile(
    r"\bsalary\b|\bcompensation\b|\bctc\b|\bpay(?:\s+range|\s+scale)?\b|\bpaying\b|"
    r"\bper annum\b|\bannually\b|\blpa\b|\blakhs?\b|\bper month\b|\bmonthly\b|\bbudget(?:ed)?\b|"
    r"[₹$£€]",
    re.I,
)
SALARY_AMOUNT_RE = re.compile(
    r"(?P<cur>[₹$£€])?\s*(?P<amt>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>lpa|lakhs?|lacs?|k|m|million)?",
    re.I,
)
YEARS_AFTER_RE = re.compile(r"^\s*\+?\s*(?:years?|yrs?)\b", re.I)
CURRENCY_BY_SYMBOL = {"₹": "INR", "$": "USD", "£": "GBP", "€": "EUR"}

# ------------------------------------------------------------- location

LOCATION_RE = re.compile(
    r"\b(?:location|based in|located in|office in|situated in|work from)\s*[:\-]?\s*"
    r"(?P<loc>[A-Za-z][A-Za-z .,'’&]{2,50})",
    re.I,
)

# Country mentions (canonical name -> matchers). Used to check against the
# profile's work_authorized_in list and to ground the sponsorship check in
# an actual place instead of free text.
COUNTRY_ALIASES: dict[str, list[str]] = {
    "india": ["india", "indian", "bengaluru", "bangalore", "mumbai", "delhi", "hyderabad",
              "chennai", "pune", "vadodara", "ahmedabad", "kolkata", "noida", "gurgaon", "gurugram"],
    "united states": ["united states", "usa", "u.s.", "america", "new york", "san francisco",
                      "seattle", "austin", "boston", "chicago", "washington dc", "atlanta", "dallas"],
    "united kingdom": ["united kingdom", "u.k.", "britain", "england", "london", "manchester",
                       "scotland", "wales", "edinburgh", "birmingham"],
    "united arab emirates": ["united arab emirates", "uae", "u.a.e", "dubai", "abu dhabi", "sharjah"],
    "canada": ["canada", "toronto", "vancouver", "montreal", "ottawa", "calgary"],
    "australia": ["australia", "sydney", "melbourne", "brisbane", "perth"],
    "singapore": ["singapore"],
    "germany": ["germany", "berlin", "munich", "hamburg", "frankfurt"],
    "netherlands": ["netherlands", "amsterdam", "hague"],
    "ireland": ["ireland", "dublin"],
    "new zealand": ["new zealand", "auckland"],
    "south africa": ["south africa", "johannesburg", "cape town"],
    "japan": ["japan", "tokyo", "osaka"],
    "hong kong": ["hong kong"],
    "saudi arabia": ["saudi arabia", "saudi", "riyadh", "jeddah"],
    "qatar": ["qatar", "doha"],
    "kuwait": ["kuwait"],
    "bahrain": ["bahrain", "manama"],
    "oman": ["oman", "muscat"],
    "france": ["france", "paris"],
    "spain": ["spain", "madrid", "barcelona"],
    "switzerland": ["switzerland", "zurich", "geneva"],
}

# NOTE: "us" alone is deliberately NOT a United States matcher — it matches
# too much ordinary English ("with us", "about us"). "usa" / "u.s." / city
# names carry the signal instead.
_C_MATCHERS = [
    (country, re.compile(rf"(?<![a-z])(?:{'|'.join(re.escape(m) for m in matchers)})(?![a-z])", re.I))
    for country, matchers in COUNTRY_ALIASES.items()
]


def normalize_country(name: str) -> str:
    """Canonical country name for a user-supplied string ('UAE', 'us', 'Dubai')."""
    s = (name or "").strip().lower()
    if not s:
        return ""
    for country, pattern in _C_MATCHERS:
        if pattern.fullmatch(s):
            return country
    for country, pattern in _C_MATCHERS:
        if pattern.search(s):
            return country
    return s


def detect_countries(text: str) -> list[str]:
    """Canonical country names mentioned anywhere in the text (deduped)."""
    found: list[str] = []
    for country, pattern in _C_MATCHERS:
        if pattern.search(text):
            found.append(country)
    return found


def _salary_from_line(line: str):
    """(min, max, currency) parsed from one salary-context line, or None.

    Units can attach per-number ("$120k-140k") or once after a range
    ("₹6-10 LPA" — the LPA applies to BOTH ends), so unit handling is
    two-level: per-number units apply to their own amount; when a range has
    no per-number unit, a line-level unit (lpa/lakh/million) applies to
    both ends.
    """
    matches = []
    for m in SALARY_AMOUNT_RE.finditer(line):
        tail = line[m.end(): m.end() + 10]
        if YEARS_AFTER_RE.match(tail):
            continue  # "3+ years" on a salary line is tenure, not money
        matches.append((float(m.group("amt").replace(",", "")),
                        (m.group("unit") or "").lower(), m.group("cur")))
    if not matches:
        return None

    currency = next((CURRENCY_BY_SYMBOL[cur] for _, _, cur in matches if cur), None)

    def _scale(amt: float, unit: str) -> float:
        if unit in ("lpa", "lakh", "lakhs", "lac", "lacs"):
            return amt * 100_000.0
        if unit in ("m", "million"):
            return amt * 1_000_000.0
        if unit == "k":
            return amt * 1_000.0
        return amt

    # line-level unit: applies to a range whose numbers carried none
    line_unit = ""
    lu = re.search(r"\b(lpa|lakhs?|lacs?|million)\b", line, re.I)
    if lu:
        line_unit = lu.group(1).lower()

    low = re.search(r"\bup to\b", line, re.I)
    high = re.search(r"\b(?:from|starting(?: at)?|minimum(?: of)?)\b", line, re.I)

    if len(matches) >= 2:
        (a, ua, _), (b, ub, _) = matches[0], matches[1]
        # A unit written on EITHER end of a range describes the whole range
        # ("₹6-10 LPA" — the LPA scales both numbers), so propagate it across
        # the pair; the line-level unit fills any remaining gap ("LPA: 6-10").
        if not ua and ub:
            ua = ub
        if not ub and ua:
            ub = ua
        if not ua:
            ua = line_unit
        if not ub:
            ub = line_unit
        a, b = _scale(a, ua), _scale(b, ub)
        return min(a, b), max(a, b), currency

    amt, unit, _ = matches[0]
    amt = _scale(amt, unit or line_unit)
    if low:
        return None, amt, currency
    if high:
        return amt, None, currency
    # "₹50,000+ per month" style: single figure with a plus -> the floor
    if re.search(r"\d[\d,]*(?:\.\d+)?\s*\+", line):
        return amt, None, currency
    return None, None, currency  # figure present but role unclear — don't guess


@dataclass
class Requirement:
    kind: str
    value: object
    source_line: str
    mandatory: bool = True


@dataclass
class JDRequirements:
    min_years: float | None = None
    min_degree: str | None = None
    degree_mandatory: bool = True
    certifications: list[str] = field(default_factory=list)
    work_modes: list[str] = field(default_factory=list)
    sponsorship_unavailable: bool = False
    work_auth_required: bool = False
    seniority: str = ""
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    jd_location: str = ""
    countries: list[str] = field(default_factory=list)
    evidence: list[Requirement] = field(default_factory=list)

    def cite(self, kind: str) -> str:
        for req in self.evidence:
            if req.kind == kind:
                return req.source_line
        return ""


def _lines(jd_text: str) -> list[str]:
    return [ln.strip() for ln in jd_text.splitlines() if ln.strip()]


def _classified(jd_text: str):
    """(line, section) pairs, tracking Requirements/Qualifications headers so
    the years floor can tell a stated gate from a nice-to-have."""
    from .keywords import _classify_line_section

    current = "default"
    out = []
    for line in jd_text.splitlines():
        if not line.strip():
            continue
        current = _classify_line_section(line, current)
        out.append((line.strip(), current))
    return out


def extract(jd_text: str) -> JDRequirements:
    reqs = JDRequirements()
    lines = _lines(jd_text)

    # --- years of experience, SECTION-AWARE. A floor stated under
    # Requirements/Qualifications is the role's gate and wins. Only if no
    # hard-section floor exists do we fall back to the lowest floor anywhere
    # — so "2+ years with Spark preferred" in a nice-to-have no longer
    # lowers the bar for the whole role, which used to under-gate it.
    best_hard: float | None = None
    best_hard_line = ""
    best_any: float | None = None
    best_any_line = ""
    for line, section in _classified(jd_text):
        for pattern, kind in YOE_PATTERNS:
            m = pattern.search(line)
            if not m:
                continue
            floor = float(m.group(1))
            if YEARS_SOFTENERS.search(line):
                continue  # explicitly a nice-to-have figure
            if best_any is None or floor < best_any:
                best_any, best_any_line = floor, line
            if section == "hard" and (best_hard is None or floor < best_hard):
                best_hard, best_hard_line = floor, line
    floor, floor_line = (
        (best_hard, best_hard_line) if best_hard is not None else (best_any, best_any_line)
    )
    if floor is not None:
        reqs.min_years = floor
        reqs.evidence.append(Requirement("min_years", floor, floor_line))

    # --- degree: highest degree named wins as the stated bar, but if the
    # line softens it ("or equivalent experience"), mark it non-mandatory.
    found_degree, degree_line = None, ""
    for line in lines:
        for pattern, level in DEGREE_PATTERNS:
            if pattern.search(line):
                rank = DEGREE_RANK.get(level, 0)
                if found_degree is None or rank > DEGREE_RANK.get(found_degree, 0):
                    found_degree, degree_line = level, line
    if found_degree:
        reqs.min_degree = found_degree
        reqs.degree_mandatory = not bool(DEGREE_SOFTENERS.search(degree_line))
        reqs.evidence.append(
            Requirement("min_degree", found_degree, degree_line, reqs.degree_mandatory)
        )

    # --- certifications
    seen = set()
    for line in lines:
        for m in CERT_PATTERN.finditer(line):
            cert = m.group(1).strip()
            key = cert.lower()
            if key in seen:
                continue
            seen.add(key)
            reqs.certifications.append(cert)
            mandatory = not bool(DEGREE_SOFTENERS.search(line))
            reqs.evidence.append(Requirement("certification", cert, line, mandatory))

    # --- work mode
    for line in lines:
        for pattern, mode in WORK_MODE_PATTERNS:
            if pattern.search(line) and mode not in reqs.work_modes:
                reqs.work_modes.append(mode)
                reqs.evidence.append(Requirement("work_mode", mode, line))

    # --- sponsorship / work authorisation
    for line in lines:
        if not reqs.sponsorship_unavailable and SPONSORSHIP_BLOCKED.search(line):
            reqs.sponsorship_unavailable = True
            reqs.evidence.append(Requirement("sponsorship_unavailable", True, line))
        if not reqs.work_auth_required and WORK_AUTH_REQUIRED.search(line):
            reqs.work_auth_required = True
            reqs.evidence.append(Requirement("work_auth_required", True, line))

    # --- salary range, when stated. First salary-context line with a
    # parseable range wins; JDs rarely state two different ranges and
    # merging across lines could invent one that was never written.
    for line in lines:
        if not SALARY_CONTEXT.search(line):
            continue
        parsed = _salary_from_line(line)
        if parsed is None:
            continue
        smin, smax, cur = parsed
        if smin is None and smax is None:
            continue
        reqs.salary_min, reqs.salary_max, reqs.salary_currency = smin, smax, cur
        reqs.evidence.append(Requirement("salary", (smin, smax, cur), line, mandatory=False))
        break

    # --- stated location + country mentions
    for line in lines:
        m = LOCATION_RE.search(line)
        if m and not reqs.jd_location:
            reqs.jd_location = m.group("loc").strip(" .,-")
            reqs.evidence.append(Requirement("location", reqs.jd_location, line, mandatory=False))
    reqs.countries = detect_countries(jd_text)
    if reqs.countries:
        reqs.evidence.append(
            Requirement("countries", reqs.countries,
                         reqs.jd_location or "country mentioned in the JD", mandatory=False)
        )

    # --- seniority, read from the title line (usually the first line, or a
    # line starting with "Job Title"/"Role")
    title_line = ""
    for line in lines[:6]:
        if re.match(r"^(job\s*title|role|position)\s*[:\-]", line, re.I) or line is lines[0]:
            title_line = line
            break
    if title_line:
        m = SENIORITY_PATTERN.search(title_line)
        if m:
            reqs.seniority = m.group(1).lower().rstrip(".")
            reqs.evidence.append(Requirement("seniority", reqs.seniority, title_line))

    return reqs


def seniority_rank(label: str) -> int:
    return SENIORITY_RANK.get((label or "").lower().strip().rstrip("."), 3)