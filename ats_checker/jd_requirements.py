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
    # "Scrum Master", "master data" and "MS Excel/Office/SQL" are not degrees —
    # matching them used to turn a plain analyst JD into a mandatory masters gate.
    (re.compile(
        r"\bmaster'?s\b(?!\s+data)|\bmaster\s+(?:of|degree)\b|\bm\.?sc\b|\bm\.s\.?(?!\w)"
        r"|(?<!\d)(?<!\d )\bms\b(?!\s*(?:excel|office|word|sql|access|teams|project|visio|power|azure|"
        r"dynamics|outlook|powerpoint|365|fabric|sharepoint|-|\d))",
        re.I), "masters"),
    # bare "BA" is Business Analyst far more often than a degree in a JD
    (re.compile(r"\bbachelor'?s?\b|\bb\.?tech\b|\bb\.?sc\b|\bb\.a\.?(?!\w)|\bba\s+(?:in|degree|\(?hons)\b"
                r"|\bb\.?b\.?a\b|\bundergraduate degree\b"
                # BS / B.S. / B.E. / BSN, and India's "any graduate"
                r"|\bb\.s\.?(?!\w)|\bbs\s+(?:in|degree)\b|\bb\.e\.?(?![a-z])|\bbsn\b"
                r"|(?<!post)\bgraduate\b(?!\s+(?:school|program|programme|trainee|engineer))",
                re.I), "bachelors"),
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
    (re.compile(r"\bon-?site\b|\bin[\s-]?office\b|\bin person\b|\bwork from office\b|\bwfo\b"
                r"|\boffice[\s-]based\b", re.I), "onsite"),
]

SPONSORSHIP_BLOCKED = re.compile(
    r"(?:sponsorship|visa)[^.\n]{0,60}(?:not (?:be )?available|cannot|can not|unable|do(?:es)? not (?:provide|offer|sponsor))"
    r"|(?:we do not|will not)[^.\n]{0,30}sponsor"
    r"|no (?:visa )?sponsorship"
    r"|(?:unable to|not able to|cannot|can not|do not|does not|will not|won't)\s+(?:provide\s+)?sponsor",
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
    r"CSM|CSPO|PSM\s?I{1,3}|(?-i:SAFe)(?:\s+\w+)?|ITIL(?:\s+\w+)?|CBAP|"
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
    # capitalised words only, max 4: "Certified Scrum Master preferred for
    # this" used to be captured whole and then never matched the resume
    r"(?-i:Certified(?:\s+[A-Z][\w+]*){1,4})"
    r")(?![A-Za-z0-9])",
    re.I,
)

SENIORITY_PATTERN = re.compile(
    r"\b(intern|trainee|junior|jr\.?|associate|senior|sr\.?|"
    r"staff(?=\s+(?:\w+\s+)?(?:engineer|scientist|developer|architect|designer))|"
    r"principal|lead(?!\s+generation)|head of|director|vp|manager)\b",
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
    r"[₹$£€]|\b(?:INR|USD|GBP|EUR|AED|SAR|QAR|CAD|SGD|AUD)\b",
    re.I,
)
SALARY_AMOUNT_RE = re.compile(
    # unit must end at a word boundary: "10 members" is not 10 million
    r"(?P<cur>[₹$£€])?\s*(?P<amt>\d[\d,]*(?:\.\d+)?)\s*(?:(?P<unit>lpa|lakhs?|lacs?|million|k|m)(?![a-z]))?",
    re.I,
)
YEARS_AFTER_RE = re.compile(r"^\s*\+?\s*(?:years?|yrs?)\b", re.I)
CURRENCY_BY_SYMBOL = {"₹": "INR", "$": "USD", "£": "GBP", "€": "EUR"}
CURRENCY_CODES = {
    "INR": "INR", "RS": "INR", "USD": "USD", "US$": "USD", "GBP": "GBP", "EUR": "EUR",
    "AED": "AED", "SAR": "SAR", "QAR": "QAR", "KWD": "KWD", "BHD": "BHD", "OMR": "OMR",
    "CAD": "CAD", "C$": "CAD", "SGD": "SGD", "S$": "SGD", "AUD": "AUD", "A$": "AUD",
    "NZD": "NZD", "CHF": "CHF", "JPY": "JPY", "HKD": "HKD", "ZAR": "ZAR",
}
CURRENCY_CODE_RE = re.compile(
    r"(?<![A-Za-z])(US\$|C\$|S\$|A\$|Rs\.?|INR|USD|GBP|EUR|AED|SAR|QAR|KWD|BHD|OMR|CAD|SGD|AUD"
    r"|NZD|CHF|JPY|HKD|ZAR)(?![A-Za-z])",
)

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
                      "seattle", "austin", "boston", "chicago", "washington dc", "washington, dc", "d.c.", "atlanta", "dallas"],
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

    # An explicit code beats a bare symbol: "$120,000 CAD" is Canadian and
    # "S$8,000" Singaporean; lakh/LPA figures are rupees even without "₹".
    code = CURRENCY_CODE_RE.search(line)
    if code:
        currency = CURRENCY_CODES[code.group(1).upper().rstrip(".")]
    else:
        currency = next((CURRENCY_BY_SYMBOL[cur] for _, _, cur in matches if cur), None)
    if currency is None and any(u in ("lpa", "lakh", "lakhs", "lac", "lacs") for _, u, _c in matches):
        currency = "INR"
    if currency is None and re.search(r"\b(?:lpa|lakhs?|lacs?)\b", line, re.I):
        currency = "INR"

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
    jd_title: str = ""          # role title without seniority/extras, lowercase
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

    # --- degree: the FLOOR, clause by clause. Alternatives in one clause
    # ("MS or PhD", "Bachelor's or Master's") set the bar at the lowest
    # named; a softened clause ("Master's preferred", "or equivalent
    # experience") never outranks a mandatory one. Highest-named-wins used
    # to turn "Bachelor's required; Master's preferred" into a masters gate.
    mandatory, softened = [], []   # (rank, level, line)
    for line in lines:
        for clause in re.split(r";|\.\s", line):
            levels = [lvl for pat, lvl in DEGREE_PATTERNS if pat.search(clause)]
            if not levels:
                continue
            level = min(levels, key=lambda lv: DEGREE_RANK.get(lv, 0))
            soft = bool(DEGREE_SOFTENERS.search(clause))
            (softened if soft else mandatory).append((DEGREE_RANK.get(level, 0), level, line))
    pool = mandatory or softened
    if pool:
        _rank, level, degree_line = min(pool, key=lambda t: t[0])
        reqs.min_degree = level
        reqs.degree_mandatory = bool(mandatory)
        reqs.evidence.append(
            Requirement("min_degree", level, degree_line, reqs.degree_mandatory)
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

    # --- work mode. "Hybrid - 3 days in office" describes the hybrid
    # pattern, not a second onsite mode, so office wording on a hybrid line
    # doesn't add onsite.
    for line in lines:
        hybrid_line = bool(WORK_MODE_PATTERNS[1][0].search(line))
        for pattern, mode in WORK_MODE_PATTERNS:
            if mode == "onsite" and hybrid_line:
                continue
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
            # the capture runs to 50 chars; keep only the first sentence of it
            loc = re.split(r"\.\s", m.group("loc"))[0]
            reqs.jd_location = loc.strip(" .,-")
            reqs.evidence.append(Requirement("location", reqs.jd_location, line, mandatory=False))
    reqs.countries = detect_countries(jd_text)
    if reqs.countries:
        reqs.evidence.append(
            Requirement("countries", reqs.countries,
                         reqs.jd_location or "country mentioned in the JD", mandatory=False)
        )

    # --- seniority, read from the title line (usually the first line, or a
    # line starting with "Job Title"/"Role")
    # (the old loop also accepted lines[0] on its first iteration, so an
    # explicit "Job Title:" on any later line was never reached)
    title_line = next(
        (ln for ln in lines[:6] if re.match(r"^(job\s*title|role|position)\s*[:\-]", ln, re.I)),
        lines[0] if lines else "",
    )
    reqs.jd_title = clean_title(title_line)
    if title_line:
        m = SENIORITY_PATTERN.search(title_line)
        if m:
            reqs.seniority = m.group(1).lower().rstrip(".")
            reqs.evidence.append(Requirement("seniority", reqs.seniority, title_line))

    return reqs


# A title line must name a role; otherwise line 1 is usually "About us" or a
# company tagline, and searching for it would be meaningless.
ROLE_NOUNS = re.compile(
    r"\b(analyst|engineer|developer|manager|consultant|specialist|scientist|designer|"
    r"architect|administrator|coordinator|executive|officer|accountant|representative|"
    r"director|intern|lead|associate|strategist|programmer|technician|tester|writer|"
    r"recruiter|advisor|auditor|controller|owner|partner|head|generalist|teacher|nurse|"
    r"planner|agent|assistant|clerk|therapist|pharmacist|physician|editor|instructor|trainer|"
    r"lecturer|professor|supervisor|operator|technologist|coach)s?\b",
    re.I,
)
_TITLE_SENIORITY = re.compile(
    r"\b(senior|sr\.?|junior|jr\.?|principal|intern(?=\s*[-–—:]|$)|trainee|"
    # "Staff" is a level only for engineering-ladder titles ("Staff Engineer"),
    # not for "Staff Accountant"; "Lead" only as a prefix ("Lead Data Analyst"),
    # never in "Team Lead" or "Lead Generation"
    r"staff(?=\s+(?:\w+\s+)?(?:engineer|scientist|developer|architect|designer))|"
    r"^\s*lead(?=\s+(?!generation)\w)|"
    r"entry[\s-]level|mid[\s-]level|[iv]{1,3})\b\.?",
    re.I,
)


def clean_title(line: str) -> str:
    """The searchable role title from a JD title line: 'Job Title: Senior
    Business Intelligence Manager (Hybrid - Dubai)' -> 'business intelligence
    manager'. Seniority words, parentheticals and anything after a dash/pipe
    (location, team, req id) are dropped — recruiters search the role, and
    seniority is checked separately by the HR layer. '' when the line doesn't
    name a role."""
    t = re.sub(r"^\s*(job\s*title|role|position)\s*[:\-]\s*", "", line or "", flags=re.I)
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    t = re.split(r"\s[-–—|]\s|\s@\s|,", t)[0]
    t = _TITLE_SENIORITY.sub(" ", t)
    t = " ".join(re.sub(r"[^A-Za-z0-9&/+.# ]", " ", t).split()).lower()
    if not t or len(t.split()) > 6 or not ROLE_NOUNS.search(t):
        return ""
    return t


def seniority_rank(label: str) -> int:
    return SENIORITY_RANK.get((label or "").lower().strip().rstrip("."), 3)