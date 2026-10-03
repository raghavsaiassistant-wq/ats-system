"""LLM JD extraction that has to show its work.

The rule engine is precise but brittle: it only knows the skills in its
taxonomy and the phrasings in its regexes (held-out corpus: 68% of required
skills found). An LLM reads any phrasing, but it also invents things. So the
LLM is used here under one hard rule:

    Every item it returns must come with a VERBATIM quote from the JD, and
    the item must be visible inside that quote. Anything that fails the
    check is dropped — and listed, so the drop is auditable.

That turns "the model says the JD wants Kubernetes" into "the model points
at the line '- Kubernetes in production', and that line is in the JD".
A hallucinated requirement cannot cite a line that doesn't exist.

Checks per field (all on normalized text: case, whitespace, punctuation):
  quote       appears in the JD; at most MAX_QUOTE_CHARS long, so "quote the
              whole JD" can't vouch for everything
  skill       the term (or every word of it) appears in its quote
  years       the number appears in the quote next to "year"/"yr"; 0 < n <= 30
  degree      a known level, and the quote reads like a degree line
  work mode   remote | hybrid | onsite, with a matching word in the quote
  salary      min <= max, a 3-letter currency, digits in the quote
  country     named in the quote, or the quote names a city in it

The result is merged with the rule engine ("hybrid"): verified LLM values
win, the rules fill every gap, skills are the union, and each disagreement
is recorded. If the LLM is unreachable the rules answer alone and the
report says so — the LLM can only add coverage, never take the tool down.

Raw LLM answers are cached by (prompt version, model, JD text) under
$ATS_HOME/cache/jd_llm, so re-scoring the same JD costs nothing and returns
the same answer. Verification runs on every read, so tightening a check
applies to cached answers too.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from . import jd_requirements as jd_mod
from . import keywords as kw_mod
from . import learned as learned_mod
from . import llm_client
from .terms import alias_normalize, canonical, term_pattern

PROMPT_VERSION = "jd-extract-v2"
MAX_QUOTE_CHARS = 300
MAX_TERM_WORDS = 5
MAX_YEARS = 30
WORK_MODES = ("remote", "hybrid", "onsite")
# regions and work-mode words a model sometimes files under "countries"
NOT_COUNTRIES = {"remote", "global", "worldwide", "anywhere", "emea", "apac", "latam",
                 "europe", "asia", "africa", "north america", "south america", "middle east",
                 "gcc", "mena", "hybrid", "onsite", "international"}

SYSTEM_PROMPT = """You extract the hiring requirements from a job description.

Return ONLY a JSON object with exactly these keys:

{
  "title":    {"value": "<role title without seniority words, lowercase>", "quote": "<JD text>"},
  "seniority": {"value": "<intern|junior|associate|senior|staff|lead|principal|manager|head of|director|vp or empty>", "quote": "<JD text>"},
  "min_years": {"value": <number or null>, "mandatory": <true|false>, "quote": "<JD text>"},
  "degree":   {"value": "<diploma|bachelors|masters|mba|phd or null>", "mandatory": <true|false>, "quote": "<JD text>"},
  "certifications": [{"name": "<certification>", "quote": "<JD text>"}],
  "work_modes": [{"value": "<remote|hybrid|onsite>", "quote": "<JD text>"}],
  "sponsorship_unavailable": {"value": <true|false>, "quote": "<JD text or empty>"},
  "salary":   {"min": <number or null>, "max": <number or null>, "currency": "<ISO code>", "quote": "<JD text>"} or null,
  "countries": [{"value": "<country name, lowercase>", "quote": "<JD text>"}],
  "skills":   [{"term": "<skill, tool or method, as written in the JD>", "required": <true|false>, "quote": "<JD text>"}]
}

Rules:
- Every "quote" must be copied EXACTLY from the job description: one line or
  part of a line, under 200 characters. Never paraphrase a quote.
- Only include what the JD states. Use null / [] / "" when it says nothing.
- min_years is the lowest number of years the JD asks for ("3-5 years" -> 3).
  mandatory=false when it says preferred / nice to have / or equivalent.
- degree is the LOWEST level that satisfies the JD. mandatory=false when it
  says preferred / or equivalent experience.
- salary is the amount AS STATED, for the period the JD states (per month
  stays per month, per hour stays per hour), written out as plain numbers:
  "18-24 LPA" -> min 1800000, max 2400000, currency INR; "$90k-$110k" ->
  90000, 110000, USD; "AED 18,000 - 22,000 per month" -> 18000, 22000, AED.
- sponsorship_unavailable=true only when the JD says it will not or cannot
  sponsor a visa. A citizenship or security-clearance requirement alone is
  not a sponsorship statement.
- skills: concrete skills, tools, platforms, methods and domain knowledge a
  recruiter could search for (e.g. "sql", "power bi", "gap analysis",
  "gaap"). NOT soft traits ("team player"), NOT duties, NOT company facts.
  required=true when it is a requirement / must-have / qualification;
  false when preferred, nice to have, a plus, or only mentioned in the
  company or responsibilities description.
"""


# ------------------------------------------------------------- normalization

_PUNCT_RE = re.compile(r"[‘’“”\"'`–—−]")
_NON_TEXT_RE = re.compile(r"[^\w+#&$₹£€%]+")


def norm_text(text: str) -> str:
    """Comparison form: lowercase, quotes/dashes folded, every run of other
    punctuation/whitespace collapsed to one space. LLMs routinely drop a
    bullet marker or re-space a line; they shouldn't fail verification for it,
    but they still have to reproduce the words in order."""
    t = _PUNCT_RE.sub(" ", (text or "").lower())
    return " ".join(_NON_TEXT_RE.sub(" ", t).split())


def _words(text: str) -> set[str]:
    return set(norm_text(alias_normalize((text or "").lower())).split())


# ------------------------------------------------------------- verification

@dataclass
class Rejected:
    field: str
    value: object
    reason: str


@dataclass
class Verified:
    """LLM output that survived verification. None/[] = not stated (or dropped)."""
    title: str | None = None
    seniority: str | None = None
    min_years: float | None = None
    years_mandatory: bool = True
    min_degree: str | None = None
    degree_mandatory: bool = True
    certifications: list[str] = field(default_factory=list)
    work_modes: list[str] = field(default_factory=list)
    sponsorship_unavailable: bool | None = None
    salary: list | None = None              # [min, max, currency]
    countries: list[str] = field(default_factory=list)
    skills: dict[str, tuple[bool, str]] = field(default_factory=dict)  # term -> (required, quote)
    quotes: dict[str, str] = field(default_factory=dict)              # field -> supporting quote
    rejected: list[Rejected] = field(default_factory=list)


_DEGREE_LINE_RE = re.compile(
    r"degree|bachelor|master|\bmba\b|ph\s?d|doctorate|diploma|graduat|b\s?tech|b\s?sc|"
    r"\bb\s?e\b|\bb\s?s\b|\bm\s?s\b|\bbsn\b|m\s?sc|b\s?b\s?a|\bb\s?a\b|undergrad|post ?grad",
    re.I)
_WORK_MODE_WORDS = {
    "remote": re.compile(r"remote|work from home|wfh|anywhere", re.I),
    "hybrid": re.compile(r"hybrid|days? (?:a week )?in (?:the )?office|days? on ?site", re.I),
    "onsite": re.compile(r"on ?site|in office|in person|office based|work from office|\bwfo\b|"
                         r"on the ground|at our .{0,30}(?:office|site|plant|facility)", re.I),
}
_SPONSOR_RE = re.compile(r"sponsor|visa|authori[sz]|right to work|work permit|citizen", re.I)
_YEARS_WORD_RE = re.compile(r"\b(?:years?|yrs?)\b", re.I)
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


def _quote_ok(quote, jd_norm: str) -> tuple[bool, str]:
    if not isinstance(quote, str) or not quote.strip():
        return False, "no quote"
    q = norm_text(quote)
    if len(q) < 2:
        return False, "quote too short"
    if len(q) > MAX_QUOTE_CHARS:
        return False, f"quote longer than {MAX_QUOTE_CHARS} chars"
    if q not in jd_norm:
        return False, "quote not found in the JD"
    return True, ""


def _term_in_quote(term: str, quote: str) -> bool:
    """The term must be visible in its quote: as a phrase (alias- and
    plural-tolerant), or failing that every word of it present."""
    hay = norm_text(alias_normalize(quote.lower()))
    t = norm_text(canonical(term))
    if not t:
        return False
    if term_pattern(t).search(hay) or t in hay:
        return True
    words = [w for w in t.split() if len(w) > 2]
    qwords = set(hay.split())
    return bool(words) and all(w in qwords or w + "s" in qwords or w.rstrip("s") in qwords
                               for w in words)


def _number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _as_dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _as_list(v) -> list:
    return v if isinstance(v, list) else []


def verify(raw: dict, jd_text: str) -> Verified:
    """Keep only the parts of an LLM answer the JD itself supports."""
    jd_norm = norm_text(jd_text)
    out = Verified()
    rej = out.rejected

    def grounded(name: str, value, quote) -> bool:
        ok, why = _quote_ok(quote, jd_norm)
        if not ok:
            rej.append(Rejected(name, value, why))
        return ok

    # ---- title
    t = _as_dict(raw.get("title"))
    title = str(t.get("value") or "").strip().lower()
    if title and grounded("title", title, t.get("quote")):
        # both sides through the same alias pass: "Data Engineer" in the quote
        # reads as "data engineering", so the title must too
        quote_words = _words(t["quote"])
        if all(w in quote_words for w in _words(title)):
            out.title = (jd_mod.clean_title(title)
                         or " ".join(jd_mod._TITLE_SENIORITY.sub(" ", norm_text(title)).split()))
            out.quotes["title"] = t["quote"]
        else:
            rej.append(Rejected("title", title, "title words not in quote"))

    # ---- seniority
    s = _as_dict(raw.get("seniority"))
    sen = str(s.get("value") or "").strip().lower()
    if sen:
        sen = {"sr": "senior", "sr.": "senior", "jr": "junior", "jr.": "junior"}.get(sen, sen)
        if sen not in jd_mod.SENIORITY_RANK:
            rej.append(Rejected("seniority", sen, "unknown seniority level"))
        elif grounded("seniority", sen, s.get("quote")):
            m = jd_mod.SENIORITY_PATTERN.findall(s["quote"])
            found = {{"sr": "senior", "sr.": "senior", "jr": "junior", "jr.": "junior"}
                     .get(x.lower(), x.lower()) for x in m}
            if sen in found:
                out.seniority = sen
                out.quotes["seniority"] = s["quote"]
            else:
                rej.append(Rejected("seniority", sen, "seniority word not in quote"))

    # ---- years
    y = _as_dict(raw.get("min_years"))
    years = _number(y.get("value"))
    if years is not None:
        if not 0 < years <= MAX_YEARS:
            rej.append(Rejected("min_years", years, f"outside 0-{MAX_YEARS}"))
        elif grounded("min_years", years, y.get("quote")):
            num = f"{years:g}"
            q = y["quote"]
            if re.search(rf"(?<!\d){re.escape(num)}(?!\d)", q) and _YEARS_WORD_RE.search(q):
                out.min_years = years
                out.years_mandatory = y.get("mandatory") is not False
                out.quotes["min_years"] = q
            else:
                rej.append(Rejected("min_years", years, "number + 'years' not in quote"))

    # ---- degree
    d = _as_dict(raw.get("degree"))
    deg = str(d.get("value") or "").strip().lower() or None
    if deg:
        deg = {"bachelor": "bachelors", "bachelor's": "bachelors", "master": "masters",
               "master's": "masters", "doctorate": "phd", "ph.d": "phd"}.get(deg, deg)
        if deg not in jd_mod.DEGREE_RANK:
            rej.append(Rejected("min_degree", deg, "unknown degree level"))
        elif grounded("min_degree", deg, d.get("quote")):
            if _DEGREE_LINE_RE.search(norm_text(d["quote"])):
                out.min_degree = deg
                out.degree_mandatory = d.get("mandatory") is not False
                out.quotes["min_degree"] = d["quote"]
            else:
                rej.append(Rejected("min_degree", deg, "quote is not a degree line"))

    # ---- certifications
    for c in _as_list(raw.get("certifications")):
        c = _as_dict(c)
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        if grounded("certifications", name, c.get("quote")):
            if _term_in_quote(name, c["quote"]):
                if name.lower() not in (x.lower() for x in out.certifications):
                    out.certifications.append(name)
            else:
                rej.append(Rejected("certifications", name, "name not in quote"))

    # ---- work modes
    for w in _as_list(raw.get("work_modes")):
        w = _as_dict(w)
        mode = str(w.get("value") or "").strip().lower().replace("on-site", "onsite")
        mode = {"on site": "onsite", "in office": "onsite", "office": "onsite"}.get(mode, mode)
        if mode not in WORK_MODES:
            if mode:
                rej.append(Rejected("work_modes", mode, "unknown work mode"))
            continue
        if grounded("work_modes", mode, w.get("quote")):
            if _WORK_MODE_WORDS[mode].search(norm_text(w["quote"])):
                if mode not in out.work_modes:
                    out.work_modes.append(mode)
            else:
                rej.append(Rejected("work_modes", mode, "mode word not in quote"))

    # ---- sponsorship (only "unavailable" needs proof; silence means False)
    sp = _as_dict(raw.get("sponsorship_unavailable"))
    if sp.get("value") is True:
        if grounded("sponsorship_unavailable", True, sp.get("quote")):
            if _SPONSOR_RE.search(sp["quote"]):
                out.sponsorship_unavailable = True
                out.quotes["sponsorship_unavailable"] = sp["quote"]
            else:
                rej.append(Rejected("sponsorship_unavailable", True, "no sponsorship wording in quote"))
    elif sp.get("value") is False:
        out.sponsorship_unavailable = False

    # ---- salary
    sal = raw.get("salary")
    if isinstance(sal, dict) and (sal.get("min") is not None or sal.get("max") is not None):
        lo, hi = _number(sal.get("min")), _number(sal.get("max"))
        cur = str(sal.get("currency") or "").strip().upper()
        if (lo is not None and lo <= 0) or (hi is not None and hi <= 0) \
                or (lo is not None and hi is not None and lo > hi):
            rej.append(Rejected("salary", [lo, hi, cur], "amounts out of order or not positive"))
        elif not _CURRENCY_RE.match(cur):
            rej.append(Rejected("salary", [lo, hi, cur], "currency is not a 3-letter code"))
        elif grounded("salary", [lo, hi, cur], sal.get("quote")):
            if re.search(r"\d", sal["quote"]):
                out.salary = [lo, hi, cur]
                out.quotes["salary"] = sal["quote"]
            else:
                rej.append(Rejected("salary", [lo, hi, cur], "no amount in quote"))

    # ---- countries
    for c in _as_list(raw.get("countries")):
        c = _as_dict(c)
        name = jd_mod.normalize_country(str(c.get("value") or "").strip().lower())
        if not name:
            continue
        if grounded("countries", name, c.get("quote")):
            q = c["quote"]
            if name in NOT_COUNTRIES or not re.fullmatch(r"[a-z]+(?: [a-z]+){0,2}", name):
                rej.append(Rejected("countries", name, "not a country"))
            elif name in norm_text(q) or name in jd_mod.detect_countries(q):
                if name not in out.countries:
                    out.countries.append(name)
            else:
                rej.append(Rejected("countries", name, "country not in quote"))

    # ---- skills
    for sk in _as_list(raw.get("skills")):
        sk = _as_dict(sk)
        term = canonical(str(sk.get("term") or "").strip())
        if not term:
            continue
        if len(term.split()) > MAX_TERM_WORDS or len(term) > 50:
            rej.append(Rejected("skills", term, "too long to be a searchable skill"))
            continue
        if term in kw_mod.STOPWORDS or term in kw_mod.NON_SKILL_ACRONYMS:
            rej.append(Rejected("skills", term, "boilerplate word, not a skill"))
            continue
        if not grounded("skills", term, sk.get("quote")):
            continue
        if not _term_in_quote(term, sk["quote"]):
            rej.append(Rejected("skills", term, "term not in quote"))
            continue
        required = sk.get("required") is True
        prev = out.skills.get(term)
        # the same skill quoted twice: required anywhere means required
        if prev is None or (required and not prev[0]):
            out.skills[term] = (required, sk["quote"])
    return out


# ------------------------------------------------------------- cache + call

def _cache_key(jd_text: str, model: str, host: str) -> str:
    h = hashlib.sha256()
    for part in (PROMPT_VERSION, model or "", host or "", jd_text):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def cache_dir():
    return learned_mod.ats_home() / "cache" / "jd_llm"


def _cache_read(key: str) -> dict | None:
    p = cache_dir() / f"{key}.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _cache_write(key: str, raw: dict) -> None:
    try:
        d = cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{key}.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass   # a cache is an optimization; a read-only home must not fail the run


def call_llm(jd_text: str, model: str | None = None, host: str | None = None,
             api_key: str | None = None, provider: str | None = None,
             use_cache: bool = True, timeout: int = 180) -> tuple[dict | None, str | None, bool]:
    """(raw_answer, error, from_cache)."""
    cfg = llm_client.current_config()
    key = _cache_key(jd_text, model or cfg["model"], host or cfg["base_url"])
    if use_cache:
        hit = _cache_read(key)
        if hit is not None:
            return hit, None, True
    raw, err = llm_client.call_json(
        SYSTEM_PROMPT, "JOB DESCRIPTION:\n\n" + jd_text, model=model, host=host,
        api_key=api_key, provider=provider, timeout=timeout, temperature=0.0,
    )
    if raw is None:
        return None, err or "no answer", False
    if use_cache:
        _cache_write(key, raw)
    return raw, None, False


# ------------------------------------------------------------- merge

@dataclass
class Conflict:
    field: str
    llm: object
    rules: object


@dataclass
class HybridResult:
    reqs: jd_mod.JDRequirements
    keywords: list[kw_mod.JDKeyword]
    source: str                      # "llm+rules" | "rules"
    error: str | None = None         # why the LLM didn't contribute
    from_cache: bool = False
    verified: Verified | None = None
    conflicts: list[Conflict] = field(default_factory=list)
    learned: list[str] = field(default_factory=list)

    @property
    def rejected(self) -> list[Rejected]:
        return self.verified.rejected if self.verified else []

    def to_dict(self) -> dict:
        v = self.verified
        return {
            "source": self.source,
            "error": self.error,
            "from_cache": self.from_cache,
            "llm_skills": ({t: {"required": r, "quote": q} for t, (r, q) in v.skills.items()}
                           if v else {}),
            "rejected": [{"field": r.field, "value": r.value, "reason": r.reason}
                         for r in self.rejected],
            "conflicts": [{"field": c.field, "llm": c.llm, "rules": c.rules}
                          for c in self.conflicts],
            "learned_candidates": self.learned,
        }


def merge(rules_reqs: jd_mod.JDRequirements, rules_kws: list[kw_mod.JDKeyword],
          v: Verified) -> tuple[jd_mod.JDRequirements, list[kw_mod.JDKeyword], list[Conflict]]:
    """Verified LLM values win, the rules fill the gaps, skills are the union.

    Why the LLM wins a disagreement it can prove: its value passed a quote
    check against the JD, and the corpus shows the rules' misses are
    phrasing they don't know, not lines they misread. Every disagreement is
    returned, so a bad override is visible rather than silent."""
    r = rules_reqs
    conflicts: list[Conflict] = []
    out = jd_mod.JDRequirements(**{f: getattr(r, f) for f in r.__dataclass_fields__})
    out.certifications = list(r.certifications)
    out.work_modes = list(r.work_modes)
    out.countries = list(r.countries)
    out.evidence = list(r.evidence)

    def take(name, llm_val, rule_val, setter, quote_key=None):
        if llm_val is None or llm_val == [] or llm_val == "":
            return
        if rule_val not in (None, "", []) and rule_val != llm_val:
            conflicts.append(Conflict(name, llm_val, rule_val))
        setter(llm_val)
        if quote_key and v.quotes.get(quote_key):
            out.evidence.append(jd_mod.Requirement(name, llm_val, v.quotes[quote_key].strip(),
                                                   mandatory=True))

    take("title", v.title, r.jd_title, lambda x: setattr(out, "jd_title", x))
    take("seniority", v.seniority, r.seniority, lambda x: setattr(out, "seniority", x))
    if v.min_years is not None:
        take("min_years", v.min_years, r.min_years, lambda x: setattr(out, "min_years", x),
             "min_years")
    if v.min_degree:
        take("min_degree", v.min_degree, r.min_degree,
             lambda x: (setattr(out, "min_degree", x),
                        setattr(out, "degree_mandatory", v.degree_mandatory)), "min_degree")
    take("work_modes", sorted(v.work_modes), sorted(r.work_modes),
         lambda x: setattr(out, "work_modes", list(x)))
    if v.sponsorship_unavailable is True:
        take("sponsorship_unavailable", True, r.sponsorship_unavailable or None,
             lambda x: setattr(out, "sponsorship_unavailable", True), "sponsorship_unavailable")
    if v.salary:
        rule_sal = ([r.salary_min, r.salary_max, r.salary_currency]
                    if (r.salary_min is not None or r.salary_max is not None) else None)
        take("salary", v.salary, rule_sal,
             lambda x: (setattr(out, "salary_min", x[0]), setattr(out, "salary_max", x[1]),
                        setattr(out, "salary_currency", x[2])), "salary")

    # sets that are naturally additive: union (rules' precision on these is ~100%)
    for c in v.certifications:
        if not any(c.lower() in x.lower() or x.lower() in c.lower() for x in out.certifications):
            out.certifications.append(c)
    for c in v.countries:
        if c not in out.countries:
            out.countries.append(c)

    # skills: the LLM adds what the rules missed only when it is a REQUIREMENT,
    # and its required/preferred call re-sections terms both found. A term
    # only the LLM saw, and only in a duty or nice-to-have line, is left out:
    # on the tuning corpus those were mostly not searchable skills
    # (precision 98.5% -> 89.5% when they were added).
    by_term = {canonical(k.term): k for k in rules_kws}
    merged: dict[str, kw_mod.JDKeyword] = dict(by_term)
    for term, (required, _q) in v.skills.items():
        section = "hard" if required else "nice"
        old = by_term.get(term)
        if old is None:
            if required:
                merged[term] = kw_mod.JDKeyword(term, kw_mod.WEIGHTS[section], section)
            continue
        old_req = old.section in ("hard", "skills")
        if old_req != required:
            conflicts.append(Conflict(f"skill:{term}", section, old.section))
            bump = old.weight / kw_mod.WEIGHTS.get(old.section, 1.0)
            merged[term] = kw_mod.JDKeyword(term, round(kw_mod.WEIGHTS[section] * bump, 2), section)
    keywords = sorted(merged.values(), key=lambda k: -k.weight)
    return out, keywords, conflicts


def extract_hybrid(jd_text: str, top_n: int = 40, use_llm: bool = True,
                   model: str | None = None, host: str | None = None,
                   api_key: str | None = None, provider: str | None = None,
                   use_cache: bool = True, learn: bool = True) -> HybridResult:
    """Rules always run; the verified LLM answer is layered on when available."""
    rules_reqs = jd_mod.extract(jd_text)
    rules_kws = kw_mod.extract_jd_keywords(jd_text, top_n=1000)
    if not use_llm:
        return HybridResult(rules_reqs, rules_kws[:top_n], "rules", error="LLM extraction off")

    raw, err, cached = call_llm(jd_text, model=model, host=host, api_key=api_key,
                                provider=provider, use_cache=use_cache)
    if raw is None:
        return HybridResult(rules_reqs, rules_kws[:top_n], "rules",
                            error=f"LLM extraction unavailable ({err}); rules only")
    v = verify(raw, jd_text)
    reqs, kws, conflicts = merge(rules_reqs, rules_kws, v)

    learned: list[str] = []
    if learn:
        unknown = {t: q for t, (_r, q) in v.skills.items()
                   if t not in kw_mod.SKILL_TAXONOMY and canonical(t) not in
                   {canonical(k.term) for k in rules_kws}}
        learned = learned_mod.record_candidates(unknown)
    return HybridResult(reqs, kws[:top_n], "llm+rules", from_cache=cached, verified=v,
                        conflicts=conflicts, learned=learned)
