"""Measure the JD extractors against a hand-labelled corpus.

"The rule engine is brittle" is a feeling until it's a number. This module
turns it into numbers: every file in the corpus holds a real-world-style JD
plus what a careful human would extract from it (title, years floor, degree,
certifications, work mode, sponsorship, salary, countries, the required
skills), and the evaluator reports how often an extractor gets each field
right — with every miss listed, so a regression is visible line by line.

Labels describe the TRUTH of the JD, not what the current code happens to
return. A requirement the taxonomy doesn't know yet ("GD&T", "ISTQB") is
still labelled — it shows up as a recall miss, which is the honest signal
that coverage is missing.

The extractor is a plain function (jd_text -> Extraction), so the Phase 3
LLM extractor can be scored on exactly the same corpus and compared.

Corpus file format (YAML, one JD per file):

    id: data_analyst_blr
    domain: data / analytics
    jd: |
      ...the job description...
    expected:                 # omit a key to leave that field unscored
      title: data analyst     # searchable role title, lowercase ('' = none)
      seniority: ""           # intern/junior/.../senior/lead/manager/... or ""
      min_years: 2            # null = JD states none
      min_degree: bachelors   # diploma|bachelors|masters|mba|phd, or null
      degree_mandatory: true  # only scored when min_degree is set
      certifications: [PMP]   # named certs, any casing
      work_modes: [hybrid]    # remote|hybrid|onsite
      sponsorship_unavailable: false
      salary: [600000, 900000, INR]   # [min, max, currency]; null = none stated
      countries: [india]
      required_terms: [sql, python]   # skills a recruiter would treat as must-have
      other_terms: [tableau]          # other genuine skills/tools in the JD
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import jd_requirements as jd_mod
from . import keywords as kw_mod
from .terms import canonical

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "tests" / "jd_corpus"
HARD_SECTIONS = ("hard", "skills")

SCALAR_FIELDS = ("title", "seniority", "min_years", "min_degree", "degree_mandatory",
                 "sponsorship_unavailable", "salary")
SET_FIELDS = ("certifications", "work_modes", "countries")


@dataclass
class Extraction:
    """What an extractor pulled out of one JD, in comparable form."""
    title: str = ""
    seniority: str = ""
    min_years: float | None = None
    min_degree: str | None = None
    degree_mandatory: bool | None = None
    certifications: list[str] = field(default_factory=list)
    work_modes: list[str] = field(default_factory=list)
    sponsorship_unavailable: bool = False
    salary: list | None = None            # [min, max, currency] or None
    countries: list[str] = field(default_factory=list)
    keywords: dict[str, str] = field(default_factory=dict)   # canonical term -> section


Extractor = Callable[[str], Extraction]


def rule_extractor(jd_text: str) -> Extraction:
    """The current deterministic engine (jd_requirements + keywords)."""
    reqs = jd_mod.extract(jd_text)
    kws = kw_mod.extract_jd_keywords(jd_text, top_n=1000)
    salary = None
    if reqs.salary_min is not None or reqs.salary_max is not None:
        salary = [reqs.salary_min, reqs.salary_max, reqs.salary_currency]
    return Extraction(
        title=reqs.jd_title,
        seniority=reqs.seniority,
        min_years=reqs.min_years,
        min_degree=reqs.min_degree,
        degree_mandatory=reqs.degree_mandatory if reqs.min_degree else None,
        certifications=list(reqs.certifications),
        work_modes=list(reqs.work_modes),
        sponsorship_unavailable=reqs.sponsorship_unavailable,
        salary=salary,
        countries=list(reqs.countries),
        keywords={canonical(k.term): k.section for k in kws},
    )


EXTRACTORS: dict[str, Extractor] = {"rules": rule_extractor}


# ------------------------------------------------------------------ corpus IO

@dataclass
class CorpusItem:
    id: str
    domain: str
    jd: str
    expected: dict
    path: str = ""


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[CorpusItem]:
    import yaml

    items = []
    for p in sorted(Path(path).glob("*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        items.append(CorpusItem(
            id=str(data.get("id") or p.stem), domain=str(data.get("domain") or ""),
            jd=str(data.get("jd") or ""), expected=dict(data.get("expected") or {}),
            path=str(p),
        ))
    return items


def validate_item(item: CorpusItem) -> list[str]:
    """Schema problems in one corpus file (empty list = valid)."""
    problems = []
    if not item.jd.strip():
        problems.append("empty jd")
    allowed = set(SCALAR_FIELDS) | set(SET_FIELDS) | {"required_terms", "other_terms"}
    for key in item.expected:
        if key not in allowed:
            problems.append(f"unknown expected key {key!r}")
    sal = item.expected.get("salary")
    if sal is not None and not (isinstance(sal, list) and len(sal) == 3):
        problems.append("salary must be [min, max, currency] or null")
    if not item.expected.get("required_terms"):
        problems.append("required_terms must list at least one term")
    return problems


# ------------------------------------------------------------------ scoring

def _norm(s) -> str:
    return " ".join(str(s or "").lower().split())


def _scalar_equal(name: str, want, got) -> bool:
    if name == "salary":
        if want is None or got is None:
            return want is None and got is None
        w_min, w_max, w_cur = want
        g_min, g_max, g_cur = got
        return (_num_eq(w_min, g_min) and _num_eq(w_max, g_max)
                and _norm(w_cur) == _norm(g_cur))
    if name == "min_years":
        return _num_eq(want, got)
    if isinstance(want, str) or isinstance(got, str):
        return _norm(want) == _norm(got)
    return want == got


def _num_eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) < 1e-6


def _cert_match(a: str, b: str) -> bool:
    """'PMP' matches 'PMP'; 'AWS Certified Solutions Architect' matches
    'AWS Certified Solutions Architect Associate' (one contains the other)."""
    a, b = _norm(a), _norm(b)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def _set_counts(name: str, want: list, got: list) -> tuple[int, int, int]:
    """(true positives, extracted count, expected count)."""
    want = [_norm(w) for w in want or []]
    got = [_norm(g) for g in got or []]
    if name == "certifications":
        tp_want = sum(1 for w in want if any(_cert_match(w, g) for g in got))
        tp_got = sum(1 for g in got if any(_cert_match(w, g) for w in want))
        return min(tp_want, tp_got), len(got), len(want)
    tp = len(set(want) & set(got))
    return tp, len(set(got)), len(set(want))


@dataclass
class FieldStat:
    correct: int = 0
    total: int = 0
    tp: int = 0          # set fields: item-level true positives
    extracted: int = 0
    expected: int = 0

    @property
    def accuracy(self) -> float | None:
        return round(self.correct / self.total, 3) if self.total else None

    @property
    def precision(self) -> float | None:
        return round(self.tp / self.extracted, 3) if self.extracted else None

    @property
    def recall(self) -> float | None:
        return round(self.tp / self.expected, 3) if self.expected else None


@dataclass
class Miss:
    item: str
    field: str
    expected: object
    got: object


@dataclass
class EvalReport:
    extractor: str
    n_items: int
    fields: dict[str, FieldStat]
    misses: list[Miss]

    def metrics(self) -> dict[str, float]:
        """Flat metric name -> value, for baselines and comparisons."""
        out: dict[str, float] = {}
        for name, st in self.fields.items():
            if name in SET_FIELDS or name.startswith("terms_"):
                for label, val in (("precision", st.precision), ("recall", st.recall)):
                    if val is not None:
                        out[f"{name}.{label}"] = val
            if st.accuracy is not None and not name.startswith("terms_"):
                out[f"{name}.accuracy"] = st.accuracy
        return out


def evaluate(items: list[CorpusItem], extractor: Extractor, name: str = "rules") -> EvalReport:
    fields: dict[str, FieldStat] = {f: FieldStat() for f in (*SCALAR_FIELDS, *SET_FIELDS)}
    fields["terms_required"] = FieldStat()       # required found anywhere
    fields["terms_required_hard"] = FieldStat()  # required found under Requirements/Skills
    fields["terms_all"] = FieldStat()            # precision of everything extracted
    misses: list[Miss] = []

    for item in items:
        exp = item.expected
        got = extractor(item.jd)

        for f in SCALAR_FIELDS:
            if f not in exp:
                continue
            if f == "degree_mandatory" and not exp.get("min_degree"):
                continue
            st = fields[f]
            st.total += 1
            g = getattr(got, f)
            if _scalar_equal(f, exp[f], g):
                st.correct += 1
            else:
                misses.append(Miss(item.id, f, exp[f], g))

        for f in SET_FIELDS:
            if f not in exp:
                continue
            st = fields[f]
            st.total += 1
            tp, n_got, n_want = _set_counts(f, exp[f] or [], getattr(got, f))
            st.tp += tp
            st.extracted += n_got
            st.expected += n_want
            if tp == n_got == n_want:
                st.correct += 1
            else:
                misses.append(Miss(item.id, f, exp[f] or [], getattr(got, f)))

        required = {canonical(t) for t in exp.get("required_terms") or []}
        acceptable = required | {canonical(t) for t in exp.get("other_terms") or []}
        extracted = got.keywords

        found = required & set(extracted)
        hard = {t for t in found if extracted[t] in HARD_SECTIONS}
        for key, hits in (("terms_required", found), ("terms_required_hard", hard)):
            st = fields[key]
            st.tp += len(hits)
            st.expected += len(required)
            st.extracted += len(required)
        missing = sorted(required - found)
        if missing:
            misses.append(Miss(item.id, "required_terms (not extracted)", missing, None))
        soft = sorted(found - hard)
        if soft:
            misses.append(Miss(item.id, "required_terms (extracted, but not as requirement)",
                               soft, {t: extracted[t] for t in soft}))

        st = fields["terms_all"]
        good = set(extracted) & acceptable
        st.tp += len(good)
        st.extracted += len(extracted)
        st.expected += len(acceptable)
        noise = sorted(set(extracted) - acceptable)
        if noise:
            misses.append(Miss(item.id, "extracted terms that aren't skills", noise, None))

    return EvalReport(extractor=name, n_items=len(items), fields=fields, misses=misses)


def print_report(report: EvalReport, show_misses: bool = True) -> None:
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    console = Console()
    t = Table(title=f"JD extraction vs {report.n_items} labelled JDs — extractor: {report.extractor}",
              show_header=True, header_style="bold")
    t.add_column("Field")
    t.add_column("Accuracy", justify="right")
    t.add_column("Precision", justify="right")
    t.add_column("Recall", justify="right")
    t.add_column("n", justify="right")

    def pct(v):
        if v is None:
            return "—"
        c = "green" if v >= 0.9 else "yellow" if v >= 0.7 else "red"
        return f"[{c}]{v:.0%}[/{c}]"

    labels = {
        "terms_required": "required skills — found",
        "terms_required_hard": "required skills — found AS requirements",
        "terms_all": "all extracted terms",
    }
    for name, st in report.fields.items():
        is_set = name in SET_FIELDS or name.startswith("terms_")
        t.add_row(labels.get(name, name),
                  pct(st.accuracy) if not name.startswith("terms_") else "—",
                  pct(st.precision) if is_set else "—",
                  pct(st.recall) if is_set else "—",
                  str(st.total if not name.startswith("terms_") else st.expected))
    console.print(t)
    console.print("[dim]Accuracy = share of JDs where the field is exactly right. Precision = "
                  "share of extracted items that are real; recall = share of real items that "
                  "were extracted. Labels describe the JD, not the current code.[/dim]")

    if show_misses and report.misses:
        m = Table(title=f"Misses ({len(report.misses)})", show_header=True, header_style="bold")
        m.add_column("JD")
        m.add_column("Field")
        m.add_column("Expected")
        m.add_column("Got")
        for miss in report.misses:
            m.add_row(miss.item, miss.field, escape(str(miss.expected)), escape(str(miss.got)))
        console.print(m)
