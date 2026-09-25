"""Weighted keyword/skill matching — the rule-based layer every real ATS
still runs (Boolean filters for tools, certifications, named platforms,
compliance terms), on top of which modern systems layer semantic scoring.

Methodology:
  - Split the JD into sections; terms found under "Requirements /
    Qualifications / Must have" are weighted highest (hard filters in real
    ATS), "Skills" next, "Responsibilities" lower, everything else baseline.
  - A broad taxonomy of tools/tech/business-skill terms across domains
    boosts precision over naive frequency counting (this is what tools like
    Jobscan approximate with their own internal skill dictionaries).
  - ALL-CAPS short tokens (acronyms: SQL, AWS, PMP, HIPAA) and versioned
    vendor exam codes (PL-300, AZ-104, DP-203) get an extra weight bump —
    these are almost always hard requirements, not narrative fluff.
  - Alias canonicalization (ats_checker.terms) runs on BOTH sides: a JD
    asking for "Postgres" matches a resume saying "PostgreSQL", and the
    generator selects against the same canonical terms the scorer measures.
    The scoring layer and the generator can no longer disagree about
    synonyms — one shared alias table decides.
  - A simple plural 's' tolerance means "dashboard" matches "dashboards"
    without full stemming.
  - Final score = (weight of matched keywords) / (weight of all keywords) * 100
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .terms import ALIASES, alias_normalize, canonical, term_pattern

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to", "in",
    "on", "at", "for", "with", "by", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its",
    "we", "you", "your", "our", "they", "their", "will", "would", "should",
    "can", "could", "may", "might", "must", "shall", "have", "has", "had",
    "do", "does", "did", "not", "no", "nor", "than", "too", "very", "just",
    "about", "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "again", "further", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "role", "job", "work", "working", "team", "company", "years",
    "year", "including", "etc", "strong", "excellent", "ability",
    "experience", "skills", "knowledge", "required", "preferred", "plus",
    "using", "used", "use", "within", "across", "candidate",
    "candidates", "applicant", "applicants", "opportunity", "environment",
}

SECTION_HEADERS = {
    "hard": [r"requirements?", r"qualifications?", r"must[\s-]?have",
             r"minimum qualifications?", r"basic qualifications?"],
    "skills": [r"skills?", r"technical skills?", r"tech stack", r"tools?"],
    "responsibilities": [r"responsibilit(y|ies)", r"what you.ll do",
                          r"duties", r"day[\s-]?to[\s-]?day"],
    "nice": [r"nice[\s-]?to[\s-]?have", r"preferred qualifications?",
              r"bonus"],
}

WEIGHTS = {"hard": 3.0, "skills": 2.5, "responsibilities": 1.5, "nice": 1.0, "default": 1.0}

# A broad taxonomy of terms that are almost always real signal in a JD
# (as opposed to filler words), spanning the common professional domains.
# Extend freely for your field.
SKILL_TAXONOMY = {
    # ---------------- data / BI / analytics ----------------
    "sql", "sql server", "python", "power bi", "dax", "excel", "vba",
    "tableau", "looker", "looker studio", "r", "sas", "spss", "stata",
    "etl", "elt", "dbt", "data warehouse", "data modeling", "data mining",
    "data visualization", "business intelligence", "data analysis",
    "data engineering", "data pipeline", "machine learning", "nlp", "llm",
    "generative ai", "ai", "deep learning", "tensorflow", "pytorch",
    "statistics", "statistical analysis", "a/b testing", "forecasting",
    "kpi", "dashboard", "reporting", "insights", "analytics",
    "snowflake", "bigquery", "redshift", "databricks", "synapse",
    "postgresql", "mysql", "mongodb", "nosql", "oracle",
    "teradata", "hive", "hadoop", "spark", "kafka", "airflow", "db2",
    "sqlite", "redis", "elasticsearch", "delta lake", "data governance",
    "data quality", "master data management", "powerpoint", "power query",
    "qlik", "qliksense", "microstrategy", "alteryx",
    "ssis", "ssrs", "ssms", "adobe analytics", "google analytics",
    "web analytics", "matillion", "fivetran", "airbyte", "streamlit",
    "pandas", "numpy", "jupyter", "scikit-learn", "copilot",
    # ---------------- software engineering ----------------
    "java", "c++", "c#", "scala", "golang", "rust", "php", "ruby",
    "javascript", "typescript", "react", "angular", "vue", "next.js",
    "node.js", "html", "css", "rest api", "graphql", "api design",
    "microservices", "flask", "django", "fastapi", "spring boot",
    "laravel", "asp.net", ".net core", "spring", "hibernate",
    "unit testing", "test automation", "selenium", "cypress", "junit",
    "pytest", "code review", "object-oriented programming",
    "system design", "distributed systems", "debugging",
    # ---------------- cloud / devops / infrastructure ----------------
    "aws", "azure", "gcp", "cloud", "docker", "kubernetes", "ci/cd",
    "devops", "terraform", "ansible", "jenkins", "github actions",
    "gitlab ci", "linux", "bash", "shell scripting", "powershell",
    "windows server", "vmware", "networking", "tcp/ip", "dns",
    "load balancing", "serverless", "ec2", "s3", "rds", "azure devops",
    "sre", "infrastructure as code", "monitoring", "observability",
    "prometheus", "grafana", "datadog", "splunk", "nginx", "saas",
    # ---------------- security / compliance ----------------
    "cybersecurity", "information security", "penetration testing",
    "vulnerability management", "incident response", "siem", "soc",
    "encryption", "iam", "zero trust", "firewall", "risk assessment",
    "iso 27001", "soc 2", "gdpr", "hipaa", "pci dss", "sox",
    "governance risk compliance", "grc", "compliance", "disaster recovery",
    # ---------------- project / delivery / method ----------------
    "project management", "program management", "agile", "scrum", "kanban",
    "waterfall", "jira", "confluence", "asana", "trello",
    "monday.com", "notion", "stakeholder management", "change management",
    "roadmap", "okr", "process improvement", "lean", "six sigma",
    "business analysis", "requirements gathering", "user stories",
    "gap analysis", "bpmn", "process mapping", "vendor management",
    "budget management", "resource planning", "pmp", "prince2",
    # ---------------- sales / marketing / support ----------------
    "crm", "salesforce", "hubspot", "zoho", "dynamics 365", "sap",
    "erp", "marketing automation", "seo", "sem", "google ads",
    "meta ads", "content marketing", "email marketing", "copywriting",
    "brand management", "market research", "customer segmentation",
    "campaign management", "conversion rate optimization",
    "lead generation", "pipeline management", "inside sales",
    "account management", "customer success", "customer retention",
    "upselling", "negotiation", "zendesk", "freshdesk", "intercom",
    "helpdesk", "itil", "service desk", "technical support",
    "customer service", "call center", "contact center", "bpo",
    # ---------------- finance / accounting ----------------
    "financial analysis", "financial modeling", "financial reporting",
    "budgeting", "variance analysis", "p&l",
    "accounts payable", "accounts receivable", "general ledger",
    "reconciliation", "month-end close", "taxation", "gst", "vat",
    "us gaap", "ifrs", "audit", "payroll", "quickbooks", "sap fico",
    "netsuite", "treasury", "credit analysis",
    "underwriting", "risk management", "fp&a", "billing",
    "cost accounting", "cpa", "banking operations",
    # ---------------- HR / people ops ----------------
    "recruitment", "talent acquisition", "onboarding", "offboarding",
    "hris", "workday", "bamboohr", "successfactors", "payroll processing",
    "employee engagement", "performance management",
    "compensation and benefits", "hr compliance",
    # ---------------- design / product ----------------
    "ux", "ui", "figma", "adobe xd", "wireframing", "prototyping",
    "user research", "usability testing", "design system", "photoshop",
    "illustrator", "indesign", "after effects", "premiere pro", "canva",
    "product management", "product roadmap", "go-to-market",
    "user experience design", "interaction design", "responsive design",
    # ---------------- healthcare / science / other domains ----------------
    "clinical trials", "medical coding", "medical billing", "ehr",
    "epic", "cerner", "patient care", "nursing", "phlebotomy",
    "medical terminology", "pharmacology",
    "laboratory", "gmp", "clinical research", "regulatory affairs",
    "supply chain", "logistics", "procurement", "inventory management",
    "warehouse management", "demand planning", "quality control",
    "quality assurance", "iso 9001", "preventive maintenance", "calibration",
    "autocad", "solidworks", "revit", "cnc", "plc", "scada", "hvac",
    "instrumentation", "mechanical engineering", "civil engineering",
    "curriculum development", "lesson planning", "classroom management",
    # ---------------- generic hard business skills ----------------
    "microsoft office", "microsoft 365", "sharepoint", "microsoft teams",
    "power automate", "power apps", "power platform",
    "microsoft copilot studio", "copilot studio", "servicenow", "zapier",
    "n8n", "automation anywhere", "uipath", "blue prism", "rpa",
    "excel macros", "pivot tables", "vlookup", "communication",
    "problem solving", "critical thinking", "attention to detail",
    "time management", "public speaking", "presentation",
    "documentation", "report writing", "mentoring", "training delivery",
    "cross-functional collaboration", "client-facing",
    "escalation management", "openai", "chatgpt", "claude", "gemini",
    "prompt engineering", "vector database", "mlops", "blockchain",
    # ---------------- business analysis / pre-sales / consulting ----------------
    "rfp", "rfi", "brd", "frd", "sow", "rfp response", "rfi response",
    "request for proposal", "request for information",
    "solution design", "solutioning", "presales", "pre-sales",
    "client engagement", "client requirements", "functional requirements",
    "non-functional requirements", "scope document", "use case",
    "user acceptance testing", "uat", "integration testing",
    "test cases", "root cause analysis", "change request",
    "workflow charts", "workflow diagram", "process documentation",
    "specification", "specification document", "requirement elicitation",
    "stakeholder workshop", "requirements sign-off", "solution demo",
    "prototype", "solution presentation", "client demo", "whitepaper",
    "point of view", "domain expertise", "knowledge repository",
    "client cadence", "engagement partner", "consulting",
    "milestone planning", "delivery team", "principal consultant",
    "financial impact", "operational impact", "system capabilities",
    "data modelling", "workflow design", "business requirement document",
    "functional requirement document", "client requirement",
    "requirements clarification", "requirements analysis",
    "requirements documentation", "requirements traceability",
    "solution architecture", "market research", "lead generation",
    "customer satisfaction", "requirements workshop",
}

MULTIWORD_TAXONOMY = sorted((t for t in SKILL_TAXONOMY if " " in t), key=len, reverse=True)

# Versioned vendor exam codes (PL-300, AZ-104, DP-203...) are near-always
# hard requirements — treat like acronyms for extraction.
EXAM_CODE_RE = re.compile(r"^[A-Z]{2}-\d{3}$")


@dataclass
class JDKeyword:
    term: str
    weight: float
    section: str


@dataclass
class KeywordMatchResult:
    matched: list[JDKeyword] = field(default_factory=list)
    missing: list[JDKeyword] = field(default_factory=list)
    score: float = 0.0

    @property
    def matched_terms(self) -> list[str]:
        return [k.term for k in self.matched]

    @property
    def missing_terms(self) -> list[str]:
        return [k.term for k in self.missing]


def _classify_line_section(line: str, current: str) -> str:
    lower = line.strip().lower().rstrip(":")
    for section, patterns in SECTION_HEADERS.items():
        for pat in patterns:
            if re.fullmatch(pat, lower) or (len(lower) < 40 and re.search(pat, lower)):
                return section
    return current


def _extract_multiword_terms(text_lower: str) -> set[str]:
    return {term for term in MULTIWORD_TAXONOMY if term in text_lower}


def _plural_fold(token: str) -> str:
    """'kpis' -> 'kpi', 'dashboards' -> 'dashboard'. Only when the folded
    form is itself a known term, so narrative words are untouched."""
    if len(token) > 3 and token.endswith("s") and token[:-1] in SKILL_TAXONOMY:
        return token[:-1]
    return token


def extract_jd_keywords(jd_text: str, top_n: int = 40) -> list[JDKeyword]:
    """Walk the JD line by line, tracking which section we're in, and pull
    out weighted keyword candidates (taxonomy terms + acronyms + exam
    codes), canonicalized through the shared alias table."""
    lines = jd_text.splitlines()
    current_section = "default"
    section_by_term: dict[str, str] = {}

    normalized_full = alias_normalize(jd_text.lower())
    full_lower = jd_text.lower()
    # Multiword taxonomy scan runs on the ALIAS-NORMALIZED text: a JD that
    # writes "data modelling" (British) must surface the taxonomy's
    # "data modeling" — that is the entire point of the shared alias table.
    for term in _extract_multiword_terms(normalized_full):
        idx = normalized_full.find(canonical(term)) if canonical(term) != term else normalized_full.find(term)
        section_by_term[term] = _last_section_before(
            normalized_full[: idx if idx >= 0 else 0]
        )

    candidates: dict[str, tuple[str, float]] = {
        canonical(term): (section, 1.0) for term, section in section_by_term.items()
    }

    for line in lines:
        current_section = _classify_line_section(line, current_section)
        # A shouted headline ("WE'RE HIRING | BUSINESS ANALYST") makes every
        # word look like an acronym; on such lines only trust taxonomy terms
        # and exam codes.
        shouting = _is_shouting(line)
        line_toks = re.findall(r"[A-Za-z][A-Za-z0-9+.#/-]{1,}", line)
        for i, tok in enumerate(line_toks):
            # strip trailing punctuation the token regex swallows ("RFP/",
            # "BA s" -> "RFP", "BA") — an acronym glued to a slash must still
            # be recognized as the acronym.
            tok_clean = tok.rstrip("+.#/-")
            if not tok_clean:
                continue
            low = _plural_fold(tok_clean.lower())
            if low in STOPWORDS or len(low) < 2:
                continue
            is_acronym = (
                tok_clean.isupper() and tok_clean.isalpha() and 2 <= len(tok_clean) <= 6
                and not shouting
                and low not in ACRONYM_NOISE
                # "MS" in "MS Excel": the prefix of an alias phrase the
                # taxonomy scan already captured (as "excel"), not a skill.
                and not _starts_alias_phrase(low, line_toks[i + 1:i + 3])
                # "BI" in "Power BI": the tail of a multiword term already
                # captured whole, not a second requirement.
                and not (i > 0 and _ends_known_phrase(line_toks[i - 1], low))
            )
            is_exam_code = bool(EXAM_CODE_RE.match(tok_clean))
            is_taxonomy = low in SKILL_TAXONOMY
            if not (is_acronym or is_exam_code or is_taxonomy):
                continue
            canon = canonical(low)
            if canon in candidates:
                continue
            # acronym/exam-code bump survives canonicalization
            bump = 1.3 if (is_acronym or is_exam_code) else 1.0
            candidates[canon] = (current_section, bump)

    keywords = []
    for term, (section, bump) in candidates.items():
        weight = WEIGHTS.get(section, WEIGHTS["default"]) * bump
        keywords.append(JDKeyword(term=term, weight=round(weight, 2), section=section))

    keywords.sort(key=lambda k: k.weight, reverse=True)
    return keywords[:top_n]


# Capitalised tokens that are application boilerplate, not skills.
ACRONYM_NOISE = {"cv", "re", "ll", "ve", "jd", "asap", "fyi", "etc", "ctc", "lpa", "pm", "am", "dm", "pfb"}


def _is_shouting(line: str) -> bool:
    words = re.findall(r"[A-Za-z]{2,}", line)
    if len(words) < 3:
        return False
    return sum(w.isupper() for w in words) / len(words) >= 0.7


def _starts_alias_phrase(low: str, following: list[str]) -> bool:
    for n in range(1, len(following) + 1):
        phrase = " ".join([low] + [t.rstrip("+.#/-").lower() for t in following[:n]])
        if phrase in ALIASES:
            return True
    return False


def _ends_known_phrase(prev_tok: str, low: str) -> bool:
    phrase = f"{prev_tok.rstrip('+.#/-').lower()} {low}"
    return phrase in SKILL_TAXONOMY or phrase in ALIASES


def _last_section_before(text_before: str) -> str:
    section = "default"
    for line in text_before.splitlines():
        section = _classify_line_section(line, section)
    return section


def score_keywords(resume_text: str, jd_keywords: list[JDKeyword]) -> KeywordMatchResult:
    """Alias-aware weighted matching.

    The JD's terms are canonicalized and merged (a requirement written three
    ways in one JD is still one requirement — merged with MAX weight, not
    summed), the resume is alias-normalized, and matching uses the shared
    plural-tolerant pattern. A resume saying 'PostgreSQL' now MATCHES a JD
    asking for 'Postgres' — the same fact the generator's selector already
    knew — instead of being reported missing by the scorer.
    """
    result = KeywordMatchResult()
    resume_normalized = alias_normalize((resume_text or "").lower())

    merged: dict[str, JDKeyword] = {}
    for kw in jd_keywords:
        c = canonical(kw.term)
        existing = merged.get(c)
        if existing is None or kw.weight > existing.weight:
            merged[c] = kw

    total_weight = sum(k.weight for k in merged.values()) or 1.0
    matched_weight = 0.0

    for c, kw in merged.items():
        if term_pattern(c).search(resume_normalized):
            result.matched.append(kw)
            matched_weight += kw.weight
        else:
            result.missing.append(kw)

    result.matched.sort(key=lambda k: k.weight, reverse=True)
    result.missing.sort(key=lambda k: k.weight, reverse=True)
    result.score = round((matched_weight / total_weight) * 100, 1)
    return result


def skill_tokens(text: str) -> set[str]:
    """Every canonical skill term a text demonstrates: taxonomy terms, alias
    variants, ALL-CAPS acronyms, and vendor exam codes. Used by the rewrite
    verifier to enforce that a rewording never silently drops or swaps a
    named skill — the deterministic guard behind 'the LLM only rewords,
    never invents'."""
    haystack = alias_normalize((text or "").lower())
    found: set[str] = set()
    for term in MULTIWORD_TAXONOMY:
        if term in haystack:
            found.add(term)
    for tok in re.findall(r"[a-z][a-z0-9+.#/-]{1,}", haystack):
        low = _plural_fold(tok)
        if low in SKILL_TAXONOMY and len(low) >= 2:
            found.add(low)
    multiword = set(MULTIWORD_TAXONOMY)
    words = re.findall(r"[A-Za-z][A-Za-z0-9+.#/-]{1,}", text or "")
    for idx, tok in enumerate(words):
        if tok.isupper() and tok.isalpha() and 2 <= len(tok) <= 6:
            low = tok.lower()
            # 'BI' inside the phrase 'Power BI' is part of the multiword term,
            # not a separate skill — adding both would false-gate every
            # rewrite that touches the phrase's spelling.
            prev = words[idx - 1].lower() if idx > 0 else ""
            if prev and f"{prev} {low}" in multiword:
                continue
            found.add(canonical(low))
        elif EXAM_CODE_RE.match(tok):
            found.add(tok.lower())
    return found