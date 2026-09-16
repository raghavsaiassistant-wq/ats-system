"""Manager Evidence Score — how well the resume holds up to the person who
actually has to decide you can do the job.

Important framing: this is NOT a probability that a hiring manager passes
you. A manager's decision depends mostly on things absent from both
documents — who else is in the shortlist, what the team is missing right
now, budget, whether someone is already lined up. What this DOES measure is
evidence strength: whether your claims are specific, quantified, backed by
actual work, matched in scope to the role, and whether they'd survive an
interview question. That part is genuinely assessable from the text, and
it's the part you can act on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import llm_client as ollama_client

RUBRIC = {
    "quantification": "Are accomplishments backed by numbers, scale, or measurable impact "
                      "rather than vague description?",
    "outcome_focus": "Do bullets describe outcomes achieved ('built X, which cut Y by Z') "
                     "rather than responsibilities held ('responsible for X')?",
    "evidence_backing": "Is each significant claimed skill demonstrated somewhere in actual "
                        "work/projects, rather than only listed in a skills section?",
    "scope_match": "Does the scale and autonomy of the work shown match what this role's "
                   "level would require?",
    "domain_relevance": "Is the day-to-day work actually similar to what this JD needs, as "
                        "opposed to superficially adjacent?",
    "credibility": "Are claims specific and defensible, with no inflated or unverifiable-looking "
                   "assertions that would collapse under one follow-up question?",
}

SYSTEM_PROMPT = """You are a demanding hiring manager reviewing a resume for a role you
need to fill. You are NOT screening for keywords — HR already did that. You
are judging whether this person can actually do the work, based on the
evidence in front of you.

Score each rubric dimension 0-100. Be strict and specific: a resume full of
responsibility statements with no numbers should score low on quantification
and outcome_focus even if the person seems competent. Do not inflate scores
to be encouraging — the candidate is using this to fix their resume, so
vague praise actively harms them.

Respond with STRICT JSON ONLY, no markdown fences, matching this schema:
{
  "quantification": <0-100>,
  "outcome_focus": <0-100>,
  "evidence_backing": <0-100>,
  "scope_match": <0-100>,
  "domain_relevance": <0-100>,
  "credibility": <0-100>,
  "weak_bullets": [
    {"bullet": "<the weak line, quoted from the resume>",
     "problem": "<what's wrong with it in one short sentence>",
     "rewrite": "<a stronger version that stays truthful to what was stated;
                  use [X] placeholders where a real number is needed>"}
  ],
  "interview_risks": [<0-4 short strings: claims a manager would probe and the
                       candidate had better be able to defend>],
  "manager_verdict": "<2-4 sentences: would you shortlist this, and what is the
                       single biggest thing holding it back>"
}

Include at most 5 weak_bullets — the highest-leverage ones only."""


@dataclass
class ManagerResult:
    available: bool
    score: int | None = None
    dimensions: dict = field(default_factory=dict)
    weak_bullets: list[dict] = field(default_factory=list)
    interview_risks: list[str] = field(default_factory=list)
    verdict: str = ""
    model: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "score": self.score,
            "dimensions": self.dimensions,
            "weak_bullets": self.weak_bullets,
            "interview_risks": self.interview_risks,
            "verdict": self.verdict,
            "model": self.model,
            "error": self.error,
        }


# Weights: the first three are what most resumes actually fail on, and are
# the most fixable. scope/domain are about genuine fit, credibility gates.
DIMENSION_WEIGHTS = {
    "quantification": 0.20,
    "outcome_focus": 0.20,
    "evidence_backing": 0.20,
    "scope_match": 0.15,
    "domain_relevance": 0.15,
    "credibility": 0.10,
}


def score_manager_review(
    resume_text: str,
    jd_text: str,
    model: str = ollama_client.DEFAULT_MODEL,
    host: str = ollama_client.DEFAULT_HOST,
    api_key: str = ollama_client.DEFAULT_API_KEY,
    provider: str | None = None,
) -> ManagerResult:
    parsed, error = ollama_client.call_json(
        SYSTEM_PROMPT,
        f"JOB DESCRIPTION:\n{jd_text.strip()}\n\nRESUME:\n{resume_text.strip()}",
        model=model,
        host=host,
        api_key=api_key,
        provider=provider,
    )
    if error or parsed is None:
        return ManagerResult(available=False, error=error or "Unknown Ollama error")

    dimensions = {}
    for key in DIMENSION_WEIGHTS:
        try:
            dimensions[key] = max(0, min(100, int(parsed.get(key, 0))))
        except (TypeError, ValueError):
            dimensions[key] = 0

    weighted = sum(dimensions[k] * w for k, w in DIMENSION_WEIGHTS.items())

    weak = []
    for item in parsed.get("weak_bullets", [])[:5]:
        if isinstance(item, dict):
            weak.append({
                "bullet": str(item.get("bullet", "")),
                "problem": str(item.get("problem", "")),
                "rewrite": str(item.get("rewrite", "")),
            })

    return ManagerResult(
        available=True,
        score=round(weighted),
        dimensions=dimensions,
        weak_bullets=weak,
        interview_risks=[str(r) for r in parsed.get("interview_risks", [])[:4]],
        verdict=str(parsed.get("manager_verdict", "")),
        model=model,
    )
