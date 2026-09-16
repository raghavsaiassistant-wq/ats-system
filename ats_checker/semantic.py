"""Semantic fit scoring via Ollama — part of the ATS layer.

Modern ATS platforms (Workday, Eightfold, Phenom) embed the resume and JD
into vector space and score by closeness of *meaning*, not just string
overlap — so "reduced customer churn" can match "improved retention rate"
even with zero shared keywords. We approximate that layer with an LLM
judgment instead of raw embeddings, because it also yields readable
reasoning (strengths, gaps, rewording hints), which is more useful for a
personal job-search tool than a bare cosine-similarity number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import llm_client as ollama_client

# Re-exported so existing callers/imports keep working
DEFAULT_HOST = ollama_client.DEFAULT_HOST
DEFAULT_MODEL = ollama_client.DEFAULT_MODEL
DEFAULT_API_KEY = ollama_client.DEFAULT_API_KEY

SYSTEM_PROMPT = """You are an experienced technical recruiter and ATS analyst.
You will be given a JOB DESCRIPTION and a RESUME. Judge contextual/semantic
fit the way a modern embedding-based ATS ranking layer would: reward
equivalent experience described in different words, and do not just count
keyword repetition. Be honest and specific — this is for the candidate's own
job-search prep, not a sales pitch.

Respond with STRICT JSON ONLY, no markdown fences, matching this schema:
{
  "semantic_score": <integer 0-100, overall contextual fit>,
  "strengths": [<3-6 short strings: genuine matches between resume and JD>],
  "gaps": [<2-6 short strings: real gaps or unclear areas vs the JD>],
  "reworded_matches": [<0-5 short strings: resume experience that matches JD
                        intent but uses different wording than the JD>],
  "recommendation": <one paragraph, 2-4 sentences, direct and actionable>
}"""


@dataclass
class SemanticResult:
    available: bool
    semantic_score: int | None = None
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    reworded_matches: list[str] = field(default_factory=list)
    recommendation: str = ""
    error: str | None = None
    model: str = ""


def score_semantic(
    resume_text: str,
    jd_text: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    api_key: str = DEFAULT_API_KEY,
    provider: str | None = None,
    timeout: int = 180,
) -> SemanticResult:
    """Fails soft: on any connection/parse error returns
    SemanticResult(available=False, error=...) so the caller can fall back
    to keyword-only scoring instead of crashing."""
    parsed, error = ollama_client.call_json(
        SYSTEM_PROMPT,
        f"JOB DESCRIPTION:\n{jd_text.strip()}\n\nRESUME:\n{resume_text.strip()}",
        model=model,
        host=host,
        api_key=api_key,
        provider=provider,
        timeout=timeout,
    )
    if error or parsed is None:
        return SemanticResult(available=False, error=error or "Unknown Ollama error")

    try:
        score = max(0, min(100, int(parsed.get("semantic_score", 0))))
    except (TypeError, ValueError):
        score = 0

    return SemanticResult(
        available=True,
        semantic_score=score,
        strengths=[str(s) for s in parsed.get("strengths", [])],
        gaps=[str(s) for s in parsed.get("gaps", [])],
        reworded_matches=[str(s) for s in parsed.get("reworded_matches", [])],
        recommendation=str(parsed.get("recommendation", "")),
        model=model,
    )
