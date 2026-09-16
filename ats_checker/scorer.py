"""Three-layer scoring pipeline.

  Layer 1 — ATS Score              : machine filter (parse + keywords + semantic)
  Layer 2 — Recruiter Screen Score : the ~30s human hard-filter checklist
  Layer 3 — Manager Evidence Score : does the evidence hold up to someone
                                     who has to decide you can do the job

These are deliberately reported as three separate scores, not blended into
one number, because they fail for different reasons and the fix for each is
different. There is no combined "you'll get the job" score, and none of the
three is a calibrated probability — see README.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import jd_requirements as jd_mod
from . import keywords as kw_mod
from . import manager as mgr_mod
from . import llm_client as ollama_client
from . import parsing
from . import recruiter as rec_mod
from . import semantic as sem_mod
from .profile import CandidateProfile

ATS_WEIGHTS = {"formatting": 0.15, "keyword": 0.40, "semantic": 0.45}
ATS_WEIGHTS_NO_LLM = {"formatting": 0.25, "keyword": 0.75, "semantic": 0.0}


def band(score: float | None) -> str:
    if score is None:
        return "Not scored"
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Workable"
    if score >= 40:
        return "Weak"
    return "Very weak"


@dataclass
class FullReport:
    ats_score: float
    ats_components: dict
    ats_weights: dict
    recruiter_score: float | None
    manager_score: int | None

    parse_result: parsing.ParseResult
    keyword_result: kw_mod.KeywordMatchResult
    semantic_result: sem_mod.SemanticResult
    recruiter_result: rec_mod.RecruiterResult | None
    manager_result: mgr_mod.ManagerResult | None
    jd_reqs: jd_mod.JDRequirements

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        rec = self.recruiter_result
        return {
            "scores": {
                "ats_score": self.ats_score,
                "hr_screen_criteria_met_pct": self.recruiter_score,
                "hr_resume_fixable_pct": rec.resume_pct if rec else None,
                "hr_candidacy_fit_pct": rec.candidacy_pct if rec else None,
                "manager_evidence_strength_pct": self.manager_score,
            },
            "bands": {
                "ats_score": band(self.ats_score),
                "hr_screen_criteria_met_pct": band(self.recruiter_score),
                "manager_evidence_strength_pct": band(self.manager_score),
            },
            "disclaimer": (
                "These are percentages of things MEASURED, not probabilities of passing. "
                "'ats_score' = weighted machine-filter criteria met; "
                "'hr_screen_criteria_met_pct' = share of the JD's stated screening criteria you "
                "meet; 'manager_evidence_strength_pct' = rubric score for how well your evidence "
                "holds up. None of them predict selection, which depends on the rest of the "
                "applicant pool and on factors absent from both documents. Real passing rates "
                "require outcome data — see the application log."
            ),
            "ats_layer": {
                "score": self.ats_score,
                "components": self.ats_components,
                "weights": self.ats_weights,
                "formatting": {
                    "file_type": self.parse_result.file_type,
                    "word_count": self.parse_result.word_count,
                    "sections_found": self.parse_result.sections_found,
                    "sections_missing": self.parse_result.sections_missing,
                    "has_email": self.parse_result.has_email,
                    "has_phone": self.parse_result.has_phone,
                    "has_links": self.parse_result.has_links,
                    "multi_column_risk": self.parse_result.multi_column_risk,
                    "has_tables": self.parse_result.has_tables,
                    "warnings": self.parse_result.warnings,
                },
                "keywords": {
                    "matched": self.keyword_result.matched_terms,
                    "missing": self.keyword_result.missing_terms,
                },
                "semantic": {
                    "available": self.semantic_result.available,
                    "score": self.semantic_result.semantic_score,
                    "model": self.semantic_result.model,
                    "strengths": self.semantic_result.strengths,
                    "gaps": self.semantic_result.gaps,
                    "reworded_matches": self.semantic_result.reworded_matches,
                    "recommendation": self.semantic_result.recommendation,
                    "error": self.semantic_result.error,
                },
            },
            "recruiter_layer": self.recruiter_result.to_dict() if self.recruiter_result else None,
            "manager_layer": self.manager_result.to_dict() if self.manager_result else None,
            "jd_requirements": {
                "min_years": self.jd_reqs.min_years,
                "min_degree": self.jd_reqs.min_degree,
                "degree_mandatory": self.jd_reqs.degree_mandatory,
                "certifications": self.jd_reqs.certifications,
                "work_modes": self.jd_reqs.work_modes,
                "sponsorship_unavailable": self.jd_reqs.sponsorship_unavailable,
                "seniority": self.jd_reqs.seniority,
                "salary_min": self.jd_reqs.salary_min,
                "salary_max": self.jd_reqs.salary_max,
                "salary_currency": self.jd_reqs.salary_currency,
                "jd_location": self.jd_reqs.jd_location,
                "countries": self.jd_reqs.countries,
                "evidence": [
                    {"kind": r.kind, "value": r.value, "source_line": r.source_line,
                     "mandatory": r.mandatory}
                    for r in self.jd_reqs.evidence
                ],
            },
            "notes": self.notes,
        }


def run_full_check(
    resume_path: str | None = None,
    resume_text: str | None = None,
    jd_text: str = "",
    profile: CandidateProfile | None = None,
    model: str = ollama_client.DEFAULT_MODEL,
    host: str = ollama_client.DEFAULT_HOST,
    api_key: str = ollama_client.DEFAULT_API_KEY,
    provider: str | None = None,
    skip_semantic: bool = False,
    skip_manager: bool = False,
) -> FullReport:
    if not jd_text.strip():
        raise ValueError("Job description text is required")

    notes: list[str] = []

    # ---- shared parsing
    parse_result = parsing.analyze(path=resume_path, text=resume_text)
    resume_body = parse_result.text

    # ---- Layer 1: ATS
    jd_keywords = kw_mod.extract_jd_keywords(jd_text)
    keyword_result = kw_mod.score_keywords(resume_body, jd_keywords)

    if skip_semantic:
        semantic_result = sem_mod.SemanticResult(
            available=False, error="Semantic scoring skipped by request."
        )
    else:
        semantic_result = sem_mod.score_semantic(
            resume_text=resume_body, jd_text=jd_text, model=model, host=host,
            api_key=api_key, provider=provider,
        )
        if not semantic_result.available:
            notes.append(
                f"ATS semantic layer unavailable ({semantic_result.error}). "
                "Falling back to keyword + formatting only."
            )

    weights = ATS_WEIGHTS if semantic_result.available else ATS_WEIGHTS_NO_LLM
    total_w = sum(weights.values()) or 1.0
    weights = {k: v / total_w for k, v in weights.items()}

    formatting_score = parse_result.formatting_score()
    sem_score = semantic_result.semantic_score if semantic_result.available else 0
    ats_score = round(
        formatting_score * weights["formatting"]
        + keyword_result.score * weights["keyword"]
        + sem_score * weights["semantic"],
        1,
    )

    # ---- Layer 2: Recruiter screen
    jd_reqs = jd_mod.extract(jd_text)
    prof = profile or CandidateProfile()
    if prof.is_empty:
        notes.append(
            "No profile.yaml found (or it's empty) — recruiter checks that need your "
            "fixed facts were skipped rather than guessed. Run `init-profile` to set it up."
        )
    must_have_terms = [k.term for k in jd_keywords[:10]]
    recruiter_result = rec_mod.score_recruiter_screen(
        resume_text=resume_body, profile=prof, reqs=jd_reqs, must_have_terms=must_have_terms
    )

    # ---- Layer 3: Manager evidence
    if skip_manager:
        manager_result = mgr_mod.ManagerResult(
            available=False, error="Manager review skipped by request."
        )
    else:
        manager_result = mgr_mod.score_manager_review(
            resume_text=resume_body, jd_text=jd_text, model=model, host=host,
            api_key=api_key, provider=provider,
        )
        if not manager_result.available:
            notes.append(f"Manager evidence layer unavailable ({manager_result.error}).")

    return FullReport(
        ats_score=ats_score,
        ats_components={
            "formatting": formatting_score,
            "keyword_match": keyword_result.score,
            "semantic_fit": semantic_result.semantic_score,
        },
        ats_weights={k: round(v, 3) for k, v in weights.items()},
        recruiter_score=recruiter_result.score,
        manager_score=manager_result.score if manager_result.available else None,
        parse_result=parse_result,
        keyword_result=keyword_result,
        semantic_result=semantic_result,
        recruiter_result=recruiter_result,
        manager_result=manager_result,
        jd_reqs=jd_reqs,
        notes=notes,
    )
