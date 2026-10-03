"""The tailor pipeline: select, assemble, gate, verified LLM passes, score.

One pass, in this order (`tailor()`):

  1. Load the bank and keep only bullets the user has confirmed. Unreviewed
     (LLM-transcribed, unconfirmed) bullets are left out and counted in
     bank_problems; if none are confirmed, it refuses.
  2. Deterministic selection + assembly (instant, offline):
     - evidence selection: greedy marginal coverage of the JD's weighted
       terms under a line budget (see selector.py)
     - roles newest-first; within a role, bullets in pick order, so the
       strongest evidence for this JD leads
     - skills section: JD-covered skills first, by JD weight
  3. No-apply gate: candidacy-fact blockers (years floor, sponsorship, work
     mode, mandatory degree) are facts about you, not the resume. It runs
     on the assembled draft BEFORE any LLM call; with blockers and no
     --force the result comes back `blocked` (callers don't show the text).
  4. Baseline: the unchanged scorer, offline, records the keyword match and
     flags a too-thin or over-budget selection.
  5. LLM pass 1 (skipped offline), run once: gap rewording surfaces the JD's
     phrasing for missing high-weight terms that selected bullets already
     demonstrate.
  6. LLM pass 2 (skipped offline), run once: the manager layer's weak-bullet
     rewrites, checked by the same truth rule but only SUGGESTED (in
     result.suggestions), never applied: they can change what a bullet
     claims, and only the user can vouch for that.
  7. Final score with the full scorer (LLM layers too, unless offline),
     reported beside the baseline.

How a rewrite is accepted: the shared truth rule (rewriter.truthful_rewording),
the same one optimize uses. It must pass fact verification against its
original bullet (numbers/dates kept, none added, no named tool dropped), must
not lose a JD term the original covered, and may newly name a JD term only
if the bullet's own confirmed tags claim it. Gap rewording is only ever
asked for terms a bullet's tags claim, so a JD skill the bank lacks can't
reach the output, whatever the LLM writes.
Rewrites are NOT re-scored one by one and nothing iterates to convergence:
the guards make each accepted change coverage-neutral or better, and the
final score is measured once, after both passes. Rejected suggestions are
returned with their reason.

Honesty in output: JD terms the bank simply cannot cover are reported as
gaps, never papered over — a missing Snowflake bullet that isn't true would
score higher and collapse in the interview.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from .. import jd_requirements as jd_mod
from .. import keywords as kw_mod
from .. import llm_client
from .. import manager as mgr_mod
from .. import recruiter as rec_mod
from ..profile import CandidateProfile
from ..scorer import FullReport, run_full_check
from .assembler import assemble
from .evidence_bank import EvidenceBank, load_bank
from .rewriter import ascii_safe, reword_for_terms, truthful_rewording
from .selector import ChosenBullet, Selection, build_jd_map, select, text_covers


@dataclass
class TailorResult:
    resume_text: str
    selection: Selection
    report: FullReport | None = None        # final three-layer score of the output
    baseline_keyword_pct: float | None = None   # deterministic assembly, before LLM moves
    final_keyword_pct: float | None = None
    rewordings: list[dict] = field(default_factory=list)   # applied, verified
    # manager-layer rewrites that passed the checks but were NOT applied:
    # they can change what a bullet claims, so they wait for the user's yes
    suggestions: list[dict] = field(default_factory=list)
    rejected_rewrites: list[dict] = field(default_factory=list)  # failed verification
    honest_gaps: list[str] = field(default_factory=list)   # JD terms the bank can't cover
    candidacy_blockers: list[str] = field(default_factory=list)
    bank_problems: list[str] = field(default_factory=list)
    used_llm: bool = False
    blocked: bool = False
    notes: list[str] = field(default_factory=list)


def _candidacy_blockers(
    resume_text: str, profile: CandidateProfile, reqs: jd_mod.JDRequirements,
    jd_keywords: list[kw_mod.JDKeyword],
) -> list[str]:
    """Hard fails on facts about the candidate — the no-apply gate. Resume-
    fixable fails are deliberately excluded: those are ours to fix."""
    rec = rec_mod.score_recruiter_screen(
        resume_text=resume_text, profile=profile, reqs=reqs,
        must_have_terms=[k.term for k in jd_keywords[:10]],
    )
    return [
        c.detail for c in rec.checks
        if c.category == rec_mod.CANDIDACY and c.status == "fail" and c.importance >= rec_mod.GATE
    ]


def _match_weak_bullet(quoted: str, chosen: list[ChosenBullet]) -> ChosenBullet | None:
    """The manager layer quotes a bullet; find the atom it refers to.

    Strict on purpose: a loose match attaches the manager's rewrite to the
    WRONG bullet and silently rewords a different line than the one that
    was criticised. Containment wins; otherwise a 0.75 similarity floor —
    below that we'd rather skip the suggestion than risk a mismatch.
    """
    q = " ".join((quoted or "").lower().split())
    if not q:
        return None
    best, best_ratio = None, 0.0
    for cb in chosen:
        t = " ".join(cb.text.lower().split())
        if q in t or t in q:
            return cb
        ratio = difflib.SequenceMatcher(None, q, t[: len(q) + 60]).ratio()
        if ratio > best_ratio:
            best, best_ratio = cb, ratio
    return best if best_ratio >= 0.75 else None


def tailor(
    master_path: str,
    jd_text: str,
    profile: CandidateProfile | None = None,
    offline: bool = False,
    force: bool = False,
    max_current: int = 5,
    max_other: int = 3,
    max_lines: int = 26,
    model: str = llm_client.DEFAULT_MODEL,
    host: str = llm_client.DEFAULT_HOST,
    api_key: str = llm_client.DEFAULT_API_KEY,
    provider: str | None = None,
) -> TailorResult:
    if not jd_text.strip():
        raise ValueError("Job description text is required")

    bank, unreviewed = load_bank(master_path).reviewed_only()
    if unreviewed and bank.is_empty:
        raise ValueError(
            f"All {unreviewed} bullets in the evidence bank are still unreviewed (LLM-transcribed, "
            "not yet confirmed by you). Confirm the true ones in the web UI's Setup tab, or "
            "remove their `reviewed: false` lines in master_resume.yaml — the tailor only "
            "uses bullets you have vouched for."
        )
    bank_problems = bank.validate()
    if unreviewed:
        bank_problems.insert(0, f"{unreviewed} unreviewed bullet(s) left out — confirm them in "
                                "the Setup tab (or drop `reviewed: false`) to make them usable.")
    prof = profile or CandidateProfile()

    jd_keywords = kw_mod.extract_jd_keywords(jd_text)
    jd_reqs = jd_mod.extract(jd_text)
    jd_map, _reps = build_jd_map(jd_keywords)

    selection = select(bank, jd_keywords, max_current=max_current, max_other=max_other,
                       max_lines=max_lines)
    resume_text = assemble(bank, selection, jd_keywords)

    # ---------------- no-apply gate ----------------
    blockers = _candidacy_blockers(resume_text, prof, jd_reqs, jd_keywords)
    honest_gaps = [k.term for k in selection.uncovered]

    result = TailorResult(
        resume_text=resume_text, selection=selection,
        honest_gaps=honest_gaps, candidacy_blockers=blockers,
        bank_problems=bank_problems,
    )
    if blockers and not force:
        result.blocked = True
        result.notes.append(
            "Candidacy-fact blockers found — no rewrite fixes these. Re-run with --force "
            "to generate anyway (for practice, or because you think the gate is wrong)."
        )
        return result

    # ---------------- baseline: deterministic assembly, scored offline ----------------
    def offline_score(text: str) -> FullReport:
        return run_full_check(
            resume_text=text, jd_text=jd_text, profile=prof,
            skip_semantic=True, skip_manager=True,
        )

    base = offline_score(resume_text)
    result.baseline_keyword_pct = base.keyword_result.score

    rewordings: list[dict] = []
    rejected: list[dict] = []
    notes: list[str] = []

    # A too-short output isn't a formatting failure, it's a thin bank — and
    # only the user can fix that, by writing more real bullets.
    if base.parse_result.word_count < 100:
        notes.append(
            f"Assembled resume is only {base.parse_result.word_count} words — real one-pagers run "
            "400+. Your evidence bank is too thin; add more real bullets to master_resume.yaml "
            "(the scorer flags low word counts as a parse-failure risk too)."
        )

    # A too-LONG output is now visible too: the structure minimum (every
    # role keeps at least one bullet) can exceed the line budget on a long
    # career history. That's a trade-off to surface, not hide.
    if selection.total_lines > max_lines:
        notes.append(
            f"Selection used {selection.total_lines} estimated lines against a budget of "
            f"{max_lines} — the every-role-needs-bullets structure minimum pushed past it. "
            "Raise --max-lines, or trim earlier roles in the bank."
        )

    # ---------------- LLM pass 1: surface missing terms honestly ----------------
    if not offline:
        missing = [k.term for k in base.keyword_result.missing if k.weight >= 2.0][:6]
        if missing and selection.chosen:
            rewrites = reword_for_terms(
                selection.chosen, missing,
                model=model, host=host, api_key=api_key, provider=provider,
            )
            applied_any = False
            for r in rewrites:
                if r.index < 0:
                    notes.append(f"Gap-rewording skipped: {r.note}")
                    continue
                cb = selection.chosen[r.index]
                if not r.verified:
                    if r.rewrite:
                        rejected.append({"original": r.original, "suggested": r.rewrite, "reason": r.note})
                    continue
                # the shared truth rule: coverage guard + fabrication guard (a
                # newly named JD term must be one this bullet's tags claim)
                ok, reason = truthful_rewording(cb, r.rewrite, jd_map)
                if not ok:
                    rejected.append({"original": cb.text, "suggested": r.rewrite, "reason": reason})
                    continue
                rewordings.append({
                    "original": cb.text, "rewrite": r.rewrite,
                    "reason": f"surfaces the JD's phrasing for: {', '.join(missing)}",
                    "source": "gap-rewording",
                })
                cb.text = r.rewrite
                applied_any = True
            if applied_any:
                resume_text = assemble(bank, selection, jd_keywords)
                notes.append("Gap rewording applied — selected bullets now use the JD's phrasing.")

    # ---------------- LLM pass 2: manager weak-bullet rewrites ----------------
    if not offline:
        mgr = mgr_mod.score_manager_review(
            resume_text=resume_text, jd_text=jd_text,
            model=model, host=host, api_key=api_key, provider=provider,
        )
        if mgr.available and mgr.weak_bullets:
            for wb in mgr.weak_bullets:
                cb = _match_weak_bullet(wb.get("bullet", ""), selection.chosen)
                if cb is None:
                    continue
                suggestion = ascii_safe((wb.get("rewrite") or "").strip())
                if not suggestion:
                    continue
                # A manager rewrite may strengthen wording, but the only new
                # skill terms it may introduce are ones the ORIGINAL bullet
                # already covered — swapping Power BI for Tableau here would
                # be invention, and the verifier now catches it structurally.
                ok, reason = truthful_rewording(cb, suggestion, jd_map,
                                                allowed_extra=text_covers(cb.text, jd_map))
                if not ok:
                    rejected.append({"original": cb.text, "suggested": suggestion, "reason": reason})
                    continue
                # The checks above catch new numbers, tools and JD terms, but
                # not an invented outcome ("...that transformed pricing
                # strategy"). So a manager rewrite is only SUGGESTED: it never
                # goes into the resume without the user's explicit yes.
                result.suggestions.append({
                    "original": cb.text, "rewrite": suggestion,
                    "reason": wb.get("problem", "manager-layer weak bullet"),
                    "source": "manager-rewrite", "needs_review": True,
                })
            if result.suggestions:
                notes.append(f"{len(result.suggestions)} manager-layer rewrite(s) suggested but NOT "
                             "applied: they can change what a bullet claims. Use the ones that are "
                             "true by editing that bullet in your evidence bank (or run `optimize`, "
                             "which asks you about each one).")
        elif not mgr.available:
            notes.append(f"Manager layer unavailable ({mgr.error}) — weak-bullet rewrites skipped.")

    # ---------------- final score ----------------
    final = run_full_check(
        resume_text=resume_text, jd_text=jd_text, profile=prof,
        skip_semantic=offline, skip_manager=offline,
        model=model, host=host, api_key=api_key, provider=provider,
    )
    # Gaps are what the FINAL resume still doesn't cover — rewording may have
    # closed some of the assembly-time gaps, so don't report stale ones.
    result.honest_gaps = [k.term for k in final.keyword_result.missing]
    result.resume_text = resume_text
    result.report = final
    result.final_keyword_pct = final.keyword_result.score
    result.rewordings = rewordings
    result.rejected_rewrites = rejected
    result.used_llm = bool(rewordings) or (not offline and final.semantic_result.available)
    result.notes = notes + result.notes
    return result