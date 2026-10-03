"""`optimize`: an honest improve-and-rescore loop around the tailor.

`tailor()` makes one pass. `optimize()` keeps going: score the draft, look
at what's missing, try the next TRUTHFUL change, rescore, and keep only what
helps. It stops on a plateau (two rounds in a row gaining under 1 point),
after max_rounds, or when no truthful change is left. Then it says plainly
which gaps only real experience can close.

The score it climbs is offline and deterministic (see `objective()`): a
weighted mean of Search Visibility, the JD keyword match and the HR screen's
resume-fixable checks. None of these rise when a term is repeated, so
keyword stuffing can't raise it.

Truth rules (each one is enforced in code, not just promised):
  - Only CONFIRMED evidence-bank bullets are used (unreviewed ones are left
    out, exactly as in tailor()). Selection moves swap or add real bullets.
  - A rewording may surface a JD term only if THIS bullet's own confirmed
    skill tags already claim it, and it must pass verify_rewrite (numbers
    and dates kept, none added, no named tool dropped or swapped) and the
    coverage guard (it can't lose a JD term the original covered).
  - Fabrication guard: before a rewording is kept, any JD term it newly
    covers must be one of the bullet's own tags. A JD skill the bank lacks
    therefore can never appear; it stays in the gaps list.
  - Rewordings come back as PROPOSALS. finalize() writes only the ones the
    user accepted (auto_accept=True accepts all, and is the riskier mode).
  - The no-apply gate still blocks: candidacy-fact blockers stop the run
    unless force=True.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import jd_requirements as jd_mod
from .. import keywords as kw_mod
from .. import llm_client
from .. import manual_llm
from .. import manager as mgr_mod
from ..profile import CandidateProfile
from ..scorer import FullReport, run_full_check
from ..terms import canonical
from .assembler import assemble
from .evidence_bank import EvidenceBank, load_bank
from .optimizer import _candidacy_blockers, _match_weak_bullet
from .rewriter import ascii_safe, reword_for_terms, tag_terms, truthful_rewording
from .selector import ChosenBullet, Selection, build_jd_map, lines_cost, select, text_covers

# The objective: what the loop climbs. Weights sum to 1; a missing component
# (e.g. no searches could be built for this JD) is left out and the rest
# renormalised.
OBJECTIVE_WEIGHTS = {"search_visibility": 0.45, "keyword_match": 0.35, "hr_resume_fixable": 0.20}
PLATEAU_GAIN = 1.0          # a round gaining less than this counts toward the plateau
PLATEAU_ROUNDS = 2          # ...and this many such rounds in a row stop the loop
MAX_EVALS_PER_ROUND = 400   # bound on candidate selections scored per round


# ------------------------------------------------------------ results

@dataclass
class Proposal:
    """A verified rewording waiting for the user's yes/no."""
    id: int
    company: str
    title: str
    original: str
    rewrite: str
    reason: str
    source: str                     # gap-rewording | manager-rewrite
    accepted: bool | None = None    # None = not decided yet
    bullet_key: int = field(default=0, repr=False)   # id() of the bank bullet it rewords

    @property
    def needs_review(self) -> bool:
        """Manager rewrites change how a bullet reads, not just which JD word
        it uses, so they always need an explicit yes (never --auto-accept)."""
        return self.source == "manager-rewrite"

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("id", "company", "title", "original", "rewrite", "reason", "source", "accepted")}
        d["needs_review"] = self.needs_review
        return d


@dataclass
class RoundLog:
    round: int
    score: float
    components: dict
    gain: float
    changes: list[dict] = field(default_factory=list)   # {"what": ..., "why": ...}


@dataclass
class TruthGap:
    term: str
    weight: float
    required: bool
    status: str        # nothing-in-bank | unconfirmed | not-shown
    detail: str

    def to_dict(self) -> dict:
        return {"term": self.term, "weight": self.weight, "required": self.required,
                "status": self.status, "detail": self.detail}


@dataclass
class OptimizeResult:
    blocked: bool = False
    candidacy_blockers: list[str] = field(default_factory=list)
    bank_problems: list[str] = field(default_factory=list)
    rounds: list[RoundLog] = field(default_factory=list)
    stop_reason: str = ""
    proposals: list[Proposal] = field(default_factory=list)
    gaps: list[TruthGap] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pending_ids: list[str] = field(default_factory=list)   # copy-paste prompts still unanswered
    # set by finalize(): the resume with only the accepted proposals applied
    resume_text: str = ""
    report: FullReport | None = None
    final_score: float | None = None
    # internal state finalize() needs
    _bank: EvidenceBank | None = None
    _selection: Selection | None = None
    _jd_keywords: list = field(default_factory=list)
    _jd_text: str = ""
    _profile: CandidateProfile | None = None

    @property
    def baseline_score(self) -> float | None:
        return self.rounds[0].score if self.rounds else None

    @property
    def best_score(self) -> float | None:
        return max((r.score for r in self.rounds), default=None)

    def to_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "candidacy_blockers": self.candidacy_blockers,
            "bank_problems": self.bank_problems,
            "rounds": [{"round": r.round, "score": r.score, "components": r.components,
                        "gain": r.gain, "changes": r.changes} for r in self.rounds],
            "stop_reason": self.stop_reason,
            "baseline_score": self.baseline_score,
            "best_score": self.best_score,
            "final_score": self.final_score,
            "proposals": [p.to_dict() for p in self.proposals],
            "pending_ids": self.pending_ids,
            "gaps_truth_cannot_close": [g.to_dict() for g in self.gaps],
            "notes": self.notes,
            "resume_text": self.resume_text,
            "disclaimer": ("Scores are percentages of things measured offline, not a chance of "
                           "selection. The loop only made changes your evidence bank supports."),
        }


# ------------------------------------------------------------ scoring

def objective(rep: FullReport) -> tuple[float, dict]:
    """(score 0-100, components). Offline, deterministic, stuffing-proof."""
    rec = rep.recruiter_result
    comps = {
        "search_visibility": rep.visibility.score if rep.visibility else None,
        "keyword_match": rep.keyword_result.score,
        "hr_resume_fixable": (rec.resume_pct if rec and rec.resume_pct is not None
                              else (rec.criteria_met_pct if rec else None)),
    }
    present = {k: v for k, v in comps.items() if v is not None}
    total_w = sum(OBJECTIVE_WEIGHTS[k] for k in present) or 1.0
    score = sum(OBJECTIVE_WEIGHTS[k] * v for k, v in present.items()) / total_w
    comps["parse_safe"] = rep.visibility.parse_safe if rep.visibility else None
    return round(score, 2), comps


class _Scorer:
    def __init__(self, jd_text: str, profile: CandidateProfile):
        self.jd_text, self.profile = jd_text, profile
        self._memo: dict[str, tuple[float, dict, FullReport]] = {}

    def __call__(self, text: str) -> tuple[float, dict, FullReport]:
        hit = self._memo.get(text)
        if hit is None:
            rep = run_full_check(resume_text=text, jd_text=self.jd_text, profile=self.profile,
                                 skip_semantic=True, skip_manager=True)
            s, comps = objective(rep)
            hit = (s, comps, rep)
            self._memo[text] = hit
        return hit


# ------------------------------------------------------------ selection state

def _rebuild(chosen: list[ChosenBullet], bank: EvidenceBank, jd_map: dict, reps: dict) -> Selection:
    """A Selection for a modified chosen-list, its coverage recomputed from
    the CURRENT texts and tags (what the assembler and gap report read)."""
    covered_by_bullets: set[str] = set()
    for cb in chosen:
        cb.covered = text_covers(cb.text, jd_map) | {canonical(s) for s in cb.bullet.skills
                                                     if canonical(s) in jd_map}
        covered_by_bullets |= cb.covered
    skills_covered = {canonical(s) for s in bank.all_skills()} & set(jd_map)
    covered = covered_by_bullets | skills_covered
    uncovered = sorted((reps[t] for t in set(jd_map) - covered), key=lambda k: k.weight, reverse=True)
    return Selection(chosen=chosen, covered=covered, covered_by_bullets=covered_by_bullets,
                     uncovered=uncovered, covered_by_skills_section=skills_covered,
                     total_lines=sum(lines_cost(cb.bullet) for cb in chosen))


def _selection_moves(bank: EvidenceBank, chosen: list[ChosenBullet], max_lines: int,
                     max_current: int, max_other: int) -> list[tuple[list[ChosenBullet], dict]]:
    """Every truthful selection change: swap a chosen bullet for an unused
    confirmed bullet of the SAME role, or add one where the role cap and the
    line budget allow. Role structure (bullets per role) is never reduced."""
    used = {id(cb.bullet) for cb in chosen}
    lines = sum(lines_cost(cb.bullet) for cb in chosen)
    moves = []
    for role in bank.roles:
        idx = [i for i, cb in enumerate(chosen) if cb.role is role]
        unused = [b for b in role.bullets if id(b) not in used]
        if not unused:
            continue
        for b in unused:
            for i in idx:          # swap
                old = chosen[i]
                if lines - lines_cost(old.bullet) + lines_cost(b) > max_lines:
                    continue
                new = list(chosen)
                new[i] = ChosenBullet(bullet=b, role=role, text=b.text)
                moves.append((new, {"what": f"swap bullet in {role.company}",
                                    "out": old.text, "in": b.text}))
            cap = max_current if role.is_current else max_other
            if idx and len(idx) < cap and lines + lines_cost(b) <= max_lines:   # add
                new = list(chosen)
                new.insert(idx[-1] + 1, ChosenBullet(bullet=b, role=role, text=b.text))
                moves.append((new, {"what": f"add bullet to {role.company}", "in": b.text}))
    return moves


def _copy_chosen(chosen: list[ChosenBullet]) -> list[ChosenBullet]:
    return [ChosenBullet(bullet=cb.bullet, role=cb.role, text=cb.text, covered=set(cb.covered))
            for cb in chosen]


# ------------------------------------------------------------ truth checks

def _tag_terms(cb: ChosenBullet) -> set[str]:
    return tag_terms(cb)


def _rewrite_is_truthful(cb: ChosenBullet, rewrite: str, jd_map: dict,
                         allowed_extra: set[str] | None = None) -> tuple[bool, str]:
    """The shared truth rule (rewriter.truthful_rewording): verify_rewrite +
    coverage guard + fabrication guard, against the bullet's bank original."""
    return truthful_rewording(cb, rewrite, jd_map, allowed_extra)


# ------------------------------------------------------------ the loop

def optimize(
    master_path: str,
    jd_text: str,
    profile: CandidateProfile | None = None,
    offline: bool = False,
    force: bool = False,
    max_rounds: int = 5,
    max_current: int = 5,
    max_other: int = 3,
    max_lines: int = 26,
    model: str = llm_client.DEFAULT_MODEL,
    host: str = llm_client.DEFAULT_HOST,
    api_key: str = llm_client.DEFAULT_API_KEY,
    provider: str | None = None,
) -> OptimizeResult:
    if not jd_text.strip():
        raise ValueError("Job description text is required")
    prof = profile or CandidateProfile()

    full_bank = load_bank(master_path)
    bank, unreviewed = full_bank.reviewed_only()
    if unreviewed and bank.is_empty:
        raise ValueError(
            f"All {unreviewed} bullets in the evidence bank are still unreviewed. Confirm the "
            "true ones (web UI Setup tab, or remove `reviewed: false` in master_resume.yaml) — "
            "optimize only uses bullets you have vouched for.")
    res = OptimizeResult(_bank=bank, _jd_text=jd_text, _profile=prof)
    res.bank_problems = bank.validate()
    if unreviewed:
        res.bank_problems.insert(0, f"{unreviewed} unreviewed bullet(s) left out — confirm them "
                                    "to let optimize use them.")

    jd_keywords = kw_mod.extract_jd_keywords(jd_text)
    jd_reqs = jd_mod.extract(jd_text)
    jd_map, reps = build_jd_map(jd_keywords)
    res._jd_keywords = jd_keywords
    score = _Scorer(jd_text, prof)

    # ---- round 0: exactly tailor()'s deterministic draft
    sel = select(bank, jd_keywords, max_current=max_current, max_other=max_other, max_lines=max_lines)
    text = assemble(bank, sel, jd_keywords)
    res.candidacy_blockers = _candidacy_blockers(text, prof, jd_reqs, jd_keywords)
    if res.candidacy_blockers and not force:
        res.blocked = True
        res.notes.append("Candidacy-fact blockers found — no rewrite fixes these. Re-run with "
                         "--force to optimize anyway.")
        return res

    best_s, best_c, _ = score(text)
    res.rounds.append(RoundLog(0, best_s, best_c, 0.0, [{"what": "deterministic draft (= tailor)",
                                                          "why": "starting point"}]))
    chosen = sel.chosen
    rewritten: dict[int, Proposal] = {}       # id(bullet) -> its kept-in-loop proposal
    rejected_bullets: set[int] = set()        # id(bullet) whose rewording failed this run
    llm_on = not offline
    tried_reword: set[tuple] = set()
    weak_rounds = 0
    res.stop_reason = f"reached --max-rounds ({max_rounds})"
    pending_before = {p.id for p in manual_llm.pending()}

    # The manager review reads the round-0 draft, so in copy-paste mode its
    # prompt goes into the SAME first bundle as round 1's rewording prompt.
    weak_bullets: list[dict] = []
    if llm_on:
        mgr = mgr_mod.score_manager_review(resume_text=text, jd_text=jd_text, model=model,
                                           host=host, api_key=api_key, provider=provider)
        if mgr.available:
            weak_bullets = list(mgr.weak_bullets)
        elif not (mgr.error or "").startswith(manual_llm.PENDING_PREFIX):
            res.notes.append(f"Manager layer unavailable ({mgr.error}) — weak-bullet rewrites "
                             "skipped.")

    ctx = dict(bank=bank, jd_map=jd_map, reps=reps, jd_keywords=jd_keywords, score=score,
               rewritten=rewritten, rejected_bullets=rejected_bullets, res=res)

    for rnd in range(1, max_rounds + 1):
        start = best_s
        changes: list[dict] = []

        # -- 1. selection: best single truthful swap/add, repeated while it helps
        evals = 0
        while evals < MAX_EVALS_PER_ROUND:
            best_move = None
            for new, info in _selection_moves(bank, chosen, max_lines, max_current, max_other):
                if evals >= MAX_EVALS_PER_ROUND:
                    break
                evals += 1
                trial = _copy_chosen(new)
                for cb in trial:   # keep in-loop rewordings on bullets that stay
                    p = rewritten.get(id(cb.bullet))
                    if p is not None:
                        cb.text = p.rewrite
                s, c, _ = score(assemble(bank, _rebuild(trial, bank, jd_map, reps), jd_keywords))
                if s > best_s + 1e-9 and (best_move is None or s > best_move[0]):
                    best_move = (s, c, trial, info)
            if best_move is None:
                break
            best_s, best_c, chosen, info = best_move
            for p_id in [k for k in rewritten if k not in {id(cb.bullet) for cb in chosen}]:
                rewritten.pop(p_id)          # its bullet was swapped out
            info = dict(info, why=f"score {start:.1f} -> {best_s:.1f}")
            changes.append(info)

        # -- 2. rewording: say in the bullet's TEXT the JD's name for a skill
        # the bullet's own confirmed tags already claim (a recruiter reading
        # the experience section, and the visibility searches, see the text)
        if llm_on:
            fresh, targets = [], []
            for cb in chosen:
                if id(cb.bullet) in rewritten or id(cb.bullet) in rejected_bullets:
                    continue          # don't reword a rewording, or re-ask a rejected one
                unsaid = (_tag_terms(cb) & set(jd_map)) - text_covers(cb.text, jd_map)
                if unsaid:
                    fresh.append(cb)
                    targets += [reps[t].term for t in sorted(unsaid) if reps[t].term not in targets]
            key = (tuple(cb.text for cb in fresh), tuple(targets))
            if fresh and key not in tried_reword:
                tried_reword.add(key)
                for r in reword_for_terms(fresh, targets, model=model, host=host,
                                          api_key=api_key, provider=provider):
                    if r.index < 0:
                        if not (r.note or "").startswith("rewording call failed: " +
                                                         manual_llm.PENDING_PREFIX):
                            res.notes.append(f"Rewording skipped: {r.note}")
                        continue
                    if not r.rewrite:
                        continue      # the LLM declined: nothing to report
                    if not r.verified:
                        rejected_bullets.add(id(fresh[r.index].bullet))
                        res.notes.append(f"Dropped a gap-rewording of \"{_short(r.original)}\": "
                                         f"{r.note}")
                        continue
                    best_s, best_c = _try_rewrite(
                        chosen, chosen.index(fresh[r.index]), ascii_safe(r.rewrite),
                        "gap-rewording",
                        "names the JD's term for a skill this bullet's tags already claim",
                        best_s, best_c, changes, **ctx)

            # -- 3. the manager layer's weak-bullet rewrites (from the round-0
            # review; used once, in round 1)
            for wb in weak_bullets:
                suggestion = ascii_safe((wb.get("rewrite") or "").strip())
                if not suggestion:
                    continue
                cb = _match_weak_bullet(wb.get("bullet", ""), chosen)
                if cb is None:
                    res.notes.append(f"Manager suggestion skipped: \"{_short(wb.get('bullet', ''))}\" "
                                     "isn't in the resume any more (or couldn't be matched).")
                    continue
                if id(cb.bullet) in rewritten or id(cb.bullet) in rejected_bullets:
                    continue
                best_s, best_c = _try_rewrite(
                    chosen, chosen.index(cb), suggestion, "manager-rewrite",
                    wb.get("problem") or "manager-layer weak bullet",
                    best_s, best_c, changes, allowed_extra=text_covers(cb.bullet.text, jd_map),
                    **ctx)
            weak_bullets = []

        gain = round(best_s - start, 2)
        res.rounds.append(RoundLog(rnd, best_s, best_c, gain, changes))
        if not changes:
            res.stop_reason = "no truthful change left that raises the score"
            break
        weak_rounds = weak_rounds + 1 if gain < PLATEAU_GAIN else 0
        if weak_rounds >= PLATEAU_ROUNDS:
            res.stop_reason = (f"plateau: {PLATEAU_ROUNDS} rounds in a row gained under "
                               f"{PLATEAU_GAIN:g} point")
            break

    res.pending_ids = [p.id for p in manual_llm.pending() if p.id not in pending_before]
    if res.pending_ids:
        res.stop_reason = "waiting for pasted answers"
    res._selection = _rebuild(chosen, bank, jd_map, reps)
    res.proposals = list(rewritten.values())
    for i, p in enumerate(res.proposals, 1):
        p.id = i
    finalize(res, accepted_ids=None)
    res.gaps = truth_gaps(full_bank, bank, res._selection, res.report, reps)
    return res


def _short(text: str, n: int = 70) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _try_rewrite(chosen, i, rewrite, source, reason, best_s, best_c, changes, *, bank, jd_map,
                 reps, jd_keywords, score, rewritten, rejected_bullets, res, allowed_extra=None):
    """Apply one rewording if it passes the truth rule and doesn't lower the
    score; every rejection leaves a note. Returns the best score + components."""
    cb = chosen[i]
    ok, why = _rewrite_is_truthful(cb, rewrite, jd_map, allowed_extra)
    if not ok:
        rejected_bullets.add(id(cb.bullet))
        res.notes.append(f"Dropped a {source} of \"{_short(cb.bullet.text)}\": {why}")
        return best_s, best_c
    trial = _copy_chosen(chosen)
    trial[i].text = rewrite
    s, c, _ = score(assemble(bank, _rebuild(trial, bank, jd_map, reps), jd_keywords))
    if s + 1e-9 < best_s:
        rejected_bullets.add(id(cb.bullet))
        res.notes.append(f"Skipped a {source} of \"{_short(cb.bullet.text)}\" that would lower "
                         f"the score ({best_s:.1f} -> {s:.1f}).")
        return best_s, best_c
    cb.text = rewrite
    rewritten[id(cb.bullet)] = Proposal(0, cb.role.company, cb.role.title, cb.bullet.text,
                                        rewrite, reason, source, bullet_key=id(cb.bullet))
    changes.append({"what": f"{source} in {cb.role.company}", "out": cb.bullet.text, "in": rewrite,
                    "why": f"{reason}; score {best_s:.1f} -> {s:.1f}"})
    return s, c


def finalize(res: OptimizeResult, accepted_ids: set[int] | None = None,
             auto_accept: bool = False) -> OptimizeResult:
    """Write the final resume: original bank wording everywhere except the
    proposals the user accepted.

    accepted_ids: the proposal ids the user said yes to (None = not asked
    yet, so nothing is accepted). auto_accept=True also accepts every
    gap-rewording, but NEVER a manager rewrite (needs_review): those can
    change what a bullet claims, so only an explicit yes applies them.
    Proposals are matched to bullets by identity, not by text, so two
    bullets that read the same can't swap rewrites."""
    if res._selection is None:
        return res
    for p in res.proposals:
        if accepted_ids is not None and p.id in accepted_ids:
            p.accepted = True
        elif auto_accept and not p.needs_review:
            p.accepted = True
        elif accepted_ids is not None:
            p.accepted = False
        else:
            p.accepted = None
    by_bullet = {p.bullet_key: p for p in res.proposals if p.accepted}
    chosen = []
    for cb in res._selection.chosen:
        p = by_bullet.get(id(cb.bullet))
        chosen.append(ChosenBullet(bullet=cb.bullet, role=cb.role,
                                   text=p.rewrite if p else cb.bullet.text))
    jd_map, reps = build_jd_map(res._jd_keywords)
    sel = _rebuild(chosen, res._bank, jd_map, reps)
    res.resume_text = assemble(res._bank, sel, res._jd_keywords)
    res.report = run_full_check(resume_text=res.resume_text, jd_text=res._jd_text,
                                profile=res._profile, skip_semantic=True, skip_manager=True)
    res.final_score, _ = objective(res.report)
    return res


def truth_gaps(full_bank: EvidenceBank, confirmed: EvidenceBank, sel: Selection,
               report: FullReport | None, reps: dict) -> list[TruthGap]:
    """Each JD term the final resume still lacks, and why: nothing in the
    bank at all, only in an unconfirmed bullet, or in the bank but not shown."""
    if report is None:
        return []

    def bank_terms(bank: EvidenceBank, only_unreviewed: bool = False) -> set[str]:
        out: set[str] = set()
        jd_map = {canonical(k.term): k.weight for k in report.keyword_result.missing}
        for role in bank.roles:
            for b in role.bullets:
                if only_unreviewed and b.reviewed:
                    continue
                out |= text_covers(b.text, jd_map) | {canonical(s) for s in b.skills}
        if not only_unreviewed:
            for extra in bank.skills_extra + bank.certifications:
                out |= text_covers(extra, jd_map) | {canonical(extra)}
        return out

    confirmed_terms = bank_terms(confirmed)
    unconfirmed_terms = bank_terms(full_bank, only_unreviewed=True)
    gaps = []
    for k in report.keyword_result.missing:
        c = canonical(k.term)
        required = k.section in ("hard", "skills")
        if c in confirmed_terms:
            status, detail = "not-shown", ("Your bank has this, but it didn't fit the resume "
                                           "(line budget / skills-line cap). Raise --max-lines "
                                           "or make room.")
        elif c in unconfirmed_terms:
            status, detail = "unconfirmed", ("Only an UNCONFIRMED bullet mentions this. If it's "
                                             "true, confirm that bullet (Setup tab) and re-run.")
        else:
            status, detail = "nothing-in-bank", ("Nothing in your evidence bank shows this. "
                                                 "Only real experience closes it — don't add it.")
        gaps.append(TruthGap(k.term, k.weight, required, status, detail))
    order = {"nothing-in-bank": 0, "unconfirmed": 1, "not-shown": 2}
    gaps.sort(key=lambda g: (order[g.status], -g.weight))
    return gaps
