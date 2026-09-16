"""Constrained LLM rewording — the only step where the model touches wording,
and it never does so unsupervised.

The highest-value honest fix the scorer flags is 'experience you have but
word differently than the JD': the bullet demonstrates the skill but uses
different phrasing. Rewording surfaces the JD's term for what you already
did — that's translation, not invention, and it's what this module does.

The fact-preservation verifier is the safety gate. Before any rewrite is
accepted:
  - every number in the original must survive verbatim ("30+", "12h/week",
    "2024" — a rewrite that drops or changes a number is a lie),
  - no new number may appear (a number the original didn't have is an
    invention; [X]-style placeholders are allowed because they contain no
    digits — they mark a spot for YOUR real figure),
  - every named skill/tool in the original must survive (canonical
    comparison: "PowerBI" satisfies "Power BI", but a rewrite that drops
    or swaps a named tool — Power BI out, Tableau in — is rejected, no
    matter how good it sounds),
  - when the caller passes an allowed-additions set, any NEW skill term
    the rewrite introduces must be one of the explicitly allowed target
    terms — the LLM may surface JD phrasing it was asked about, and
    nothing else it decided to name,
  - length stays within sane bounds (a rewrite that balloons a bullet is
    fluff, and one that truncates it lost detail).

A rewrite that fails any check is discarded and the original kept — the
loop then reports it as an unapplied suggestion rather than silently
shipping it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .. import llm_client
from ..keywords import skill_tokens
from ..terms import alias_normalize, canonical
from .selector import ChosenBullet

# digits, with optional thousands separators, decimals, and a trailing % or +
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?[%+]?")
PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]")

# ALL-CAPS 2-letter tokens that are ordinary words, not skills ("US-based",
# "PM hours") — excluded from the must-keep skill set to avoid false gates.
_NOT_SKILL_ACRONYMS = {"us", "uk", "eu", "am", "pm", "ok", "no", "id", "go"}

REWORD_SYSTEM_PROMPT = """You reword resume bullets so the skills they demonstrate are named the way
a job description names them. You are a translator, NOT a writer.

STRICT RULES — violating any makes the output useless:
1. NEVER add facts, numbers, dates, tools, employers, or outcomes that are
   not already in the bullet. Do not invent scale or impact.
2. Keep EVERY number from the original exactly as it appears (30+ stays 30+).
3. Keep EVERY tool, technology, and skill the original names. Never swap
   one tool for another, even a related one.
4. Each bullet lists its own candidate TARGET TERMS. Surface a target term
   only if the bullet already, genuinely demonstrates it under different
   words. If it does not, return an empty string for that bullet — a
   stretch here becomes an interview question the candidate can't answer.
5. Stay within roughly the original length (0.8x-1.5x).

Respond with STRICT JSON ONLY, no markdown fences:
{"rewrites": [{"index": <the bullet's index>, "rewrite": "<reworded bullet or empty string>"}]}
Include an entry only for bullets you can honestly reword."""


@dataclass
class RewriteResult:
    index: int
    original: str
    rewrite: str
    verified: bool
    note: str = ""


# LLMs love smart typography; ATS parsers love plain ASCII.
_ASCII_MAP = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    " ": " ", "…": "...", "−": "-",
}


def ascii_safe(text: str) -> str:
    for src, dst in _ASCII_MAP.items():
        text = text.replace(src, dst)
    return text


def extract_numbers(text: str) -> list[str]:
    """All numeric tokens, comma-normalised so '1,200' == '1200'."""
    return sorted(n.replace(",", "") for n in NUMBER_RE.findall(text or ""))


def strip_placeholders(text: str) -> str:
    """Remove [X]-style placeholders so their contents can't hide numbers."""
    return PLACEHOLDER_RE.sub("", text or "")


def verify_rewrite(
    original: str,
    rewrite: str,
    allowed_additions: set[str] | None = None,
) -> tuple[bool, str]:
    """Fact-preservation check. Returns (ok, reason-if-not).

    `allowed_additions` is the set of skill terms the rewrite is permitted
    to INTRODUCE (the JD terms this rewording was asked to surface,
    canonical form). None means unrestricted additions — callers who care
    about the honesty guarantee always pass the explicit set.
    """
    if not rewrite or not rewrite.strip():
        return False, "empty rewrite"
    if rewrite.strip().lower() == original.strip().lower():
        return False, "no change"

    ratio = len(rewrite.strip()) / max(1, len(original.strip()))
    if ratio < 0.4:
        return False, f"rewrite truncates the bullet ({ratio:.0%} of original length)"
    if ratio > 3.0:
        return False, f"rewrite balloons the bullet ({ratio:.0%} of original length)"

    orig_nums = set(extract_numbers(strip_placeholders(original)))
    new_nums = set(extract_numbers(strip_placeholders(rewrite)))

    lost = orig_nums - new_nums
    if lost:
        return False, f"lost number(s) {sorted(lost)} — a rewrite may never drop a figure"

    added = new_nums - orig_nums
    if added:
        return False, f"added number(s) {sorted(added)} — new figures are inventions; use [X] placeholders"

    # ---- skill preservation: every named tool/skill must survive, and any
    # new one must be an explicitly allowed target term.
    orig_skills = skill_tokens(original) - _NOT_SKILL_ACRONYMS
    new_skills = skill_tokens(rewrite) - _NOT_SKILL_ACRONYMS

    lost_skills = orig_skills - new_skills
    if lost_skills:
        return False, (
            f"lost named skill(s) {sorted(lost_skills)} — a rewrite may never "
            "drop or swap a tool the original names"
        )

    if allowed_additions is not None:
        allowed = {canonical(a) for a in allowed_additions}
        introduced = new_skills - orig_skills - allowed
        if introduced:
            return False, (
                f"introduced skill(s) {sorted(introduced)} the original didn't name "
                "and that aren't allowed target terms — that's invention, not rewording"
            )

    return True, ""


# ---------------------------------------------------- per-bullet targeting

def _content_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", alias_normalize(text.lower()).replace("-", " ")))


def plausible_targets(cb: ChosenBullet, missing_terms: list[str], per_bullet_cap: int = 4) -> list[str]:
    """Rank the missing JD terms by how plausibly THIS bullet already
    demonstrates them, and return the top few as its rewording targets.

    Ranking signal: an exact tag match is the strongest (the user asserted
    this bullet demonstrates that skill); shared words between the term and
    the bullet's tags/text are weaker evidence. Bullets with no signal at
    all fall back to the globally strongest missing terms — the LLM still
    judges 'demonstrates'; this only focuses its attention and keeps the
    prompt small instead of sending every bullet the full list.
    """
    if not missing_terms:
        return []
    tags = {canonical(s) for s in cb.bullet.skills}
    tag_words: set[str] = set()
    for t in tags:
        tag_words |= _content_words(t)
    text_words = _content_words(cb.text)

    def score(term: str) -> int:
        c = canonical(term)
        if c in tags:
            return 100
        term_words = _content_words(c)
        s = 0
        if term_words & tag_words:
            s += 2
        if term_words & text_words:
            s += 1
        return s

    ranked = sorted(missing_terms, key=lambda t: (-score(t), missing_terms.index(t)))
    top = ranked[:per_bullet_cap]
    if score(top[0]) > 0:
        return top
    # no signal at all — fall back to the strongest missing terms globally
    return missing_terms[:per_bullet_cap]


def reword_for_terms(
    chosen: list[ChosenBullet],
    target_terms: list[str],
    model: str | None = None,
    host: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> list[RewriteResult]:
    """One batched LLM call: reword selected bullets to surface the JD's
    phrasing for skills the bullets already demonstrate. Each bullet gets
    only its own plausible target terms — not the full list — and every
    returned rewrite is fact-verified (numbers, preserved skills, allowed
    additions) before it leaves this function."""
    if not chosen or not target_terms:
        return []

    items = []
    per_bullet_allowed: dict[int, set[str]] = {}
    for i, cb in enumerate(chosen):
        targets = plausible_targets(cb, target_terms)
        per_bullet_allowed[i] = {canonical(t) for t in targets}
        items.append({"index": i, "bullet": cb.text, "targets": targets})

    parsed, error = llm_client.call_json(
        REWORD_SYSTEM_PROMPT,
        "BULLETS TO REWORD (as JSON):\n"
        + json.dumps({"bullets": items})
        + "\n\nEach bullet's 'targets' are the only JD terms it may surface.",
        model=model, host=host, api_key=api_key, provider=provider,
        temperature=0.1,
    )
    if error or not isinstance(parsed, dict):
        return [RewriteResult(-1, "", "", False, f"rewording call failed: {error or 'unexpected payload'}")]

    results: list[RewriteResult] = []
    by_index = {}
    for item in parsed.get("rewrites", []) or []:
        if isinstance(item, dict):
            try:
                by_index[int(item.get("index"))] = str(item.get("rewrite") or "")
            except (TypeError, ValueError):
                continue

    for i, cb in enumerate(chosen):
        rewrite = by_index.get(i)
        if rewrite is None:
            continue
        rewrite = ascii_safe(rewrite)
        ok, reason = verify_rewrite(cb.text, rewrite, allowed_additions=per_bullet_allowed.get(i))
        if ok:
            results.append(RewriteResult(i, cb.text, rewrite.strip(), True))
        else:
            results.append(RewriteResult(i, cb.text, rewrite.strip(), False, reason))
    return results