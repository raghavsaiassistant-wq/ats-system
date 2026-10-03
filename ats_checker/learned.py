"""Taxonomy growth from verified LLM extractions — with a human in the loop.

The rule engine only finds skills its taxonomy knows. When the LLM extractor
pulls a skill out of a JD, quotes the line it came from, and the quote
checks out against the JD text, but the taxonomy has never heard of the
term, the term is recorded here as a CANDIDATE with how often it was seen
and an example line.

Candidates never change the rule engine on their own: an LLM can be
confidently wrong, and one bad term in the taxonomy is a false requirement
in every JD that happens to contain the word. A term only reaches the
taxonomy after `python cli.py learned promote TERM`.

File: $ATS_HOME/learned_terms.yaml (default ~/.ats-system/learned_terms.yaml)

    approved: [gd&t, istqb]
    candidates:
      horeca: {count: 3, example: "- Experience selling into HORECA accounts"}

Set ATS_LEARNED_TERMS=off to ignore the file entirely (the corpus tests do,
so a developer's approved terms can't move the CI baseline).
"""
from __future__ import annotations

import os
from pathlib import Path


def ats_home() -> Path:
    return Path(os.environ.get("ATS_HOME") or Path.home() / ".ats-system")


def learned_path() -> Path:
    return ats_home() / "learned_terms.yaml"


def enabled() -> bool:
    return os.environ.get("ATS_LEARNED_TERMS", "").lower() not in ("off", "0", "false", "no")


def _load(path: Path | None = None) -> dict:
    import yaml

    p = path or learned_path()
    if not p.exists():
        return {"approved": [], "candidates": {}}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {"approved": [], "candidates": {}}
    approved = [str(t).strip().lower() for t in data.get("approved") or [] if str(t).strip()]
    cands = data.get("candidates") or {}
    if not isinstance(cands, dict):
        cands = {}
    return {"approved": approved, "candidates": cands}


def _save(data: dict, path: Path | None = None) -> None:
    import yaml

    p = path or learned_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True), encoding="utf-8")


def approved_terms(path: Path | None = None) -> set[str]:
    """Terms a human promoted into the taxonomy ('' set when disabled)."""
    if not enabled():
        return set()
    return set(_load(path)["approved"])


def record_candidates(found: dict[str, str], path: Path | None = None) -> list[str]:
    """Count verified-but-unknown terms. `found` maps term -> example quote.
    Returns the terms recorded. Never raises: learning is a side effect and
    must not break a scoring run (read-only home dir, bad YAML, ...)."""
    if not enabled() or not found:
        return []
    try:
        data = _load(path)
        approved = set(data["approved"])
        recorded = []
        for term, quote in found.items():
            if term in approved:
                continue
            entry = data["candidates"].get(term) or {}
            data["candidates"][term] = {
                "count": int(entry.get("count") or 0) + 1,
                "example": entry.get("example") or quote[:200],
            }
            recorded.append(term)
        if recorded:
            _save(data, path)
        return recorded
    except Exception:  # noqa: BLE001 — see docstring
        return []


def candidates(path: Path | None = None) -> list[tuple[str, int, str]]:
    """(term, count, example), most-seen first."""
    data = _load(path)
    rows = [(t, int((e or {}).get("count") or 0), str((e or {}).get("example") or ""))
            for t, e in data["candidates"].items()]
    return sorted(rows, key=lambda r: (-r[1], r[0]))


def promote(term: str, path: Path | None = None) -> bool:
    """Move a term into `approved`. Returns False if it was already there."""
    term = term.strip().lower()
    if not term:
        return False
    data = _load(path)
    if term in data["approved"]:
        return False
    data["approved"].append(term)
    data["approved"].sort()
    data["candidates"].pop(term, None)
    _save(data, path)
    return True


def reject(term: str, path: Path | None = None) -> bool:
    """Drop a candidate (it will be re-recorded if the LLM keeps finding it)."""
    data = _load(path)
    if data["candidates"].pop(term.strip().lower(), None) is None:
        return False
    _save(data, path)
    return True
