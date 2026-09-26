#!/usr/bin/env python3
"""Corpus regression gate — the JD extractor may never get WORSE.

tests/jd_corpus holds hand-labelled JDs across many domains (a tuning set
plus a held-out set nobody tunes against); each set's baseline.json records
the metrics the extractor achieved when it was last accepted. This test
fails if any metric drops below its baseline. An improvement passes —
re-record it with:

    python cli.py eval --write-baseline tests/jd_corpus/baseline.json
    python cli.py eval --corpus tests/jd_corpus/holdout \
        --write-baseline tests/jd_corpus/holdout/baseline.json

    python tests/test_corpus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ats_checker import evaluation as ev  # noqa: E402

BASELINE = ev.DEFAULT_CORPUS / "baseline.json"
HOLDOUT = ev.DEFAULT_CORPUS / "holdout"
TOLERANCE = 0.0005   # rounding only — any real drop fails

passed = failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


items = ev.load_corpus()
print("== corpus files are valid ==")
check("corpus has at least 30 labelled JDs", len(items) >= 30, str(len(items)))
check("corpus spans at least 15 domains", len({i.domain for i in items}) >= 15,
      str(sorted({i.domain for i in items})))
check("ids are unique", len({i.id for i in items}) == len(items))
for item in items:
    problems = ev.validate_item(item)
    check(f"{item.id} is well-formed", not problems, "; ".join(problems))

holdout = ev.load_corpus(HOLDOUT)
check("held-out set has at least 10 JDs", len(holdout) >= 10, str(len(holdout)))
check("held-out ids don't overlap the tuning set",
      not ({i.id for i in holdout} & {i.id for i in items}))
for item in holdout:
    problems = ev.validate_item(item)
    check(f"holdout {item.id} is well-formed", not problems, "; ".join(problems))


def gate(label: str, corpus: list, baseline_path: Path) -> None:
    print(f"\n== {label}: no metric regresses below baseline ==")
    current = ev.evaluate(corpus, ev.rule_extractor).metrics()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for metric, base in sorted(baseline.items()):
        now = current.get(metric)
        check(f"{label} {metric}: {now} >= baseline {base}",
              now is not None and now + TOLERANCE >= base, f"(was {base}, now {now})")


gate("tuning set", items, BASELINE)
gate("held-out set", holdout, HOLDOUT / "baseline.json")

print(f"\n{'=' * 60}\nCorpus tests: {passed} passed, {failed} failed")
if failed:
    print("Run `python cli.py eval` to see every miss.")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all_checks_passed():
    """pytest entry point: the checks above run at import (collection) time."""
    assert failed == 0, f"{failed} check(s) failed — run this file directly for details"
