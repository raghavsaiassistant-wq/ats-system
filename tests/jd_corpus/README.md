# JD extraction corpus

Hand-labelled job descriptions used to **measure** the JD extractor
(`python cli.py eval`). All JDs are fictional but written to match how real
postings phrase things across domains and regions.

| Set | Files | Purpose |
|---|---|---|
| `*.yaml` (30) | tuning set | Rule fixes may be driven by misses here. Scores here are optimistic. |
| `holdout/*.yaml` (10) | held-out set | **Never tune against these.** They estimate how the extractor does on JDs it wasn't fixed for. |

`baseline.json` / `holdout/baseline.json` record the accepted scores;
`tests/test_corpus.py` fails CI if any metric drops below them.

## Rules for labels

- Labels describe **what the JD says**, not what the code returns. A skill the
  taxonomy doesn't know yet is still labelled — it shows up as a recall miss.
- `min_degree` is the **floor**: "MS or PhD" → `masters`; "Bachelor's required,
  Master's preferred" → `bachelors`, mandatory. "Associate's" maps to `diploma`.
- Omit a key when the truth is genuinely ambiguous (e.g. seniority for
  "Product Manager"), so it isn't scored.
- `required_terms`: what a recruiter would treat as must-have. `other_terms`:
  every other genuine skill, tool, method or qualification in the JD — anything
  extracted that is in neither list counts against precision.

## Workflow

```bash
python cli.py eval                               # tuning set, with every miss
python cli.py eval --corpus tests/jd_corpus/holdout
python cli.py eval --write-baseline tests/jd_corpus/baseline.json   # accept an improvement
```

Once a fix has been driven by a holdout miss, that file is no longer held out:
move it to the tuning set and write a fresh holdout JD to replace it.
