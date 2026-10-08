# Simple résumé writing checklist

Purpose: an LLM can apply this checklist directly in chat, without installing a detector. It identifies writing issues, not the history of who wrote the text.

## Checks

| Check | What to inspect | Appropriate feedback |
|---|---|---|
| Repeated wording | Several bullets repeat an opening or unnecessarily repeat the same sentence structure | Explain the repetition and suggest clearer phrasing where useful. Parallel bullets are normal; variation is not mandatory. |
| Generic self-description | Broad claims such as “results-driven” or “dynamic professional” without supporting detail | Point to the exact phrase and suggest replacing it with confirmed experience. These are not AI-only words. |
| Unclear contribution | A bullet says “worked on” or “responsible for” without explaining the person's role | Ask what the candidate actually did. Do not assume missing information. |
| Missing context or outcome | An accomplishment does not explain the problem, action, scope or result sufficiently | Request relevant truthful detail. A qualitative outcome can be sufficient; numbers are not mandatory. |
| Unsupported additions | A revision introduces a skill, metric, employer, outcome or scope absent from supplied evidence | Hold the addition for confirmation or remove it. Preserve what each number refers to, not just the numeric tokens. |
| Excessive JD copying | Long passages repeat the job description without candidate-specific evidence | Preserve legitimate matching terminology but express actual work and contributions. Copying a technical term is not a defect by itself. |
| Consistency and formatting | Inconsistent tense, unclear dates, awkward text extraction or confusing layout | Improve readability and retain conventional headings and consistent spacing. |

## Instructions for the review

Apply the checks to the provided résumé and job description. For each finding, show the exact excerpt, check name, explanation and a suggestion based only on confirmed facts. Mark unresolved factual questions clearly. Do not rewrite automatically where the meaning could change.

Do not insert random spacing, artificial mistakes or forced variations. Do not treat polished grammar, lack of metrics, particular vocabulary or parallel bullets as proof of AI use. Do not supply an AI-authorship percentage from checklist counts.

Use these results:

- **Review needed:** one or more specific issues were found; show them.
- **No issues found by this checklist:** these rules found no issue. This does not certify quality or human authorship.
- **Insufficient information:** the text or supporting facts are too incomplete for the relevant check.

The separate authorship result is **not established by this checklist**. An actual detector result, if supplied or executed, must identify its source and remain distinct from this review.

## Why this is not a yes/no AI test

A human can use repeated, conventional résumé wording. An AI can generate varied, specific wording. Therefore both can pass or fail the same style checklist. Changing those features does not establish the drafting history.

The supplied reports support investigating these writing observations and testing detector failure modes. They do not establish a universal word blacklist, a random-spacing signal, or the inference “patterns absent means human-written.” Commercial detectors combine learned signals; this checklist does not recreate their private model.

This is the lightweight solution for better writing in a repo-link chat workflow. It does not require an open detector model. It should be named **Writing Review**, rather than presented as reliable AI detection or an “AI-undetectable” certificate.

## SlopMonster patterns and recommendations

The runtime includes a pinned, MIT-licensed copy of [SlopMonster](https://github.com/ItsssssJack/SlopMonster), adapted for resume review. Check vague promotional vocabulary, filler constructions, dense punctuation, promotional three-item lists and numbers that need supporting evidence. These are advisory observations, not an authorship test.

For chat-only use, apply those checks editorially; do not claim the Python checker ran. Show the original excerpt, the issue, and proposed wording grounded in the supplied evidence. If a meaningful rewrite needs missing facts, ask for the facts or show editing guidance rather than inventing a replacement. Keep real metrics, technical terms, and genuine lists of skills. Consistent spacing improves readability; extra spacing is normalized by the runtime checker.

The runtime returns `suggestion` for every finding, and `suggested_rewrite` where a conservative literal edit is available. Every proposed edit requires review; no changes are applied to the resume automatically. It does not run SlopMonster's external second-model script or score resumes as AI/human.
