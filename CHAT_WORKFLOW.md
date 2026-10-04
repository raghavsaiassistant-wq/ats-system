# Resume workflow directly in chat

Use this document with your current resume or confirmed experience notes and the full job description. No installation or detector subscription is required for this **chat-guided review**. This document does not give a chat new browsing, file-reading, Python execution or model-loading capabilities.

If the chat cannot open the repository, paste this document's contents into it. A repository URL alone is not guaranteed to expose every file. This workflow uses the chat's existing model; it does not download a separate detector or execute the repository's Python code.

## Inputs

Collect the resume and JD. Ask only for missing facts needed for the review: actual relevant experience, education/certifications, location and relocation, work authorization, acceptable work modes, notice and salary expectations with currency and period. Unknown facts remain unknown.

If a PDF or DOCX cannot be read reliably, request readable text. Do not pretend that pasted text verifies the original file's columns, tables or ATS extraction quality.

## 1. Make an evidence inventory

List the candidate's actual roles, dates, responsibilities, achievements, tools and qualifications. Separate confirmed facts from ambiguous claims. The user's supplied resume is the starting evidence; a job-description requirement is not evidence that the candidate has that skill.

Preserve employer names, dates, titles, number-to-outcome associations and scope. Do not add metrics, tools or outcomes. Ask for clarification where a useful rewrite would need a new fact. Experience is not independently verified merely because it appears in the resume.

## 2. Extract job requirements with citations

Build a table containing:

| JD requirement | Exact supporting JD excerpt | Required/preferred/unclear | Candidate evidence | Matched/gap/unknown |
|---|---|---|---|---|

Distinguish “both A and B” from “A or B.” Keep preferred requirements separate from mandatory requirements. Preserve lowest acceptable degree, negation, years scope, salary period and location. Flag ambiguity rather than guessing.

## 3. Draft the tailored resume

Select the strongest confirmed evidence for the actual role. Use standard headings, a concise supported summary, relevant skills, experience and education. Include JD terminology only where the evidence supports it. A term named only in a skills list does not establish demonstrated professional experience.

Make writing clear and specific. Do not add artificial errors, invent achievements or repeatedly rewrite to obtain a detector label. Give the user a change list and ask them to check any wording that could change the meaning before sending the resume.

## 4. Give three separate reviews

### ATS / search relevance — chat assessment

Show matched and missing job terms and plausible title/skill searches, with supporting excerpts. Explain why a search might or might not find the resume. Do not claim these searches are the exact Python implementation or the employer's actual ATS behavior.

Do not invent the repository's search-visibility percentage without running its algorithm. If a simple coverage count is provided, show its explicit denominator and identify it as a separate chat-calculated count. Do not infer original-file parse safety from pasted text.

### HR requirements — chat assessment

For each stated requirement, report supported match, gap or unknown. Explain location and relocation separately from work authorization. Compare salary only with known matching currencies and compatible periods. Do not penalize unknown facts as failures.

Keep general preparation observations, such as possible employment gaps, separate from explicit JD requirements. A gap can require context without implying dishonesty or lack of suitability.

### Manager evidence — model judgment

Review quantification, outcome focus, evidence backing, scope, domain relevance and credibility. Explain each concern with the exact bullet and JD context. A qualitative strong/mixed/weak/unknown assessment is enough; do not present a judgment as verified capability, factual authentication or a hiring probability.

Any numeric rubric rating must be explicitly identified as this chat model's subjective rating, with the assessed dimensions and missing information shown. It is not an executed result from the repository.

## 5. Keep writing review distinct from AI detection

Apply [WRITING_CHECKLIST.md](WRITING_CHECKLIST.md) to identify repeated phrasing, broad self-description, unclear contribution and missing outcome context. For every finding, show the exact excerpt, reason and a proposed revision grounded in confirmed facts.

Preserve the candidate’s own meaning and vocabulary where clear; use wording they can comfortably explain in an interview. Vary sentence openings only when this improves clarity: consistent parallel bullets are acceptable, and artificial mistakes or random spacing are not improvements. Explain these as writing observations, not as AI fingerprints. Uniform bullets, good grammar and particular words do not prove AI authorship.

For a chat without a separately executed detector, the mandatory detector result is:

**Model-based AI detector: NOT RUN. No AI/human classification or authorship percentage is available.**

Do not replace that status with the chat's opinion, invent a detector score, or claim GPTZero/Binoculars ran. If the user supplies an external result, label it user-supplied, identify the provider and resume version, and preserve its original meaning. It does not prove authorship.

If execution tools are actually available, the assistant may use the separate runtime workflow in [AI_GUIDE.md](AI_GUIDE.md). Read real output and report failures honestly. The application includes local writing observations and an optional GPTZero adapter; a local trained detector has not yet been implemented.

## Final response format

1. Tailored resume in copyable text.
2. Requirement-to-evidence table.
3. Separate ATS relevance, HR requirements and manager evidence assessments, labeled **chat assessments** unless actually executed.
4. Writing observations and truthful suggested changes.
5. Model-based detector status and which checks could not execute.
6. Real gaps and remaining facts to confirm.

Selection depends on factors beyond these documents. Do not auto-reject a candidate because of an AI-writing signal or imply that improving wording guarantees selection.

## Capability boundary

This is the practical repository-based solution for a chat-only setting: a consistent, auditable review workflow. It meets the no-installation requirement for drafting and guided review. It **does not meet** a requirement for guaranteed execution of every software feature or a trained AI detector in every free mobile chat.

To execute those features, the chosen chat must have a suitable runtime or access a separately hosted service. A document in a repository cannot grant those capabilities. If actual AI detector execution is mandatory and every runtime/service is excluded, there is no complete solution under those constraints.
