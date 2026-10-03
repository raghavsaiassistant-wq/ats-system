# AI_GUIDE — read this first if you are an AI assistant

You (Claude, ChatGPT, Codex, any assistant) were given this repo plus a
user's resume and a job description (a file, pasted text, or a job link).
Your job: produce the **best resume the truth allows** for that job, show the
scores, and tell the user plainly which gaps only real experience can close.

This tool scores a resume the way a hiring pipeline reads it (recruiter
searches, the HR screen, the hiring manager's evidence check) and builds a
tailored resume **only from the user's own confirmed bullets**. Scores are
percentages of things measured. They are never a chance of being selected.

## Setup (3 commands)

```bash
pip install -r requirements.txt
python cli.py test-llm --provider manual
python cli.py run --resume cv.pdf --jd jd.txt --provider manual
```

`--provider manual` means **you** are the LLM: no API key is needed. (If the
user has an API key in `.env`, drop `--provider manual` and the tool calls
their LLM itself. `--offline` runs with no LLM at all: selection only.)

## Which path you are on

1. **You can run commands** (Claude Code, Codex, ChatGPT/Claude with code
   execution on the uploaded zip): run the commands yourself and act as the
   LLM as described below.
2. **You can't run code** (a plain chat): the user runs the commands on their
   computer. Each time the tool prints a prompt bundle, they paste it to you;
   you answer it; they paste your answer back. Tell them the exact command to
   run next.

Never pretend to have run the tool. Without running it, any score you quote
is a guess, and you must say so.

## Acting as the LLM (copy-paste mode)

With `--provider manual`, every LLM call the run needs is written to one file,
`llm_prompts.md` (also copied to the clipboard when possible), and the command
exits with code 4 (or, at a terminal, waits for the reply). Then:

1. Read `llm_prompts.md`. It holds numbered **INSTRUCTIONS** blocks and
   **TASK ID**s. Each task says which instructions it follows.
2. Do each task exactly as its instructions say, as if it were the only task.
3. Reply with **one JSON object** mapping every task ID to that task's JSON
   answer: `{"3f2a9c01b7de": {...}, "77d01e5a9c42": {...}}`. Save it to a file
   (for example `answers.json`).
4. Re-run the **same command** with `--answers answers.json`. Answers are
   cached by prompt, so they are never asked twice. Later rounds can ask new
   prompts that depend on your earlier answers; repeat until nothing is
   pending (exit code 0).

Rules for your answers (the tool checks them, and drops what fails):

- When a task asks for a **quote**, copy the job-description line **exactly**.
  An item whose quote isn't in the JD, or doesn't contain the item, is dropped.
- **Never add facts.** No number, tool, employer, title, date or outcome that
  isn't in the input. A rewording that adds or changes a number is rejected.
- If you can't do a task honestly (a bullet doesn't really show a skill),
  return the empty answer the instructions describe (`""`, `[]`, `null`).
- JSON only: no prose, no comments inside the JSON.

## The optimize loop, step by step

`python cli.py run --resume cv.pdf --jd <file or URL>` does all of this:

1. **Profile.** If `profile.yaml` is missing, it asks a few questions (title,
   years, degree, location, work modes, notice). Blank answers skip a check.
2. **Evidence bank.** If `master_resume.yaml` is missing, the resume is
   transcribed into it (that's an LLM task: you, in manual mode). Every
   transcribed bullet starts **unreviewed**. The user must confirm each one
   ("is this true, could you defend it in an interview?"). Unconfirmed bullets
   are never used. Ask the user; never confirm bullets for them.
3. **Job description.** A link is fetched; if that fails, paste the JD text
   (`--jd-text "..."` or a file).
4. **Optimize** (`python cli.py optimize --jd jd.txt` on its own):
   - Round 0 is the plain tailored draft.
   - Each round scores the draft offline (Search Visibility, keyword match,
     HR resume checks), looks at what's missing, and tries truthful changes:
     swap in or add other confirmed bullets, and reword a bullet with the
     JD's phrasing **only** for a skill that bullet's own tags already claim.
   - Only changes that raise the score are kept, so the score never goes down.
   - It stops on a plateau (2 rounds gaining under 1 point), after
     `--max-rounds` (default 5), or when no truthful change is left.
   - The **no-apply gate** still blocks: if the JD rules the user out on a
     fact (years, sponsorship, work mode, a mandatory degree), it stops and
     says why. `--force` overrides it, for practice only.
5. **Review.** Reworded bullets are **proposals**. At a terminal the user
   accepts or rejects each. Otherwise they're written to
   `<out>.proposals.json`; the user sets `"accepted": true` on the ones they
   agree with and re-runs with `--apply-proposals <that file>`.
   `--auto-accept` skips the review and is riskier: only use it if the user
   asks for it.
6. **Output.** `optimized_resume.txt` + `.docx`, the score per round, a diff
   of what changed and why, and **"Gaps the truth can't close"**.

## Truth rules (non-negotiable)

- Never add a skill, number, employer, title or date that the evidence bank
  doesn't have, not in the resume and not in your own chat reply.
- No keyword stuffing. A JD term appears only inside a real bullet or in a
  true skills line. Repeating a term doesn't raise any score.
- Unreviewed bullets are never used. Only the user can confirm a bullet.
- If the user asks you to "just add" a skill they don't have, decline and
  explain that it fails at the interview. Offer the gaps list instead.
- Don't promise selection. Nobody can: it depends on the other applicants,
  the interviews and the recruiter.

## What to tell the user at the end

1. The final score next to the starting score, and what changed (from the
   "What changed, and why" panel), in plain words.
2. Which reworded bullets they accepted or still need to review.
3. **Gaps the truth can't close**, grouped:
   - *nothing in your bank*: only real experience closes it. Don't add it.
   - *only an unconfirmed bullet*: confirm it if it's true, then re-run.
   - *in your bank, didn't fit*: raise `--max-lines` or make room.
4. Any no-apply-gate blockers (facts about them, not the resume).
5. Where the files are: `optimized_resume.txt` and `optimized_resume.docx`.
6. That the scores measure the resume, not their chances.

## Command reference

```bash
python cli.py run --resume cv.pdf --jd https://example.com/job   # everything, end to end
python cli.py optimize --jd jd.txt --provider manual             # the loop, you as the LLM
python cli.py optimize --jd jd.txt --answers answers.json        # continue after answering
python cli.py optimize --jd jd.txt --apply-proposals optimized_resume.proposals.json
python cli.py optimize --jd jd.txt --offline                     # no LLM: selection only
python cli.py tailor --jd jd.txt                                 # one pass, no loop
python cli.py score --resume cv.pdf --jd jd.txt --offline        # score a resume as-is
python cli.py init-profile                                       # profile.yaml template
python cli.py init-master --from cv.pdf --provider manual        # resume -> evidence bank
python cli.py test-llm                                           # check the user's API setup
python server.py --llm manual                                    # web UI, copy-paste mode
```

The web UI (`python server.py`, then http://127.0.0.1:8420) has a Setup tab
(profile form, resume upload, bullet review) and shows a **Copy prompt /
Paste answer** step whenever copy-paste mode needs an answer.
