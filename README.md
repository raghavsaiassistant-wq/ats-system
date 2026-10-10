# 🧭 ats-system — three-layer resume scoring + JD-tailored resume generator

A local tool that scores your resume against a job description the way the
hiring pipeline actually filters you: **machine first, recruiter second,
hiring manager third**. Runs on your own machine against your own LLM
(Ollama local/cloud or any OpenAI-compatible API). No third-party resume
site is required. Configured cloud LLMs receive the text needed for their requests;
local writing review sends no text to a detector. Optional GPTZero checking sends
the extracted resume only after explicit consent and requires a separate API key.

> **Honesty-first design:** the tailor never invents. Every claim traces to
> a real bullet in your evidence bank. JD terms you can't back are reported
> as gaps to fix by gaining experience — never stuffed in.

**Using Claude or ChatGPT? Upload the repo and say: "read AI_GUIDE.md".**
No API key needed: the chat AI plays the LLM in copy-paste mode.

**Phone or plain chat, with no code execution?** Read
[CHAT_WORKFLOW.md](CHAT_WORKFLOW.md) with your resume and JD. It provides
drafting and structured chat assessments without installation. A repo link
does not give a chat execution capabilities: Python scores and a trained
AI-writing detector must be marked **not run** unless actually executed.

## The three layers

Your resume passes three readers with three different criteria, and it can
fail at any one for reasons the others don't care about. So the tool reports
**three separate scores** and never blends them:

**Layer 1 — Search Visibility.** Most ATS platforms (Workday, Greenhouse,
Lever, iCIMS) don't auto-reject on a keyword percentage — a recruiter
*searches* the applicant pool, and a resume that doesn't match the search is
never opened. So this layer asks: *if a recruiter ran the searches this JD
implies, how many would find you?* It builds up to six recruiter-style
queries from the JD's title and its top requirements (e.g.
`"data analyst" AND sql AND ("power bi" OR tableau)`), runs each against
your resume, and shows every query with what it hit and missed. Tools are
OR-ed only when the JD itself names them side by side, and soft skills are
never searched. Two things sit beside the score, never blended into it:
a **parse gate** (pass/fail — can an ATS read the file at all: text
extraction, columns, tables, contact info, section headers) and an **LLM
fit read** (a model's meaning-level judgment, labelled as exactly that).
One alias table is shared by scorer and generator, so the two can never
disagree about synonyms.

**Layer 2 — HR Screen.** The ~30-second human filter. Deliberately
rule-based, not LLM — a recruiter screen is a hard-filter checklist. Extracts
years floor, degree, certifications, work mode, sponsorship, seniority,
location from the JD and checks each against your profile. Blank profile
fields are SKIPPED, never guessed. Findings split into **RESUME-fixable**
vs **YOU/candidacy** — because 42% resume-fixable with 90% candidacy fit
means your presentation is the problem, not your fit.

**Layer 3 — Manager Evidence Score.** How your evidence holds up to the
person deciding you can do the job. LLM-scored on six dimensions:
quantification, outcomes vs responsibilities, skills backed by real work,
scope match, day-to-day relevance, credibility. Returns weakest bullets
with concrete rewrites, plus the claims a manager would probe.

## Writing review

Resume review preserves standard action verbs, technical compounds, tools lists
inside experience bullets, and inline or standalone skills headings (including
Core Competencies and Tech Stack). A vocabulary term present in the supplied JD
is exempt from vocabulary feedback; the score command passes its JD automatically,
and `/writing-check` accepts optional `jd_text`. DOCX list paragraphs retain list
markers, and common PDF bullet symbols are recognized. If optional writing review
fails, core scoring still completes with an unavailable-review note.

The local rules also catch detail-oriented, team player, self-starter,
hard-working and passionate self-descriptions, plus weak contribution openers
such as Assisted with, Involved in and Was responsible for. Likely PDF
continuation lines are joined for review while preserving the first extracted
line number. Customer/user journey, curated datasets and deep dive are treated
as domain terminology. `/writing-check` rejects external providers with
`resume_path` before reading a file; paste the exact text to send instead.
The CLI reports a missing resume file with a concise error and exit code 2.

Numbers are counted in one interview-evidence reminder, without per-line proof
findings. Repeated openers are checked within each role, using non-bullet headers
as boundaries after likely continuations are joined. Languages subheadings stay
inside skills sections. The CLI, score report and browser group issues under one
excerpt and label text positions **Extracted line**, since those are not original
PDF/DOCX coordinates. JSON retains flat `findings` for compatibility and adds
`grouped_findings` with `{line, excerpt, issues}`, `number_count` and
`summary_notes`. The adapter's resume rule table controls patterns, advice and
vendor-category feedback; the vendored source remains unchanged. With the local
provider, CLI `--consent` prints a warning to stderr and sends nothing externally.
Pass `writing-check --jd job.txt` for JD vocabulary exemptions; the browser uses
the current pasted JD for local review. `samples/writing_review_resume.txt` is a
synthetic regression example with one actionable unclear-contribution bullet.

Promotional three-item lists receive feedback when at least two items are common
buzzwords and no item is a known skill. Numeric reminders exclude phone numbers,
calendar years (1900–2099), and numeric dates. Capitalized PDF continuations such
as `Python migration` stay with the previous bullet; section headings, dates,
title/company separators, and blank lines preserve role boundaries.
Short capitalized labels immediately before a new bullet also preserve undated
company boundaries. A preceding lowercase connector such as `for the` keeps a
capitalized continuation attached to its bullet. Numeric reminders include
abbreviations such as `$2M`, `5K`, `3x`, `50L`, `2mn` and `4cr`.

[WRITING_CHECKLIST.md](WRITING_CHECKLIST.md) provides the checklist for chat use.
The application flags generic phrases, unclear contributions and repeated
bullet openings locally, separately from all three scores. It also includes
[SlopMonster](https://github.com/ItsssssJack/SlopMonster) pattern checks with
resume-specific safeguards, editing recommendations, and conservative proposed
wording where possible. Review edits before using them; no changes are automatic.
Upstream MIT attribution is retained in `ats_checker/vendor/slopmonster/`. These observations
do not classify authorship. Optional GPTZero results are experimental and are
never used as a selection gate. API fees may apply; no subscription is needed
for the local review or chat checklist.

## Quick start

```bash
git clone https://github.com/raghavsaiassistant-wq/ats-system.git
cd ats-system
pip install -r requirements.txt

# Configure your LLM (Ollama Cloud, local Ollama, or any OpenAI-compatible API)
cp .env.example .env     # then edit: provider, base_url, model, api_key

# Easiest: the web UI walks you through setup on first run (no YAML editing)
python server.py                # http://127.0.0.1:8420 -> Setup tab

# Or from the command line:
python cli.py test-llm          # verify LLM connectivity in one request
python cli.py init-profile      # create profile.yaml — fill in once
python cli.py init-master --from cv.pdf   # LLM transcribes your resume into the evidence bank

# Score any resume against any JD
python cli.py score --resume cv.pdf --jd jd.txt

# Generate a JD-tailored resume from your evidence bank
python cli.py tailor --master master_resume.yaml --jd jd.txt
```

Supports `.pdf`, `.docx`, `.txt` resumes. Web UI: `python server.py` →
http://127.0.0.1:8420

### First-run setup in the web UI

On first run (no `profile.yaml` or no evidence bank yet) the web UI opens its
**Setup** tab, which has three steps:

1. **Profile.** A form for the facts a recruiter screens on (years, degree,
   location, work modes, authorisation, notice, salary, certifications). It
   writes `profile.yaml`. Leave any field blank and its check is skipped,
   never guessed.
2. **Evidence bank.** Upload your resume (PDF, DOCX or TXT) and the LLM
   transcribes it, verbatim, into `master_resume.yaml`. This is the same step
   as `init-master --from`. No LLM? Start a blank bank and type your bullets
   in.
3. **Review.** Every LLM-transcribed bullet starts **unreviewed**, and the
   tailor won't use it until you confirm it's true. You confirm each bullet
   one at a time; there is deliberately no "confirm all". You can also edit
   or delete bullets, add roles, and fix contact details. Bullets you type
   yourself count as confirmed.

The same review state lives in the YAML: an unconfirmed bullet carries
`reviewed: false`, and deleting that line confirms it. Banks you wrote by
hand have no such lines, so nothing changes for them. `tailor` lists how many
unreviewed bullets it left out.

The setup endpoints only accept JSON, so another website can't make your
browser write these files. Files are only ever written to the paths the
server was started with (`--profile`, `--master`). The resume upload goes
only to your configured LLM.

## Optimize: the best resume the truth allows

```bash
python cli.py run --resume cv.pdf --jd https://...       # everything, end to end
python cli.py optimize --jd jd.txt                       # the loop on its own
```

`optimize` builds the tailored draft, scores it offline (Search Visibility,
keyword match, the HR screen's resume checks), tries the next truthful change,
and rescores. It keeps only changes that raise the score, so the score never
goes down. It stops on a plateau (two rounds gaining under a point), after
`--max-rounds` (default 5), or when no truthful change is left.

The changes it may make are limited:
- swap in or add other **confirmed** bullets from your evidence bank;
- reword a bullet to use the JD's name for a skill that bullet's own tags
  already claim.

Every rewording must keep the bullet's numbers and tools. A **fabrication
guard** drops any rewording that would name a JD skill your bank doesn't
have.

Reworded bullets are **proposals**: you accept or reject each one before
the final file is written. `--auto-accept` skips that review for JD-phrasing
rewordings and is riskier. It never applies the manager layer's rewrites,
which can change what a bullet claims; those always need your explicit yes.
While copy-paste answers are still pending, no resume file is written.
`tailor` has no review step, so it never applies manager rewrites at all: it
lists them as suggestions for you to copy into your evidence bank if they're true.

The output is:
- the score for each round;
- what changed, and why;
- `optimized_resume.txt` and `.docx`;
- **"Gaps the truth can't close"**, which marks each missing JD term as
  nothing in your bank, only an unconfirmed bullet, or in your bank but
  didn't fit.

`run` also creates `profile.yaml` and the evidence bank first if they're
missing. It transcribes your resume, asks you to confirm each bullet, and
asks for the JD text if a job link can't be fetched.

### No API key: copy-paste mode

Add `--provider manual` (or set `ATS_LLM_PROVIDER=manual`, or start the web UI
with `python server.py --llm manual`). Every prompt a run needs goes into one
file, `llm_prompts.md`, which is also copied to your clipboard. Paste it into
Claude or ChatGPT, then save the JSON reply to a file. Re-run the command with
`--answers reply.json` (which implies `--provider manual`), or paste the reply
at the terminal.

Answers are cached, so a re-run needs no paste. Pasted answers go through
the same checks as an API model's: quote verification, sanity ranges and
rewrite verification. In the web UI this is a **Copy prompt / Paste answer**
step.

## What these percentages are (and are not)

They are **percentages of things measured** — "a recruiter's search finds
you in 5 of 6 queries", "you meet 78% of their stated screening criteria" —
not probabilities of passing. Your rank relative to
the applicant pool, whether a human opens your file, and req reality are
invisible to any resume tool. Use the scores to fix what's fixable.

The old blended "ATS score" (formatting + keyword + semantic) is still in
the JSON output as `ats_score` for existing scripts, but it's deprecated —
read `search_visibility_pct` and `parse_safe` instead.

## Reading the JD: rules, or a quote-verified LLM

By default the JD is read by the deterministic rule engine: precise, but
it only finds the skills in its taxonomy. `--jd-extractor llm` (on `score`
and `batch`, or `"jd_extractor": "llm"` in the API) adds an LLM reading
under one rule: **every item must quote the JD line it came from, and the
item must appear inside that quote**. A requirement the model invents can't
cite a line that isn't there, so it gets dropped, and the report lists
each dropped item with the reason. Verified values win, the rules fill every gap,
and if the LLM is unreachable the rules answer alone.

```bash
python cli.py score --resume cv.pdf --jd jd.txt --jd-extractor llm
python cli.py eval --extractor hybrid      # measure it on the labelled JD corpus
python cli.py learned list                 # skills the LLM found that the taxonomy lacks
python cli.py learned promote "gd&t"       # you decide what joins the taxonomy
```

LLM answers are cached per JD under `~/.ats-system/cache/` (set `ATS_HOME` to
move it), so re-scoring a JD is free and gives the same result.

What the rule engine treats as a requirement:

- **Either/or skills count once.** "Power BI or Tableau", "Python/R" and a
  list containing "or" ("SQL, Power BI, Tableau, or advanced Excel") are met
  by any one member; a plain comma list ("SQL, Python, Excel") needs every
  member. A parenthesised list of two or more examples ("CRM tools
  (Salesforce, HubSpot, Zoho CRM)") is met by any example. A single name in
  parentheses ("our data warehouse (Snowflake)") stays a required product,
  and the generic term never stands in for it.
- **Noise is not a skill.** A mostly-uppercase headline ("WE'RE HIRING |
  BUSINESS ANALYST | CHENNAI") contributes only taxonomy terms and exam
  codes; boilerplate such as CV, DM, PFB and currency codes is ignored; and
  an "About Us" / "Who we are" section describes the employer, so its terms
  are dropped.
- **Right-to-work gates.** "Must currently be based in the UAE with a valid
  visa" counts as no sponsorship, the same as "no visa sponsorship".
- **Multi-column PDFs.** A page is flagged multi-column only when a real
  vertical gutter separates the columns.

## The application log

`log` records every application and outcome; `log stats` refuses to show
conversion rates until you have 20 resolved outcomes (a 2-of-3 sample as a
percentage is noise). Past that threshold it shows, from your own data:
conversion by score band and by apply-timing, which layer actually predicts
your outcomes, and which **component** does (each recruiter search, each
recruiter check, each manager dimension, keyword match, LLM fit).

Every rate comes with its n and a Wilson 95% interval, e.g.
`12/40 = 30% (95% CI 18–45%)`. Every correlation comes with a Fisher-z 95% interval. A
correlation whose interval crosses 0 is reported as "not distinguishable
from zero", and none is called "strong" on fewer than 30 applications.
`log stats` first marks pendings older than 45 days as ghosted
(`--reap-days N`, or `--no-reap` to skip, in which case it says how many it
left out). `log stats --json` gives the same numbers as JSON.

Logging is meant to cost nothing:

```bash
python cli.py score --resume cv.pdf --jd https://boards.greenhouse.io/acme/jobs/1 --log
#   company/role default from the JD: role = the JD title; company = a
#   "Company:" / "About X" line, else the posting URL (Greenhouse, Lever,
#   Workday, ... or the employer's own careers site). --company/--role override.
python cli.py log outcome --last --status recruiter_call    # the one you just logged
python cli.py log outcome acme --status rejected_auto       # fuzzy company match
```

Set `ATS_AUTO_LOG=1` (environment or `.env`) to make `score` and `tailor`
log by default; `--no-log` skips one run. It stays off unless you turn it on,
so practice runs don't end up in your stats. In the web UI, the
Applications tab has one-click outcome buttons on every row. Each scored
application also stores its component scores (a `components` JSON column,
added to existing databases automatically), and `log export` includes them
as `components_json`.

## Privacy

Everything runs locally. Your resume, JD, and evidence bank never leave
your machine except as prompts to **your own configured LLM**. No
telemetry, no accounts, no uploads. The JD-extraction cache and learned
terms live in `~/.ats-system/` on your machine.

## License

MIT
