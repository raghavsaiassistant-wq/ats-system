# ATS Score Checker — three-layer resume scoring + JD-tailored resume generator

A local tool that scores your resume against a job description the way the
hiring pipeline actually filters you: machine first, recruiter second,
hiring manager third. Runs on your own machine against your own LLM —
Ollama (local or cloud) or any OpenAI-compatible API. No third-party
resume site, nothing uploaded anywhere.

Runs from the CLI, from a small local HTTP API so James (or n8n, or any
script) can call it, or from the built-in web UI (`python server.py`, then
open http://127.0.0.1:8420) — paste the resume and JD, click, read the
same three-layer report.

It also generates: given the JD and your **evidence bank** (a master
resume holding every real bullet you've ever written), `tailor` produces a
resume selected, ordered and rephrased to score as high as it can *honestly*
go against that JD — with every claim traceable to a real bullet, and JD
terms you can't back left out as reported gaps rather than stuffed in.

---

## The three layers

Your resume passes through three different readers with three different
criteria, and it can fail at any one of them for reasons the other two
don't care about. So the tool reports three separate scores and never
blends them into a single "you'll get the job" number.

**Layer 1 — ATS Score.** The machine filter. Three parts: whether your file
parses cleanly into structured fields (two-column layouts, tables, missing
section headers, and failed text extraction are the #1 silent-rejection
cause, ahead of keyword problems); weighted keyword overlap against the
JD's hard requirements — **alias-aware on both sides**, so a resume saying
"PostgreSQL" matches a JD asking for "Postgres", and the generator selects
against the same canonical terms the scorer measures (one shared alias
table, `ats_checker/terms.py`, so the two can never disagree about a
synonym); and semantic fit, judged by your configured LLM,
approximating the embedding-based ranking layer modern platforms
(Workday, Eightfold, Phenom) run on top of Boolean matching.

**Layer 2 — HR Screen.** The ~30-second human filter that comes after the
ATS. Deliberately *rule-based, not LLM* — a recruiter screen is mostly a
hard-filter checklist, and most of those filters are stated outright in the
JD. The tool extracts them (years floor — section-aware, so a
nice-to-have "2+ years with Spark preferred" no longer lowers the role's
gate; degree; named certifications across the common vendor families;
work mode; sponsorship; seniority; stated salary range; stated location)
and checks each against your `profile.yaml`.

Every check reports pass/warn/fail **with the JD line that triggered it**,
so you can see the exact sentence responsible and judge whether the
extraction was right. A check whose profile field is blank is SKIPPED,
never guessed.

Findings are split into two kinds, because they demand different responses:

- **RESUME** — fixable by editing the document. Top-third visibility (what
  share of the JD's top terms appear in the first third — recruiters read
  top-down and stop early), experience legibility (can a recruiter compute
  your years from your dates at a glance), title visibility.
- **YOU / candidacy** — facts about you. Notice period, sponsorship need,
  work mode, years, degree, tenure, gaps, **salary expectations vs a stated
  range, your location vs the JD's location (with `open_to_relocation`)**,
  and right-to-work grounded in the countries you listed under
  `work_authorized_in` rather than a guess. No rewrite changes these.

That split matters: 42% resume-fixable with 90% candidacy fit means your
presentation is the problem, not your fit. The reverse means the opposite,
and editing harder won't help.

Note that most candidacy facts **aren't on your resume at all** — they're
captured by the application form's knockout questions (are you authorized
to work here, will you need sponsorship, notice period, salary
expectations), which employers can configure to auto-reject *before* your
resume is ever scored. The layer models that filter, not your document.

**And it tells you what HR will expect.** Every warn/fail generates a prep
item: what they'll raise, why, and how to be ready — plus the questions
asked on virtually every screen (current/expected CTC, why you're leaving,
why this role, relocation). This is the part you actually use before a call.

**Layer 3 — Manager Evidence Score.** How your evidence holds up to the
person who has to decide you can actually do the job. Scored by your LLM
against a six-dimension rubric: quantification, outcomes vs
responsibilities, skills backed by real work, scope match, day-to-day
relevance, and credibility. Returns your weakest bullets with concrete
rewrites, plus the claims a manager would probe in an interview.

---

## What these percentages are, and what they are not

**They are percentages of things measured. They are not probabilities of
passing.** Same number, completely different claim:

- "You meet 78% of their stated screening criteria" — a measurement. Countable,
  checkable, and every input is cited back to a JD line.
- "78% chance HR passes you" — a prediction. Needs labelled outcome data
  (application → what actually happened) to fit against. With none, the
  number would be invented precision dressed up as analytics.

So the three layers report the first kind. The second kind lives in the
application log, and the tool refuses to show conversion rates until your
own data can support them.

Three things no resume tool can know, this one included:

- **Your rank relative to the other applicants.** Real ATS platforms sort
  candidates against the pool and pass a top slice to a human. You can
  score well in absolute terms and still sit below the cut. This is
  usually the single biggest factor and it's invisible from your documents.
- **Whether a human ever opens it.** For high-volume roles recruiters may
  read only the top 10–20 even when far more cleared the threshold.
- **Req reality.** Already filled internally, frozen, or the recruiter
  stopped looking in week one. None of that is about your resume.

Use the scores to fix what's fixable. A high score removes failure modes;
it doesn't create an offer.

---

## Setup

```bash
cd ats-checker
pip install -r requirements.txt
python cli.py init-profile        # creates profile.yaml — fill it in once
```

`profile.yaml` holds the fixed facts a recruiter screens for that can't be
read off a JD and mostly aren't on a resume either: years of experience,
current title, education level, certifications, location, acceptable work
modes, work authorisation, notice period, salary expectation. Any field you
leave blank turns its check into SKIPPED rather than a guess.

### LLM setup (for layers 1-semantic and 3)

Two backends, same interface. Configure with a `.env` file — **the key never
goes in source, and `.env` is gitignored**:

```bash
cp .env.example .env     # then edit it
python cli.py test-llm   # verify in one request
```

**Option A — Ollama Cloud** (native `/api/chat` protocol, Bearer auth):

```ini
ATS_LLM_PROVIDER=ollama
ATS_LLM_BASE_URL=ollama-cloud
ATS_LLM_MODEL=gpt-oss:120b-cloud
ATS_LLM_API_KEY=your-ollama-cloud-key
```

The model must be one your account can reach — cloud models carry a
`-cloud` / `:cloud` suffix. A 404 or "model not found" from `test-llm` is
nearly always this line, not the key.

**Option B — local Ollama** (no key):

```ini
ATS_LLM_PROVIDER=ollama
ATS_LLM_BASE_URL=ollama-local
ATS_LLM_MODEL=llama3.1
```

**Option C — any OpenAI-compatible API** (GLM/Zhipu, OpenAI, Groq,
DeepSeek, OpenRouter):

```ini
ATS_LLM_PROVIDER=openai
ATS_LLM_BASE_URL=glm          # preset; expands to https://api.z.ai/api/paas/v4
ATS_LLM_MODEL=glm-4-flash
ATS_LLM_API_KEY=your-key-here
```

Base-URL presets: `ollama-cloud`, `ollama-local`, `glm`, `glm-cn`,
`openai`, `groq`, `deepseek`, `openrouter` — or paste a full URL. The two
`ollama-*` presets select Ollama's native protocol; the rest select the
OpenAI-compatible one. If you omit `ATS_LLM_PROVIDER`, it's inferred from
the URL.

`test-llm` prints the resolved config (key masked) and makes one cheap
request, so you find out about a wrong model name or expired key in two
seconds rather than mid-report. If the provider is unreachable the scorer
degrades to layers 1-partial and 2 and says so — it never silently
substitutes a fake number.

**Rotate any key that has been pasted into a chat, issue tracker, or shared
terminal.** Treat it as public from that moment.

---

## Usage — CLI

```bash
# Verify your LLM provider first (one cheap request)
python cli.py test-llm

# All three layers
python cli.py score --resume cv.pdf --jd jd.txt

# No LLM at all — layers 1 (formatting+keywords) and 2, instant and offline
python cli.py score --resume cv.pdf --jd jd.txt --offline

# Fetch the JD straight from a posting URL (HTML is stripped to text)
python cli.py score --resume cv.pdf --jd https://jobs.example.com/posting/123

# Paste the JD inline; hide the long checklist table
python cli.py score --resume cv.pdf --jd-text "paste JD..." --brief

# Machine-readable, saved to a file
python cli.py score --resume cv.pdf --jd jd.txt --json --out report.json

# Score AND log the application in one go
python cli.py score --resume cv.pdf --jd jd.txt \
    --log --company "Acme" --role "BI Analyst" --days-after-posting 2

# Score one resume against a whole folder of JDs, ranked best-first
python cli.py batch --resume cv.pdf --jds-dir jds/ --top 10 --offline
```

Supports `.pdf`, `.docx`, `.txt` resumes.

## Usage — application log

This is the part that eventually turns the scores into real percentages.

```bash
python cli.py log list                              # what you've applied to
python cli.py log outcome 3 --status recruiter_call # update when you hear back
python cli.py log export --out applications.csv     # flat CSV, Power BI ready
python cli.py log stats                             # conversion by score band
python cli.py log reap-ghosts                       # pending -> ghosted after N days
```

Outcomes: `pending`, `ghosted`, `rejected_auto`, `rejected_screen`,
`recruiter_call`, `interview`, `offer`.

`log stats` deliberately refuses to report conversion rates until you have
at least 20 resolved outcomes (`--min-resolved` to change it). A 2-of-3
sample presented as a percentage is noise dressed up as analytics. Once
you're past the threshold it shows, for each of the three layers, what
share of applications in each score band actually reached a human — real
rates from your own data — plus two analyses that used to be promises:

- **Which layer actually predicts your outcomes** — the point-biserial
  correlation between each layer's score and actually reaching a human.
  If your manager-evidence score turns out uncorrelated with callbacks
  while your recruiter score tracks tightly, that's real information
  about where your applications are dying.
- **Apply timing** — conversion rate by how long after posting you
  applied (0-2 / 3-7 / 8-14 / 15+ days), from the `days_after_posting`
  you logged.

`log reap-ghosts` marks applications still `pending` after N days
(default 45) as `ghosted` — a "pending" from two months ago is a ghost by
any realistic reading, and leaving it pending quietly corrupts your
resolved-outcome counts.

The CSV export is flat and dashboard-ready if you want to build the funnel
view in Power BI — it includes a `reached_human` 0/1 column for easy
conversion measures.

## Usage — the tailor (resume generation)

The scorer is half the system; the other half generates the resume. You
maintain ONE **evidence bank** (`master_resume.yaml`) — every real bullet
you've ever written, tagged with the skills each demonstrates, richer than
any single application needs. The tailor then:

1. **No-apply gate first.** Candidacy-fact blockers (years floor, no-sponsorship
   JDs, work-mode conflicts, mandatory degree) are checked before anything is
   generated — no rewrite fixes facts about you, so it tells you *not to spend
   the evening* instead of producing a resume that will die on the form.
   `--force` overrides, for practice or when you think the gate is wrong.
2. **Selects** bullets by greedy *marginal* value-per-line against the JD's
   weighted keywords, jointly across roles under a shared line budget: two
   bullets both covering SQL don't each earn it twice, a term's second
   mention is worth 0.3× (sustained use is evidence), its third 0.1×, and
   repetition beyond that earns nothing — reinforcement is kept when
   there's room, stuffing never pays. Every role keeps a mandatory minimum
   (your current role is never a two-bullet skeleton), within a role
   bullets are ordered strongest-first so top-third visibility is
   maximised, and — since the fix — the structure minimum itself respects
   the line budget (each role keeps at least one bullet; a long career
   history no longer silently overflows into a multi-page resume, and if
   the structural minimum still exceeds the budget the run says so).
3. **Assembles deterministically** — single column, standard headers,
   parseable `Mon YYYY - Mon YYYY` dates, email/phone/LinkedIn/GitHub in
   the header — so the layer-1 formatting score is 100 *by construction*,
   not by luck.
4. **Rewords via the LLM, under guards**: bullets that demonstrate a JD's
   phrasing under different words get translated to the JD's terms — each
   bullet is sent only its own plausible target terms, not the full
   missing list. Every rewrite must pass **fact verification** — all
   original numbers preserved (`30+` stays `30+`), no new numbers invented
   (`[X]` placeholders mark spots for *your* real figures), **every named
   tool/skill must survive** (a rewrite that swaps Power BI out for
   Tableau is rejected no matter how good it sounds), and any *new* skill
   term must be one of the explicitly allowed target terms — or it's
   rejected and the original kept.
5. **Applies the manager layer's weak-bullet rewrites** where the same
   verification passes.
6. **Reports honestly**: the three-layer score of the generated resume, the
   rewordings applied (and those rejected, with reasons), and the JD terms
   your bank genuinely doesn't cover — as gaps to fix by *gaining experience*,
   never by inventing it.

```bash
python cli.py init-master                          # create the empty bank template
python cli.py init-master --from cv.pdf            # LLM transcribes your resume into the bank
                                                    # (you review every bullet — you are the truth gate)
python cli.py tailor --master master_resume.yaml --jd jd.txt
python cli.py tailor --jd https://jobs.example.com/p/123   # JD straight from a URL
python cli.py tailor --jd jd.txt --offline         # deterministic selection only, no LLM
python cli.py tailor --jd jd.txt --force           # generate despite candidacy blockers
python cli.py tailor --jd jd.txt --log --company "Acme" --role "BI Analyst"
                                                   # generate AND log the application in one go
```

Outputs `tailored_resume.txt` + `tailored_resume.docx`, plus the full
three-layer report of the generated resume. With `pip install docx2pdf`
(and MS Word installed) a `.pdf` is written too; `--no-pdf` skips it. The
honest goal: *the best truthful resume this evidence supports for this JD*
— not a probability of selection (see "What these percentages are" above).

A workable routine: build the bank once from your best resume, then after
every real project add the new bullet to the bank. The bank compounds; each
tailor run takes a minute.

## Usage — local API + web UI (James / n8n / scripts / you)

```bash
python server.py                 # http://127.0.0.1:8420
```

Open http://127.0.0.1:8420 in a browser for the built-in single-page UI —
paste the resume and JD (or a posting URL), click Score or Tailor, and
read the same three-layer report the CLI prints, plus the applications
log. No install, no build step, served by the same Flask app.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | the web UI (score / tailor / applications) |
| GET | `/health` | liveness |
| POST | `/score` | three-layer scoring + HR expectations |
| POST | `/tailor` | generate a JD-tailored resume from the evidence bank |
| POST | `/log` | record an application |
| POST | `/outcome` | update an outcome |
| GET | `/applications` | list logged applications |
| GET | `/stats` | conversion by score band + predictiveness |

```bash
curl -X POST http://127.0.0.1:8420/score \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "...", "jd_text": "...", "offline": false,
       "log": true, "company": "Acme", "role": "BI Analyst"}'

curl -X POST http://127.0.0.1:8420/tailor \
  -H "Content-Type: application/json" \
  -d '{"jd_text": "...", "offline": true, "log": true,
       "company": "Acme", "role": "BI Analyst"}'
```

`/score` accepts `resume_path` instead of `resume_text` if the file is
local to the server, and both `/score` and `/tailor` accept `jd_url` to
fetch the posting server-side. Read `scores.*` for the numbers
(`ats_score`, `hr_screen_criteria_met_pct`, `hr_resume_fixable_pct`,
`hr_candidacy_fit_pct`, `manager_evidence_strength_pct`),
`recruiter_layer.hr_expectations` for the prep list,
`recruiter_layer.blockers` for instant-filter problems, and
`manager_layer.weak_bullets` for the rewrites. `/tailor` returns
`resume_text`, `honest_gaps`, the applied `rewordings` and
`rejected_rewrites`, and `candidacy_blockers` when the no-apply gate
blocks (`blocked: true`).

Set `OLLAMA_HOST` / `OLLAMA_MODEL` / `OLLAMA_API_KEY` before starting the
server and every request uses your cloud setup without passing them each time.

---

## Reading the output

Bands are `Strong` (80+), `Workable` (60–79), `Weak` (40–59),
`Very weak` (<40) — applied per layer, since a strong HR score and a weak
ATS score mean completely different work.

Watch for the case where **ATS goes up while HR goes down**: your keywords
match the JD better, but you're further from the screening criteria. The
machine would advance you and the recruiter wouldn't. That's a real pattern
and a single blended score would hide it entirely.

- **What HR will expect** is the section to read before a screening call.
  Items under "specific to your application" came from an actual failed or
  warned check; the rest get asked on nearly every screen.
- **Hard blockers** (red panel) are HR-layer failures on things filtered
  instantly — under the years floor, missing a mandatory degree, work-mode
  mismatch, sponsorship. Fix or accept these before optimising anything else.
- **Resume-fixable vs candidacy split** tells you whether to edit or just
  prepare. Low resume %, high candidacy % = your fit is fine, your
  presentation is burying it.
- **Formatting score** is the one to actually max out. It's the most
  objective, entirely in your control, and a failure here means nothing
  else you do matters.
- **Missing keywords** are worth adding *only where they're true of your
  experience* — usually by rewording a bullet you already have.
- **"Experience you have but word differently"** flags real experience
  that just isn't phrased the way the JD phrases it. Easiest honest fix.
- **Weak bullets with rewrites** use `[X]` placeholders where a real number
  belongs. Fill them with true figures; don't invent them.

A note on gaming it: keyword-stuffing will push layer 1 up fast, and it
will show up as a *worse* layer 3 score, because a manager reading
responsibility-lists with no evidence is exactly what that rubric
penalises. A 75 you can defend in an interview beats a 95 you can't.

---

## Limitations worth knowing

- **Multi-column PDF detection is a heuristic** (clustering word left-edge
  positions). It can false-positive on heavily indented single-column
  layouts and miss unusual two-column designs.
- **Date parsing for gaps/tenure** skips lines that look education-related
  so a degree doesn't read as a fake employment gap, but unusual date
  formats may be missed entirely — it reports what it could parse.
- **Country detection for sponsorship/right-to-work** matches a curated
  list of country and major-city names, and city names can appear in JDs
  for reasons that aren't the role's location (clients, offices). The
  check still cites its evidence lines — read them before trusting a
  pass/fail here. When no country is detected, it warns and asks you to
  verify rather than guessing.
- **JD URL fetching is plain HTTP + HTML stripping** — no JavaScript
  rendering, cookies, or session handling. Dynamic posting pages (LinkedIn
  and friends) may render thin text; that's visible immediately rather
  than silently wrong. Pasting the JD text always works.
- **JD requirement extraction is regex-based**, chosen so every finding
  cites its source line. Unusual phrasing will be missed — check the
  citations if a result looks wrong.
- **The manager layer is only as good as your model.** A small/cheap model
  gives shallower rubric scores and vaguer rewrites than a large one. If
  results look thin, try a stronger model before assuming the rubric is wrong.
- **The LLM layers were verified against a mock provider**, not a live
  hosted API — the build environment blocked outbound calls to ollama.com,
  api.z.ai and api.openai.com alike. Both protocols (Ollama native
  `/api/chat` and OpenAI-compatible `/chat/completions`), auth, parsing,
  config resolution and every failure path are tested
  (`python tests/test_llm_transport.py`, 43 checks), but run `test-llm`
  once on your own machine to confirm your key and model name work.
- **Everything deterministic is tested offline**: 82 layer checks
  (`tests/test_layers.py` — aliases, JD extraction, the new salary/location/
  authorisation checks, parsing, applog correlation, the hardened rewrite
  verifier, selector budget) and 50 generator checks
  (`tests/test_generator.py`) run with no network and no LLM.

## Project structure

```
ats-checker/
  ats_checker/
    terms.py             # shared alias machinery — ONE table for scorer + generator
    parsing.py           # PDF/DOCX/TXT extraction + parseability + link/date checks
    keywords.py          # weighted, alias-aware JD keyword extraction + matching
    jd_requirements.py   # regex extraction of JD hard filters (years, certs,
                         #   salary, location, countries), every finding cited
    jd_fetch.py          # JD-from-URL fetch + HTML->text strip (CLI + server)
    profile.py           # your fixed facts (profile.yaml)
    recruiter.py         # Layer 2: HR screen checklist + expectations
    semantic.py          # Layer 1 semantic fit (LLM)
    manager.py           # Layer 3: evidence rubric (LLM)
    llm_client.py        # provider-agnostic transport (OpenAI-compatible + Ollama)
    ollama_client.py     # back-compat shim re-exporting llm_client
    scorer.py            # runs all three layers
    applog.py            # application log + conversion/correlation stats + reaping
    report.py            # terminal rendering + JSON export
    generator/           # the tailor — builds JD-tailored resumes from the bank
      evidence_bank.py   #   master_resume.yaml model + LLM atomizer (init-master)
      selector.py       #   budget-aware greedy marginal-coverage selection
      assembler.py      #   deterministic rendering (formatting 100 by construction) + .docx
      rewriter.py       #   constrained LLM rewording + fact-preservation verifier
      optimizer.py      #   generate → score → refine loop + no-apply gate
  cli.py                 # test-llm / score / batch / log / init-profile / init-master / tailor
  server.py               # local HTTP API + web UI (score/tailor/applications)
  samples/                # example resume, JDs, profile.yaml, and master_resume.yaml
  tests/
    mock_provider.py     # fake OpenAI-compatible provider for testing
    test_llm_transport.py# end-to-end transport tests (43 checks)
    test_generator.py    # tailor tests — offline, no LLM (50 checks)
    test_layers.py       # rule-layer tests — offline, no LLM (82 checks)
    smoke_server.py      # HTTP API smoke test (run with the server up)
  .env.example           # copy to .env and add your key
```
