# 🧭 ats-system — three-layer resume scoring + JD-tailored resume generator

A local tool that scores your resume against a job description the way the
hiring pipeline actually filters you: **machine first, recruiter second,
hiring manager third**. Runs on your own machine against your own LLM
(Ollama local/cloud or any OpenAI-compatible API). No third-party resume
site, nothing uploaded anywhere.

> **Honesty-first design:** the tailor never invents. Every claim traces to
> a real bullet in your evidence bank. JD terms you can't back are reported
> as gaps to fix by gaining experience — never stuffed in.

## The three layers

Your resume passes three readers with three different criteria, and it can
fail at any one for reasons the others don't care about. So the tool reports
**three separate scores** and never blends them:

**Layer 1 — ATS Score.** The machine filter. Parseability (two-column
layouts, tables, missing headers, failed text extraction are the #1 silent
rejection cause), alias-aware keyword overlap against the JD's hard
requirements (one shared alias table for scorer and generator, so the two
can never disagree), and LLM-judged semantic fit.

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

## Quick start

```bash
git clone https://github.com/raghavsaiassistant-wq/ats-system.git
cd ats-system
pip install -r requirements.txt

# Configure your LLM (Ollama Cloud, local Ollama, or any OpenAI-compatible API)
cp .env.example .env     # then edit: provider, base_url, model, api_key

python cli.py test-llm          # verify LLM connectivity in one request
python cli.py init-profile      # create profile.yaml — fill in once
python cli.py init-master --from cv.pdf   # LLM transcribes your resume into the evidence bank

# Score any resume against any JD
python cli.py score --resume cv.pdf --jd jd.txt

# Generate a JD-tailored resume from your evidence bank
python cli.py tailor --master master_resume.yaml --jd jd.txt
```

No LLM configured? The two judgment layers can still be scored: print the
exact prompts, paste them into any chat LLM, and feed its JSON replies back.

```bash
python cli.py prompts --resume cv.pdf --jd jd.txt      # prints both prompts
# save replies as judgments.json: {"semantic": {...}, "manager": {...}}
python cli.py score --resume cv.pdf --jd jd.txt --judgments judgments.json
```

Supports `.pdf`, `.docx`, `.txt` resumes. Web UI: `python server.py` →
http://127.0.0.1:8420

## What these percentages are (and are not)

They are **percentages of things measured** — "you meet 78% of their stated
screening criteria" — not probabilities of passing. Your rank relative to
the applicant pool, whether a human opens your file, and req reality are
invisible to any resume tool. Use the scores to fix what's fixable.

## The application log

`log` records every application and outcome; `log stats` refuses to show
conversion rates until you have 20 resolved outcomes (a 2-of-3 sample as a
percentage is noise). Past that threshold it shows, from your own data:
which layer actually predicts your outcomes, and conversion by
apply-timing.

## Privacy

Everything runs locally. Your resume, JD, and evidence bank never leave
your machine except as prompts to **your own configured LLM**. No
telemetry, no accounts, no uploads.

## License

MIT