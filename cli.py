#!/usr/bin/env python3
"""ATS Score Checker — CLI.

    python cli.py init-profile                                  # set up profile.yaml once
    python cli.py score --resume cv.pdf --jd jd.txt             # all three layers
    python cli.py score --resume cv.pdf --jd jd.txt --offline   # no LLM, layers 1+2 only
    python cli.py score --resume cv.pdf --jd https://...        # fetch the JD from a URL
    python cli.py batch --resume cv.pdf --jds-dir jds/          # score many JDs, ranked
    python cli.py score --resume cv.pdf --jd jd.txt --log       # company/role default from the JD
    python cli.py score --resume cv.pdf --jd jd.txt --log --company "Acme" --role "BI Analyst"
    python cli.py log list
    python cli.py log outcome 3 --status interview
    python cli.py log outcome --last --status recruiter_call      # the one you just logged
    python cli.py log outcome acme --status rejected_auto         # fuzzy company match
    python cli.py log export --out applications.csv
    python cli.py log stats
    python cli.py log reap-ghosts                               # pending -> ghosted after N days

    python cli.py init-master --from cv.pdf                     # build the evidence bank once
    python cli.py tailor --jd jd.txt                            # generate a JD-tailored resume
    python cli.py tailor --jd jd.txt --offline                  # deterministic selection only
    python cli.py tailor --jd jd.txt --log                      # generate AND log the application
                                                                # (ATS_AUTO_LOG=1 makes --log the default)

    python cli.py optimize --jd jd.txt                          # tailor, score, improve, repeat
    python cli.py run --resume cv.pdf --jd https://...          # everything, end to end
    python cli.py optimize --jd jd.txt --provider manual        # no API key: paste prompts into
                                                                # Claude / ChatGPT instead
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ats_checker import applog
from ats_checker import manual_llm
from ats_checker import llm_client as ollama_client
from ats_checker import profile as profile_mod
from ats_checker import report as report_mod
from ats_checker.scorer import run_full_check
from rich.markup import escape
from ats_checker import generator as gen_mod


def _read_jd(args) -> str:
    if args.jd:
        if args.jd.lower().startswith(("http://", "https://")):
            print(f"Fetching JD from {args.jd} ...")
            from ats_checker.jd_fetch import fetch_jd_url

            try:
                return fetch_jd_url(args.jd)
            except Exception as e:  # noqa: BLE001 — network errors reach the user directly
                print(f"Could not fetch the JD URL: {e}", file=sys.stderr)
                raise SystemExit(1)
        p = Path(args.jd)
        if not p.exists():
            print(f"JD file not found: {args.jd}", file=sys.stderr)
            raise SystemExit(1)
        return p.read_text(encoding="utf-8", errors="ignore")
    return args.jd_text


PROVIDERS = ["openai", "ollama", "manual"]
MANUAL_HELP = ("Copy-paste mode (provider manual): a file of answers you pasted back from a "
               "chat AI. Answers are cached, so a re-run needs no paste.")


def _provider_of(args) -> str:
    return (getattr(args, "provider", None) or ollama_client.current_config()["provider"]).lower()


def _interactive() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _run_manual_captured(args) -> int:
    """Copy-paste mode for one-shot commands (score, tailor, eval,
    init-master): run the command, and while LLM prompts are pending, bundle
    them, get the answers pasted (or from --answers), and run it again. Only
    the final, complete run's output is shown."""
    import contextlib
    import io

    def once():
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = args.func(args)
        return rc, out.getvalue(), err.getvalue()

    (rc, out, err), todo = manual_llm.run_with_answers(
        once, answers_file=getattr(args, "answers", None), interactive=_interactive())
    if todo:
        # this pass ran with answers missing: its output would pass off
        # fallback numbers as the real thing, so it isn't shown
        return _manual_unfinished(todo)
    sys.stdout.write(out)
    sys.stderr.write(err)
    return rc


def _manual_unfinished(todo) -> int:
    if not todo:
        return 0
    print(f"\n{len(todo)} prompt(s) are still waiting for an answer. Paste the bundle into a chat "
          "AI, save its JSON reply to a file, and re-run the same command with "
          "--answers <that file>.", file=sys.stderr)
    return 4


def cmd_init_profile(args) -> int:
    try:
        path = profile_mod.write_template(args.path, overwrite=args.force)
    except FileExistsError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"Created {path}")
    print("Fill it in — the recruiter-screen layer skips any check whose field is blank.")
    return 0


def cmd_test_llm(args) -> int:
    cfg = ollama_client.current_config()
    provider = args.provider or cfg["provider"]
    host = args.host or cfg["base_url"]
    model = args.model or cfg["model"]
    api_key = args.api_key or cfg["api_key"]

    masked = (api_key[:6] + "…" + api_key[-4:]) if len(api_key) > 12 else ("set" if api_key else "NOT SET")
    print("Resolved configuration:")
    print(f"  provider : {provider}")
    print(f"  base_url : {host}")
    print(f"  model    : {model}")
    print(f"  api_key  : {masked}")
    print("\nSending one test request...")

    ok, message = ollama_client.test_connection(
        model=model, host=host, api_key=api_key, provider=provider
    )
    if ok:
        print(f"\nOK — {message}")
        print("Both LLM layers (semantic fit, manager evidence) will work.")
        return 0

    print(f"\nFAILED — {message}")
    print("\nCommon fixes:")
    print("  401  -> wrong or expired key in .env (ATS_LLM_API_KEY)")
    print("  404  -> wrong base URL or a model name your key can't access (ATS_LLM_MODEL)")
    print("  connection refused -> for Ollama, run `ollama serve`; for a hosted API, check the URL")
    print("\nThe scorer still runs without this — use `score --offline` for layers 1 and 2.")
    return 1


def cmd_score(args) -> int:
    if not Path(args.resume).exists():
        print(f"Resume file not found: {args.resume}", file=sys.stderr)
        return 1

    jd_text = _read_jd(args)
    prof = profile_mod.load_profile(args.profile)

    skip_llm = args.offline
    result = run_full_check(
        resume_path=args.resume,
        jd_text=jd_text,
        profile=prof,
        model=args.model,
        host=args.host,
        api_key=args.api_key,
        provider=args.provider,
        skip_semantic=skip_llm or args.no_semantic,
        skip_manager=skip_llm or args.no_manager,
        jd_extractor="rules" if skip_llm else args.jd_extractor,
    )

    if args.json:
        print(report_mod.to_json(result))
    else:
        report_mod.print_report(result, show_checks=not args.brief)

    if args.out:
        Path(args.out).write_text(report_mod.to_json(result), encoding="utf-8")
        if not args.json:
            print(f"\nSaved JSON report to {args.out}")

    if _should_log(args):
        app_id = _log_scored(args, result, jd_text, Path(args.resume).name)
        if app_id is None:
            return 1
        if not args.json:
            print(f"\nLogged as application #{app_id}. When you hear back:")
            print("  python cli.py log outcome --last --status recruiter_call")

    return 0


AUTO_LOG_ENV = "ATS_AUTO_LOG"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _should_log(args) -> bool:
    """--log / --no-log win; otherwise ATS_AUTO_LOG=1 (env or .env) turns
    logging on by default. Off unless asked: a practice run against a JD you
    won't apply to shouldn't pollute your conversion stats."""
    if getattr(args, "no_log", False):
        return False
    if manual_llm.pending():   # copy-paste pass that will be re-run: don't log twice
        return False
    if getattr(args, "log", False):
        return True
    return _env_flag(AUTO_LOG_ENV)


def _log_scored(args, report, jd_text: str, resume_version: str) -> int | None:
    """Log a scored application. Company/role default from the JD (role =
    the JD title; company = a 'Company:'/'About X' line or the posting URL's
    host) when --company/--role aren't given. None (with a message) when
    they can't be worked out."""
    jd_url = args.jd if args.jd and args.jd.lower().startswith(("http://", "https://")) else None
    if report is not None:
        jd_title = report.jd_reqs.jd_title
    else:
        from ats_checker import jd_requirements

        jd_title = jd_requirements.extract(jd_text).jd_title
    company_guess, role_guess = applog.default_company_role(jd_text, jd_url, jd_title)
    company = args.company or company_guess
    role = args.role or role_guess
    if not company or not role:
        missing = " and ".join(f"--{k}" for k, v in (("company", company), ("role", role)) if not v)
        print(f"\nNot logged: couldn't tell the {missing.replace('--', '')} from the JD — "
              f"pass {missing}.", file=sys.stderr)
        return None
    guessed = [f"{k} '{v}'" for k, v, given in (("company", company, args.company),
                                               ("role", role, args.role)) if not given]
    if guessed and not getattr(args, "json", False):
        print(f"\nUsing {' and '.join(guessed)} from the JD (override with --company/--role).")
    return applog.log_application(
        company=company,
        role=role,
        ats_score=report.ats_score if report else None,
        visibility_score=report.visibility.score if report and report.visibility else None,
        recruiter_score=report.recruiter_score if report else None,
        manager_score=report.manager_score if report else None,
        jd_text=jd_text,
        resume_version=resume_version,
        days_after_posting=args.days_after_posting,
        db_path=args.db,
        report=report,
    )


def cmd_log_list(args) -> int:
    apps = applog.list_applications(limit=args.limit, db_path=args.db)
    if not apps:
        print("No applications logged yet.")
        return 0
    print(f"{'ID':<4} {'Applied':<12} {'Company':<20} {'Role':<24} {'VIS':>5} {'REC':>5} {'MGR':>5}  Outcome")
    print("-" * 100)
    for a in apps:
        def fmt(v):
            return f"{v:.0f}" if v is not None else "-"
        print(
            f"{a.id:<4} {a.applied_date:<12} {a.company[:19]:<20} {a.role[:23]:<24} "
            f"{fmt(a.visibility_score):>5} {fmt(a.recruiter_score):>5} {fmt(a.manager_score):>5}  {a.outcome}"
        )
    return 0


def _resolve_outcome_target(args) -> int | None:
    """Which application `log outcome` means: an id, --last, or a company
    name (fuzzy). None, with a message, when it's missing or ambiguous."""
    target = (args.target or "").strip()
    if args.last:
        if target:
            print("Give an id/company OR --last, not both.", file=sys.stderr)
            return None
        app_id = applog.last_application_id(db_path=args.db)
        if app_id is None:
            print("No applications logged yet.", file=sys.stderr)
        return app_id
    if not target:
        print("Say which application: an id, a company name, or --last.", file=sys.stderr)
        return None
    if target.isdigit():
        return int(target)
    matches = applog.find_applications(target, db_path=args.db)
    if len(matches) > 1:
        pending = [a for a in matches if a.outcome == "pending"]
        matches = pending if len(pending) == 1 else matches
    if not matches:
        print(f"No logged application matches company '{target}'.", file=sys.stderr)
        return None
    if len(matches) > 1:
        print(f"'{target}' matches {len(matches)} applications — use the id:", file=sys.stderr)
        for a in matches[:10]:
            print(f"  #{a.id:<4} {a.applied_date}  {a.company} — {a.role}  ({a.outcome})",
                  file=sys.stderr)
        return None
    return matches[0].id


def cmd_log_outcome(args) -> int:
    app_id = _resolve_outcome_target(args)
    if app_id is None:
        return 1
    try:
        ok = applog.set_outcome(app_id, args.status, notes=args.notes, db_path=args.db)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    if not ok:
        print(f"No application with id {app_id}", file=sys.stderr)
        return 1
    app = next((a for a in applog.list_applications(limit=1_000_000, db_path=args.db)
                if a.id == app_id), None)
    label = f" ({app.company} — {app.role})" if app else ""
    print(f"Application #{app_id}{label} → {args.status}")
    return 0


def cmd_log_export(args) -> int:
    path, n = applog.export_csv(args.out, db_path=args.db)
    print(f"Exported {n} application(s) to {path}")
    return 0


def cmd_log_stats(args) -> int:
    stats = applog.conversion_stats(db_path=args.db, min_resolved=args.min_resolved,
                                    reap_days=None if args.no_reap else args.reap_days)
    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0
    report_mod.print_stats(stats)
    return 0


def cmd_log_reap(args) -> int:
    n = applog.reap_ghosts(db_path=args.db, days=args.days)
    print(f"Marked {n} pending application(s) older than {args.days} days as ghosted.")
    return 0


def cmd_batch(args) -> int:
    """Score one resume against a whole folder of JDs and rank the results
    by ATS score — triage for a job-search inbox."""
    if not Path(args.resume).exists():
        print(f"Resume file not found: {args.resume}", file=sys.stderr)
        return 1
    jd_dir = Path(args.jds_dir)
    if not jd_dir.is_dir():
        print(f"JD folder not found: {args.jds_dir}", file=sys.stderr)
        return 1

    jd_files = sorted([p for p in jd_dir.iterdir()
                       if p.suffix.lower() in (".txt", ".md") and p.is_file()])
    if not jd_files:
        print(f"No .txt/.md JD files in {jd_dir}", file=sys.stderr)
        return 1

    prof = profile_mod.load_profile(args.profile)
    rows = []
    for f in jd_files:
        jd_text = f.read_text(encoding="utf-8", errors="ignore")
        try:
            rep = run_full_check(
                resume_path=args.resume, jd_text=jd_text, profile=prof,
                model=args.model, host=args.host, api_key=args.api_key, provider=args.provider,
                skip_semantic=args.offline, skip_manager=args.offline,
                jd_extractor="rules" if args.offline else args.jd_extractor,
            )
        except Exception as e:  # noqa: BLE001 — one bad JD shouldn't kill the batch
            print(f"  skipping {f.name}: {e}", file=sys.stderr)
            continue
        vis = rep.visibility
        rows.append((f.name, vis.score if vis else None, rep.recruiter_score, rep.manager_score,
                     vis.parse_safe if vis else None))

    # rank by search visibility — would a recruiter's search find you for this JD?
    rows.sort(key=lambda r: (r[1] is not None, r[1] or 0), reverse=True)
    if args.top:
        rows = rows[: args.top]

    if args.json:
        import json
        print(json.dumps([
            {"jd": n, "search_visibility_pct": v, "hr_screen_criteria_met_pct": h,
             "manager_evidence_strength_pct": m, "parse_safe": p}
            for n, v, h, m, p in rows
        ], indent=2))
        return 0

    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(title=f"Ranked JDs for {escape(Path(args.resume).name)} ({len(rows)} scored)",
                  show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("JD file")
    table.add_column("Visibility", justify="right")
    table.add_column("HR Screen", justify="right")
    table.add_column("Manager", justify="right")

    def _fmt(score):
        if score is None:
            return "[dim]—[/dim]"
        c = "green" if score >= 80 else "yellow" if score >= 60 else "red"
        return f"[{c}]{score:.0f}[/{c}]"

    for i, (name, vis, hr, mgr, safe) in enumerate(rows, 1):
        table.add_row(str(i), escape(name), _fmt(vis), _fmt(hr), _fmt(mgr))
    console.print(table)
    console.print("[dim]Bands: Strong 80+ · Workable 60-79 · Weak 40-59 · Very weak <40 — "
                  "applied per layer. High visibility + low HR means a recruiter's search finds "
                  "you but their screen wouldn't pass you: read the full report before "
                  "applying anyway.[/dim]")
    return 0


def cmd_eval(args) -> int:
    """Score the JD extractor against the labelled corpus (tests/jd_corpus)."""
    import json

    from ats_checker import evaluation as ev

    items = ev.load_corpus(args.corpus)
    if not items:
        print(f"No corpus files in {args.corpus}", file=sys.stderr)
        return 1
    bad = [(i.id, p) for i in items for p in ev.validate_item(i)]
    if bad:
        for item_id, problem in bad:
            print(f"{item_id}: {problem}", file=sys.stderr)
        return 1
    extractor = ev.EXTRACTORS[args.extractor]
    if args.extractor in ev.LLM_EXTRACTORS:
        extractor = ev.make_llm_extractor(
            args.extractor, model=args.model, host=args.host, api_key=args.api_key,
            provider=args.provider, use_cache=not args.no_cache)
    try:
        report = ev.evaluate(items, extractor, name=args.extractor)
    except RuntimeError as e:
        print(f"{e}\nThe {args.extractor!r} extractor needs a reachable LLM — check it with "
              "`python cli.py test-llm`.", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.metrics(), indent=2))
    else:
        ev.print_report(report, show_misses=not args.brief)
    if args.write_baseline:
        Path(args.write_baseline).write_text(
            json.dumps(report.metrics(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote baseline to {args.write_baseline}")
    return 0


JD_EXTRACTOR_HELP = ("How to read the JD. rules (default): the deterministic engine. llm: the "
                     "LLM's reading, kept only where it quotes the JD line it came from, merged "
                     "over the rules (falls back to rules if the LLM is unreachable).")


def cmd_learned_list(args) -> int:
    from ats_checker import learned

    rows = learned.candidates()
    if not rows:
        print(f"No candidate terms yet ({learned.learned_path()}). They are recorded when "
              "`score --jd-extractor llm` finds a verified skill the taxonomy doesn't know.")
        return 0
    print(f"Candidates in {learned.learned_path()} (promote only real, searchable skills):\n")
    for term, count, example in rows:
        print(f"  {count:>3}x  {term:<28} e.g. {example.strip()[:70]}")
    return 0


def cmd_learned_promote(args) -> int:
    from ats_checker import learned

    if learned.promote(args.term):
        print(f"Promoted {args.term.lower()!r}: the rule engine will extract it from now on.")
    else:
        print(f"{args.term.lower()!r} is already approved.")
    return 0


def cmd_learned_reject(args) -> int:
    from ats_checker import learned

    if learned.reject(args.term):
        print(f"Dropped candidate {args.term.lower()!r}.")
        return 0
    print(f"No candidate named {args.term.lower()!r}.", file=sys.stderr)
    return 1


def cmd_init_master(args) -> int:
    try:
        if args.from_resume:
            path, notes = gen_mod.init_from_resume(
                resume_path=args.from_resume,
                out_path=args.path,
                model=args.model, host=args.host, api_key=args.api_key, provider=args.provider,
                overwrite=args.force,
            )
            print(f"Created evidence bank at {path} (transcribed by the LLM from your resume).")
            print("\n".join(f"- {n}" for n in notes) if notes else "")
            print("\nNOW REVIEW EVERY BULLET. Each one is marked `reviewed: false` and the tailor")
            print("skips it until you confirm it: easiest in the web UI (python server.py ->")
            print("Setup tab), or delete the `reviewed: false` line of each bullet you've checked.")
            print("The bank is the truth constraint — anything wrong here propagates into every")
            print("tailored resume. Add bullets over time; keep it richer than any one application.")
        else:
            path = gen_mod.write_template(args.path, overwrite=args.force)
            print(f"Created {path}")
            print("Fill it in with your real bullets (verbatim, numbers kept) and skill tags.")
            print("Every bullet you've ever written belongs here — the tailor selects from")
            print("this bank and never invents.")
    except FileExistsError as e:
        print(str(e), file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


def cmd_tailor(args) -> int:
    jd_text = _read_jd(args)
    prof = profile_mod.load_profile(args.profile)

    try:
        result = gen_mod.tailor(
            master_path=args.master,
            jd_text=jd_text,
            profile=prof,
            offline=args.offline,
            force=args.force,
            max_current=args.max_current,
            max_other=args.max_other,
            max_lines=args.max_lines,
            model=args.model, host=args.host, api_key=args.api_key, provider=args.provider,
        )
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 1

    from rich.console import Console
    from rich.panel import Panel
    console = Console()

    for note in result.bank_problems:
        console.print(f"[yellow]Bank issue:[/yellow] {escape(note)}")
    for note in result.notes:
        console.print(f"[dim]Note: {escape(note)}[/dim]")

    if result.blocked:
        console.print(Panel(
            "\n".join(f"- {escape(b)}" for b in result.candidacy_blockers),
            title="[bold red]No-apply gate: candidacy blockers[/bold red]",
            border_style="red",
        ))
        console.print(
            "These are facts about you, not the resume — no rewrite fixes them.\n"
            "Re-run with --force to generate anyway (practice, or you think the gate is wrong)."
        )
        return 2

    # ---- write outputs
    out_txt = Path(args.out)
    out_txt.write_text(result.resume_text, encoding="utf-8")
    docx_path = None
    pdf_path = None
    if not args.no_docx:
        docx_path = out_txt.with_suffix(".docx")
        gen_mod.write_docx(result.resume_text, str(docx_path))
        if not args.no_pdf:
            pdf_path = _try_write_pdf(docx_path, console)

    # ---- final three-layer report of the generated resume
    if result.report is not None:
        report_mod.print_report(result.report, show_checks=not args.brief)

        if result.baseline_keyword_pct is not None:
            console.print(Panel(
                f"Keyword match: deterministic assembly {result.baseline_keyword_pct}%"
                + (f"  →  final {result.final_keyword_pct}%" if result.final_keyword_pct is not None else "")
                + (f"  (LLM rewording applied: {len(result.rewordings)})" if result.rewordings else ""),
                title="Generation trace", border_style="cyan",
            ))

    if result.honest_gaps:
        console.print(Panel(
            "\n".join(f"- {escape(g)}" for g in result.honest_gaps),
            title="[bold]Honest gaps — JD terms your evidence bank doesn't cover[/bold]",
            border_style="yellow",
        ))
        console.print("[dim]These are NOT added — a keyword you can't defend is an interview "
                      "trap. Gain the experience, then add the bullet to your bank.[/dim]")

    if result.rewordings:
        lines = []
        for rw in result.rewordings:
            lines.append(f"[cyan]{escape(rw['source'])}[/cyan]: {escape(rw['reason'])}")
            lines.append(f"  [dim]-[/dim] {escape(rw['original'])}")
            lines.append(f"  [green]+[/green] {escape(rw['rewrite'])}\n")
        console.print(Panel("\n".join(lines).rstrip(),
                             title="Verified rewordings applied", border_style="green"))
    if result.rejected_rewrites:
        lines = [f"- suggested: {escape(r['suggested'])}\n  rejected: {escape(r['reason'])}"
                 for r in result.rejected_rewrites]
        console.print(Panel("\n".join(lines),
                             title="Suggested but rejected (failed fact verification)",
                             border_style="red"))

    console.print(Panel(
        f"Resume text : {escape(str(out_txt.resolve()))}"
        + (f"\nWord format : {escape(str(docx_path.resolve()))}" if docx_path else "")
        + (f"\nPDF format  : {escape(str(pdf_path.resolve()))}" if pdf_path else "")
        + "\nEvery line traces to an evidence-bank bullet — check them against your bank "
          "before sending. Fill any \\[X] placeholders with your real numbers.",
        title="Output", border_style="bold",
    ))

    # ---- optionally log this application, same as `score --log`
    if _should_log(args):
        app_id = _log_scored(args, result.report, jd_text, f"tailored:{out_txt.name}")
        if app_id is None:
            return 1
        console.print(f"[dim]Logged as application #{app_id}. When you hear back:[/dim]")
        console.print("[dim]  python cli.py log outcome --last --status recruiter_call[/dim]")

    return 0



def _read_jd_or_ask(args) -> str:
    """Like _read_jd, but a job link that can't be fetched falls back to
    asking for the JD text (pasted at the terminal)."""
    if args.jd and args.jd.lower().startswith(("http://", "https://")):
        from ats_checker.jd_fetch import fetch_jd_url

        try:
            text = fetch_jd_url(args.jd)
            if len(text.split()) >= 40:
                return text
            problem = "the page had almost no text (it may need JavaScript)"
        except Exception as e:  # noqa: BLE001 — any fetch failure falls back to pasting
            problem = str(e)
        print(f"Couldn't read the job link: {problem}", file=sys.stderr)
        if not _interactive():
            print("Save the JD text to a file and pass --jd jd.txt (or --jd-text).", file=sys.stderr)
            raise SystemExit(1)
        print("Paste the job description, then a line with just END:", file=sys.stderr)
        lines = []
        for line in sys.stdin:
            if line.strip() == "END":
                break
            lines.append(line)
        text = "".join(lines).strip()
        if not text:
            raise SystemExit(1)
        return text
    return _read_jd(args)


def _optimize_kwargs(args, jd_text, prof) -> dict:
    return dict(master_path=args.master, jd_text=jd_text, profile=prof, offline=args.offline,
                force=args.force, max_rounds=args.max_rounds, max_current=args.max_current,
                max_other=args.max_other, max_lines=args.max_lines, model=args.model,
                host=args.host, api_key=args.api_key, provider=args.provider)


def _run_optimize(args, jd_text, prof):
    """optimize(), in copy-paste mode looped until no prompt is pending.
    Returns (result | None, pending_prompts)."""
    from ats_checker.generator import loop

    kwargs = _optimize_kwargs(args, jd_text, prof)
    if _provider_of(args) == "manual" and not args.offline:
        return manual_llm.run_with_answers(lambda: loop.optimize(**kwargs),
                                           answers_file=args.answers, interactive=_interactive())
    return loop.optimize(**kwargs), []


def _review_proposals(res, args, console) -> str:
    """Decide which reworded bullets go into the final file. Returns a
    short line describing what happened."""
    from ats_checker.generator import loop

    if not res.proposals:
        loop.finalize(res)
        return ""
    if args.auto_accept:
        loop.finalize(res, auto_accept=True)
        return (f"--auto-accept: all {len(res.proposals)} reworded bullet(s) went in UNREVIEWED. "
                "Riskier: read each one before you send this resume.")
    if args.apply_proposals:
        data = json.loads(Path(args.apply_proposals).read_text(encoding="utf-8"))
        wanted = {(d.get("original"), d.get("rewrite")) for d in data if d.get("accepted") is True}
        ids = {p.id for p in res.proposals if (p.original, p.rewrite) in wanted}
        loop.finalize(res, accepted_ids=ids)
        stale = len(wanted) - len(ids)
        return (f"Applied {len(ids)} accepted rewording(s) from {args.apply_proposals}"
                + (f"; {stale} no longer match this run and were skipped." if stale else "."))
    if _interactive() and not args.json:
        console.print("\n[bold]Review the reworded bullets[/bold] — accept only what is true and "
                      "you could defend in an interview.")
        ids = set()
        for p in res.proposals:
            console.print(f"\n[cyan]{escape(p.company)} — {escape(p.title)}[/cyan] "
                          f"[dim]({escape(p.source)}: {escape(p.reason)})[/dim]")
            console.print(f"  [dim]-[/dim] {escape(p.original)}")
            console.print(f"  [green]+[/green] {escape(p.rewrite)}")
            if input("  Accept this rewording? [y/N] ").strip().lower() in ("y", "yes"):
                ids.add(p.id)
        loop.finalize(res, accepted_ids=ids)
        return f"You accepted {len(ids)} of {len(res.proposals)} reworded bullet(s)."
    loop.finalize(res, accepted_ids=set())
    prop_path = Path(args.out).with_suffix(".proposals.json")
    prop_path.write_text(json.dumps([p.to_dict() for p in res.proposals], indent=2,
                                    ensure_ascii=False), encoding="utf-8")
    return (f"{len(res.proposals)} reworded bullet(s) are PROPOSED, not applied: the resume uses "
            f"your original wording. To use some, set \"accepted\": true in {prop_path} and "
            f"re-run with --apply-proposals {prop_path} (or use --auto-accept).")


def _print_optimize(res, review_note, out_paths, console) -> None:
    from rich.panel import Panel
    from rich.table import Table

    t = Table(title="Optimize — score per round (offline: visibility, keywords, HR resume checks)",
              show_header=True, header_style="bold")
    for col in ("Round", "Score", "Gain", "Visibility", "Keywords", "HR (resume)", "Changes"):
        t.add_column(col, justify="right" if col != "Changes" else "left")

    def f(v):
        return "—" if v is None else f"{v:.1f}"

    for r in res.rounds:
        c = r.components
        t.add_row(str(r.round), f(r.score), f"{r.gain:+.1f}" if r.round else "",
                  f(c.get("search_visibility")), f(c.get("keyword_match")),
                  f(c.get("hr_resume_fixable")), str(len(r.changes)) if r.round else "draft")
    console.print(t)
    console.print(f"[dim]Stopped: {escape(res.stop_reason)}. Best {f(res.best_score)}; the file you "
                  f"get scores {f(res.final_score)} (your accepted rewordings only).[/dim]")

    diff = []
    for r in res.rounds[1:]:
        for ch in r.changes:
            diff.append(f"[bold]Round {r.round}[/bold]: {escape(ch['what'])} "
                        f"[dim]({escape(ch.get('why', ''))})[/dim]")
            if ch.get("out"):
                diff.append(f"  [red]-[/red] {escape(ch['out'])}")
            if ch.get("in"):
                diff.append(f"  [green]+[/green] {escape(ch['in'])}")
    if diff:
        console.print(Panel("\n".join(diff), title="What changed, and why", border_style="cyan"))
    if review_note:
        console.print(f"[yellow]{escape(review_note)}[/yellow]")
    if "[X]" in res.resume_text:
        console.print("[bold yellow]Fill every \\[X] placeholder with your real number before "
                      "sending, or reject that rewording.[/bold yellow]")

    if res.gaps:
        g = Table(title="Gaps the truth can't close (yet)", show_header=True, header_style="bold")
        g.add_column("JD term")
        g.add_column("Required", justify="center")
        g.add_column("Your evidence")
        label = {"nothing-in-bank": "[red]nothing in your bank[/red]",
                 "unconfirmed": "[yellow]only an unconfirmed bullet[/yellow]",
                 "not-shown": "[cyan]in your bank, didn't fit[/cyan]"}
        for gap in res.gaps[:20]:
            g.add_row(escape(gap.term), "yes" if gap.required else "", label[gap.status])
        console.print(g)
        console.print("[dim]Nothing was added for these. 'Nothing in your bank' gaps close only "
                      "with real experience — then add the bullet to your bank.[/dim]")
    for note in res.bank_problems + res.notes:
        console.print(f"[dim]Note: {escape(note)}[/dim]")
    console.print(Panel("\n".join(f"{k:12}: {escape(str(v))}" for k, v in out_paths.items()),
                        title="Output", border_style="bold"))
    console.print("[dim]Scores are percentages of things measured, not a chance of selection — "
                  "that also depends on the other applicants and the interviews.[/dim]")


def _finish_optimize(args, res, pending) -> int:
    from rich.console import Console

    console = Console()
    if res is None:
        return _manual_unfinished(pending) or 1
    if res.blocked:
        if args.json:
            print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
        else:
            console.print("[bold red]No-apply gate: candidacy blockers[/bold red]")
            for b in res.candidacy_blockers:
                console.print(f"- {escape(b)}")
            console.print("These are facts about you, not the resume. Re-run with --force to "
                          "optimize anyway.")
        return 2
    review_note = _review_proposals(res, args, console)
    out_txt = Path(args.out)
    out_txt.write_text(res.resume_text, encoding="utf-8")
    paths = {"Resume text": str(out_txt.resolve())}
    if not args.no_docx:
        docx_path = out_txt.with_suffix(".docx")
        gen_mod.write_docx(res.resume_text, str(docx_path))
        paths["Word format"] = str(docx_path.resolve())
    if args.json:
        d = res.to_dict()
        d["review"], d["outputs"] = review_note, paths
        print(json.dumps(d, indent=2, ensure_ascii=False))
    else:
        _print_optimize(res, review_note, paths, console)
    return _manual_unfinished(pending)


def cmd_optimize(args) -> int:
    jd_text = _read_jd_or_ask(args)
    prof = profile_mod.load_profile(args.profile)
    try:
        res, pending = _run_optimize(args, jd_text, prof)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return _finish_optimize(args, res, pending)


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _setup_profile_interactively(path: str) -> None:
    print("\nNo profile yet — a few questions (Enter skips one; its check is then skipped):")
    data = {
        "current_title": _ask("  Current / most recent job title: "),
        "years_experience": _ask("  Years of relevant experience: "),
        "education_level": _ask("  Highest degree (high_school/diploma/bachelors/masters/mba/phd): "),
        "location": _ask("  Where you live (City, Country): "),
        "open_to_relocation": _ask("  Open to relocation? (yes/no): "),
        "acceptable_work_modes": _ask("  Work modes you'd accept (remote, hybrid, onsite): "),
        "notice_period_days": _ask("  Notice period in days: "),
        "work_authorized_in": _ask("  Countries you can work in without sponsorship: "),
    }
    prof = profile_mod.profile_from_dict(data)
    errors = profile_mod.validate_profile(prof)
    if errors:
        print(f"  Skipped saving the profile: {'; '.join(errors)}. Fix it in the web UI's Setup "
              "tab or with init-profile.")
        return
    print(f"  Saved {profile_mod.save_profile(prof, path)}")


def _review_bank_interactively(master: str) -> int:
    """y = confirm, n = leave unreviewed, d = delete, q = stop. Returns how
    many bullets are still unreviewed."""
    bank = gen_mod.load_bank(master)
    todo = [(r, b) for r in bank.roles for b in r.bullets if not b.reviewed]
    if not todo:
        return 0
    print(f"\n{len(todo)} bullet(s) were transcribed by the LLM and are UNREVIEWED. Confirm only "
          "what is true and you could defend in an interview.")
    for role, b in todo:
        print(f"\n  {role.title} — {role.company}\n    {b.text}")
        a = _ask("  [y] true  [n] not now  [d] delete  [q] stop: ").lower()
        if a == "y":
            b.reviewed = True
        elif a == "d":
            role.bullets.remove(b)
        elif a == "q":
            break
    gen_mod.save_bank(bank, master)
    return bank.unreviewed_count()


def cmd_run(args) -> int:
    """resume + JD (file or link) -> profile/bank if missing -> optimize -> files."""
    if not Path(args.resume).exists():
        print(f"Resume file not found: {args.resume}", file=sys.stderr)
        return 1
    interactive = _interactive()

    if not Path(args.profile).exists():
        if interactive:
            _setup_profile_interactively(args.profile)
        else:
            print(f"No {args.profile}: recruiter checks that need your facts will be skipped. "
                  "Set it up with `python cli.py init-profile` or the web UI's Setup tab.",
                  file=sys.stderr)

    if not Path(args.master).exists():
        if args.offline:
            print("No evidence bank yet, and --offline means no LLM to transcribe your resume. "
                  "Run without --offline (or with --provider manual), or build the bank in the "
                  "web UI's Setup tab.", file=sys.stderr)
            return 1
        print(f"No evidence bank yet: transcribing {args.resume} into {args.master} ...",
              file=sys.stderr)

        def transcribe():
            try:
                return gen_mod.init_from_resume(
                    resume_path=args.resume, out_path=args.master, model=args.model,
                    host=args.host, api_key=args.api_key, provider=args.provider)
            except RuntimeError as e:
                return e

        if _provider_of(args) == "manual":
            out, pending = manual_llm.run_with_answers(transcribe, answers_file=args.answers,
                                                       interactive=interactive)
            if pending:
                return _manual_unfinished(pending)
        else:
            out = transcribe()
        if isinstance(out, Exception):
            print(f"Couldn't build the evidence bank: {out}", file=sys.stderr)
            return 1

    bank = gen_mod.load_bank(args.master)
    if bank.unreviewed_count():
        left = _review_bank_interactively(args.master) if interactive else bank.unreviewed_count()
        if left:
            print(f"{left} bullet(s) are still unreviewed and won't be used. Confirm them in the "
                  "web UI (python server.py -> Setup tab) or delete their `reviewed: false` line.",
                  file=sys.stderr)
        if gen_mod.load_bank(args.master).reviewed_only()[0].is_empty:
            print("No confirmed bullets yet — nothing to build a resume from.", file=sys.stderr)
            return 3

    return cmd_optimize(args)


def _try_write_pdf(docx_path: Path, console) -> Path | None:
    """Best-effort .docx -> .pdf via docx2pdf (needs MS Word on Windows or
    Pages on macOS). Returns the written path or None — PDF failure is never
    fatal, the .txt/.docx outputs are already the deliverables."""
    try:
        from docx2pdf import convert  # type: ignore

        target = docx_path.with_suffix(".pdf")
        convert(str(docx_path), str(target))
        return target if target.exists() else None
    except ImportError:
        console.print(
            "[dim]PDF skipped — `pip install docx2pdf` (plus MS Word installed) to also "
            "write a .pdf next to the .docx.[/dim]"
        )
    except Exception as e:  # noqa: BLE001 — conversion is best-effort only
        console.print(f"[dim]PDF conversion failed ({escape(str(e))}) — the .txt/.docx outputs are fine.[/dim]")
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a resume against a job description across three layers: "
                     "search visibility, recruiter screen, and manager evidence."
    )
    parser.add_argument("--db", default=applog.DEFAULT_DB, help="Application log database path")
    sub = parser.add_subparsers(dest="command", required=True)

    # init-profile
    p_init = sub.add_parser("init-profile", help="Create a profile.yaml template")
    p_init.add_argument("--path", default=profile_mod.DEFAULT_PROFILE_PATH)
    p_init.add_argument("--force", action="store_true", help="Overwrite an existing file")
    p_init.set_defaults(func=cmd_init_profile)

    # test-llm
    p_test = sub.add_parser("test-llm", help="Verify your LLM provider config with one request")
    p_test.add_argument("--provider", choices=PROVIDERS)
    p_test.add_argument("--host", help="Base URL (or a preset: glm, openai, groq, deepseek...)")
    p_test.add_argument("--model")
    p_test.add_argument("--api-key")
    p_test.set_defaults(func=cmd_test_llm)

    # score
    p_score = sub.add_parser("score", help="Score a resume against a JD")
    p_score.add_argument("--resume", required=True, help="Resume file (.pdf, .docx, .txt)")
    jd_group = p_score.add_mutually_exclusive_group(required=True)
    jd_group.add_argument("--jd", help="Path to a JD text file, or a posting URL (http/https)")
    jd_group.add_argument("--jd-text", help="JD text pasted directly")
    p_score.add_argument("--profile", default=profile_mod.DEFAULT_PROFILE_PATH)
    p_score.add_argument("--model", default=ollama_client.DEFAULT_MODEL,
                          help="Override the model from .env")
    p_score.add_argument("--host", default=ollama_client.DEFAULT_HOST,
                          help="Base URL, or a preset: glm, openai, groq, deepseek, openrouter")
    p_score.add_argument("--api-key", default=ollama_client.DEFAULT_API_KEY)
    p_score.add_argument("--provider", choices=PROVIDERS,
                          help="Override the provider resolved from .env")
    p_score.add_argument("--answers", metavar="FILE", help=MANUAL_HELP)
    p_score.add_argument("--offline", action="store_true",
                          help="Skip both LLM layers — fast, no Ollama needed")
    p_score.add_argument("--no-semantic", action="store_true", help="Skip the ATS semantic layer only")
    p_score.add_argument("--no-manager", action="store_true", help="Skip the manager evidence layer only")
    p_score.add_argument("--jd-extractor", choices=["rules", "llm"], default="rules",
                         help=JD_EXTRACTOR_HELP)
    p_score.add_argument("--brief", action="store_true", help="Hide the full recruiter checklist table")
    p_score.add_argument("--json", action="store_true", help="Machine-readable output")
    p_score.add_argument("--out", help="Also write the JSON report here")
    p_score.add_argument("--log", action="store_true",
                         help=f"Save this scoring to the application log (default on with {AUTO_LOG_ENV}=1)")
    p_score.add_argument("--no-log", action="store_true", help=f"Don't log, even with {AUTO_LOG_ENV}=1")
    p_score.add_argument("--company", help="Company name for the log (default: from the JD / its URL)")
    p_score.add_argument("--role", help="Role title for the log (default: the JD title)")
    p_score.add_argument("--days-after-posting", type=int,
                          help="Days between the job being posted and you applying")
    p_score.set_defaults(func=cmd_score)

    # log
    p_log = sub.add_parser("log", help="Application log: list / outcome / export / stats")
    log_sub = p_log.add_subparsers(dest="log_command", required=True)

    p_list = log_sub.add_parser("list", help="List logged applications")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.set_defaults(func=cmd_log_list)

    p_out = log_sub.add_parser("outcome", help="Record what happened with an application")
    p_out.add_argument("target", nargs="?",
                       help="Application id, or a company name (fuzzy-matched)")
    p_out.add_argument("--last", action="store_true", help="The most recently logged application")
    p_out.add_argument("--status", required=True, choices=applog.OUTCOMES)
    p_out.add_argument("--notes")
    p_out.set_defaults(func=cmd_log_outcome)

    p_exp = log_sub.add_parser("export", help="Export the log to CSV (Power BI ready)")
    p_exp.add_argument("--out", default="applications.csv")
    p_exp.set_defaults(func=cmd_log_export)

    p_stats = log_sub.add_parser("stats", help="Conversion by score band (needs enough outcomes)")
    p_stats.add_argument("--min-resolved", type=int, default=20)
    p_stats.add_argument("--reap-days", type=int, default=45,
                         help="First mark pending applications older than this as ghosted")
    p_stats.add_argument("--no-reap", action="store_true",
                         help="Leave stale pending applications alone (they're reported, not counted)")
    p_stats.add_argument("--json", action="store_true", help="Machine-readable output")
    p_stats.set_defaults(func=cmd_log_stats)

    p_reap = log_sub.add_parser("reap-ghosts",
                                help="Mark long-pending applications as ghosted")
    p_reap.add_argument("--days", type=int, default=45,
                        help="Pending applications older than this become ghosted")
    p_reap.set_defaults(func=cmd_log_reap)

    # batch
    p_batch = sub.add_parser("batch", help="Score one resume against a folder of JDs, ranked")
    p_batch.add_argument("--resume", required=True, help="Resume file (.pdf, .docx, .txt)")
    p_batch.add_argument("--jds-dir", required=True,
                         help="Folder containing JD text files (.txt / .md)")
    p_batch.add_argument("--profile", default=profile_mod.DEFAULT_PROFILE_PATH)
    p_batch.add_argument("--top", type=int, default=0,
                         help="Show only the top N JDs (0 = all)")
    p_batch.add_argument("--offline", action="store_true", help="Skip both LLM layers")
    p_batch.add_argument("--json", action="store_true", help="Machine-readable output")
    p_batch.add_argument("--model", default=ollama_client.DEFAULT_MODEL)
    p_batch.add_argument("--host", default=ollama_client.DEFAULT_HOST)
    p_batch.add_argument("--api-key", default=ollama_client.DEFAULT_API_KEY)
    p_batch.add_argument("--provider", choices=PROVIDERS)
    p_batch.add_argument("--jd-extractor", choices=["rules", "llm"], default="rules",
                         help=JD_EXTRACTOR_HELP)
    p_batch.set_defaults(func=cmd_batch)

    # eval
    from ats_checker import evaluation as ev_mod
    p_eval = sub.add_parser("eval", help="Measure JD extraction accuracy on the labelled corpus")
    p_eval.add_argument("--corpus", default=str(ev_mod.DEFAULT_CORPUS))
    p_eval.add_argument("--extractor", default="rules", choices=sorted(ev_mod.EXTRACTORS))
    p_eval.add_argument("--brief", action="store_true", help="Hide the per-JD miss list")
    p_eval.add_argument("--json", action="store_true", help="Print metrics as JSON")
    p_eval.add_argument("--write-baseline", metavar="FILE",
                        help="Save these metrics as the regression baseline")
    p_eval.add_argument("--model", default=ollama_client.DEFAULT_MODEL)
    p_eval.add_argument("--host", default=ollama_client.DEFAULT_HOST)
    p_eval.add_argument("--api-key", default=ollama_client.DEFAULT_API_KEY)
    p_eval.add_argument("--provider", choices=PROVIDERS)
    p_eval.add_argument("--answers", metavar="FILE", help=MANUAL_HELP)
    p_eval.add_argument("--no-cache", action="store_true",
                        help="Ask the LLM again even when a cached answer exists")
    p_eval.set_defaults(func=cmd_eval)

    # learned terms
    p_learn = sub.add_parser("learned", help="Review skills the LLM found that the taxonomy lacks")
    learn_sub = p_learn.add_subparsers(dest="learned_cmd", required=True)
    learn_sub.add_parser("list", help="Show candidate terms, most-seen first").set_defaults(
        func=cmd_learned_list)
    for name, fn, text in (("promote", cmd_learned_promote, "Add a term to the rule taxonomy"),
                           ("reject", cmd_learned_reject, "Drop a candidate term")):
        p = learn_sub.add_parser(name, help=text)
        p.add_argument("term")
        p.set_defaults(func=fn)

    # init-master
    p_master = sub.add_parser("init-master", help="Create the evidence bank (master resume)")
    p_master.add_argument("--from", dest="from_resume", metavar="FILE",
                          help="Bootstrap from an existing resume (.pdf/.docx/.txt) via the LLM. "
                               "Review every bullet afterwards — you are the truth gate.")
    p_master.add_argument("--path", default=gen_mod.DEFAULT_MASTER_PATH)
    p_master.add_argument("--force", action="store_true", help="Overwrite an existing bank")
    p_master.add_argument("--model", default=ollama_client.DEFAULT_MODEL)
    p_master.add_argument("--host", default=ollama_client.DEFAULT_HOST)
    p_master.add_argument("--api-key", default=ollama_client.DEFAULT_API_KEY)
    p_master.add_argument("--provider", choices=PROVIDERS)
    p_master.add_argument("--answers", metavar="FILE", help=MANUAL_HELP)
    p_master.set_defaults(func=cmd_init_master)

    # tailor
    p_tailor = sub.add_parser("tailor", help="Generate a JD-tailored resume from the evidence bank")
    p_tailor.add_argument("--master", default=gen_mod.DEFAULT_MASTER_PATH,
                          help="Path to the evidence bank (master_resume.yaml)")
    jd_group = p_tailor.add_mutually_exclusive_group(required=True)
    jd_group.add_argument("--jd", help="Path to a JD text file, or a posting URL (http/https)")
    jd_group.add_argument("--jd-text", help="JD text pasted directly")
    p_tailor.add_argument("--profile", default=profile_mod.DEFAULT_PROFILE_PATH)
    p_tailor.add_argument("--out", default="tailored_resume.txt", help="Output resume path (.txt)")
    p_tailor.add_argument("--no-docx", action="store_true",
                          help="Skip also writing a .docx next to the .txt")
    p_tailor.add_argument("--no-pdf", action="store_true",
                          help="Skip the best-effort .docx -> .pdf conversion "
                               "(needs `pip install docx2pdf` + MS Word)")
    p_tailor.add_argument("--log", action="store_true",
                          help=f"Log the generated application (default on with {AUTO_LOG_ENV}=1)")
    p_tailor.add_argument("--no-log", action="store_true", help=f"Don't log, even with {AUTO_LOG_ENV}=1")
    p_tailor.add_argument("--company", help="Company name for the log (default: from the JD / its URL)")
    p_tailor.add_argument("--role", help="Role title for the log (default: the JD title)")
    p_tailor.add_argument("--days-after-posting", type=int,
                          help="Days between the job being posted and you applying")
    p_tailor.add_argument("--offline", action="store_true",
                          help="Deterministic selection/ordering only — no LLM rewording or scoring")
    p_tailor.add_argument("--force", action="store_true",
                          help="Generate even when the no-apply gate finds candidacy blockers")
    p_tailor.add_argument("--max-current", type=int, default=5,
                          help="Bullet cap for the current/most recent role")
    p_tailor.add_argument("--max-other", type=int, default=3,
                          help="Bullet cap for each earlier role")
    p_tailor.add_argument("--max-lines", type=int, default=26,
                          help="Estimated bullet-line budget for the whole resume")
    p_tailor.add_argument("--brief", action="store_true", help="Hide the full recruiter checklist table")
    p_tailor.add_argument("--model", default=ollama_client.DEFAULT_MODEL)
    p_tailor.add_argument("--host", default=ollama_client.DEFAULT_HOST)
    p_tailor.add_argument("--api-key", default=ollama_client.DEFAULT_API_KEY)
    p_tailor.add_argument("--provider", choices=PROVIDERS)
    p_tailor.add_argument("--answers", metavar="FILE", help=MANUAL_HELP)
    p_tailor.set_defaults(func=cmd_tailor)

    # optimize + run share their options
    def add_optimize_args(p, with_master_jd: bool = True):
        p.add_argument("--master", default=gen_mod.DEFAULT_MASTER_PATH,
                       help="Path to the evidence bank (master_resume.yaml)")
        p.add_argument("--profile", default=profile_mod.DEFAULT_PROFILE_PATH)
        p.add_argument("--out", default="optimized_resume.txt", help="Output resume path (.txt)")
        p.add_argument("--no-docx", action="store_true", help="Skip also writing a .docx")
        p.add_argument("--max-rounds", type=int, default=5, help="Most improvement rounds (default 5)")
        p.add_argument("--max-current", type=int, default=5)
        p.add_argument("--max-other", type=int, default=3)
        p.add_argument("--max-lines", type=int, default=26)
        p.add_argument("--offline", action="store_true",
                       help="No LLM: selection moves only (no rewording)")
        p.add_argument("--force", action="store_true",
                       help="Optimize even when the no-apply gate finds candidacy blockers")
        p.add_argument("--auto-accept", action="store_true",
                       help="Put every verified rewording in WITHOUT asking you (riskier)")
        p.add_argument("--apply-proposals", metavar="FILE",
                       help="Use the rewordings marked \"accepted\": true in this proposals file")
        p.add_argument("--json", action="store_true", help="Machine-readable output")
        p.add_argument("--model", default=ollama_client.DEFAULT_MODEL)
        p.add_argument("--host", default=ollama_client.DEFAULT_HOST)
        p.add_argument("--api-key", default=ollama_client.DEFAULT_API_KEY)
        p.add_argument("--provider", choices=PROVIDERS)
        p.add_argument("--answers", metavar="FILE", help=MANUAL_HELP)

    p_opt = sub.add_parser("optimize", help="Tailor, score, improve and rescore until no truthful "
                                            "gain is left")
    jd_group = p_opt.add_mutually_exclusive_group(required=True)
    jd_group.add_argument("--jd", help="Path to a JD text file, or a posting URL (http/https)")
    jd_group.add_argument("--jd-text", help="JD text pasted directly")
    add_optimize_args(p_opt)
    p_opt.set_defaults(func=cmd_optimize)

    p_run = sub.add_parser("run", help="End to end: profile + evidence bank if missing, then "
                                       "optimize, then write the files")
    p_run.add_argument("--resume", required=True, help="Your current resume (.pdf, .docx, .txt)")
    jd_group = p_run.add_mutually_exclusive_group(required=True)
    jd_group.add_argument("--jd", help="Path to a JD text file, or a posting URL (http/https)")
    jd_group.add_argument("--jd-text", help="JD text pasted directly")
    add_optimize_args(p_run)
    p_run.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    # Windows: a redirected stdout encodes as cp1252 and rich's panel/table
    # glyphs (▸, —, ✓) can crash the whole run with UnicodeEncodeError.
    # Replace unmappable characters instead of dying after all the work.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = build_parser()
    args = parser.parse_args()
    if (args.command in {"score", "tailor", "eval", "init-master"}
            and _provider_of(args) == "manual" and not getattr(args, "offline", False)):
        return _run_manual_captured(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
