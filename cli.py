#!/usr/bin/env python3
"""ATS Score Checker — CLI.

    python cli.py init-profile                                  # set up profile.yaml once
    python cli.py score --resume cv.pdf --jd jd.txt             # all three layers
    python cli.py score --resume cv.pdf --jd jd.txt --offline   # no LLM, layers 1+2 only
    python cli.py score --resume cv.pdf --jd https://...        # fetch the JD from a URL
    python cli.py batch --resume cv.pdf --jds-dir jds/          # score many JDs, ranked
    python cli.py score --resume cv.pdf --jd jd.txt --log --company "Acme" --role "BI Analyst"
    python cli.py log list
    python cli.py log outcome 3 --status interview
    python cli.py log export --out applications.csv
    python cli.py log stats
    python cli.py log reap-ghosts                               # pending -> ghosted after N days

    python cli.py init-master --from cv.pdf                     # build the evidence bank once
    python cli.py tailor --jd jd.txt                            # generate a JD-tailored resume
    python cli.py tailor --jd jd.txt --offline                  # deterministic selection only
    python cli.py tailor --jd jd.txt --log --company X --role Y # generate AND log the application
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ats_checker import applog
from ats_checker import llm_client as ollama_client
from ats_checker import profile as profile_mod
from ats_checker import report as report_mod
from ats_checker.scorer import run_full_check
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
    )

    if args.json:
        print(report_mod.to_json(result))
    else:
        report_mod.print_report(result, show_checks=not args.brief)

    if args.out:
        Path(args.out).write_text(report_mod.to_json(result), encoding="utf-8")
        if not args.json:
            print(f"\nSaved JSON report to {args.out}")

    if args.log:
        if not args.company or not args.role:
            print("\n--log needs --company and --role", file=sys.stderr)
            return 1
        app_id = applog.log_application(
            company=args.company,
            role=args.role,
            ats_score=result.ats_score,
            visibility_score=result.visibility.score if result.visibility else None,
            recruiter_score=result.recruiter_score,
            manager_score=result.manager_score,
            jd_text=jd_text,
            resume_version=Path(args.resume).name,
            days_after_posting=args.days_after_posting,
            db_path=args.db,
        )
        if not args.json:
            print(f"\nLogged as application #{app_id}. Update it later with:")
            print(f"  python cli.py log outcome {app_id} --status recruiter_call")

    return 0


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


def cmd_log_outcome(args) -> int:
    try:
        ok = applog.set_outcome(args.id, args.status, notes=args.notes, db_path=args.db)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    if not ok:
        print(f"No application with id {args.id}", file=sys.stderr)
        return 1
    print(f"Application #{args.id} → {args.status}")
    return 0


def cmd_log_export(args) -> int:
    path, n = applog.export_csv(args.out, db_path=args.db)
    print(f"Exported {n} application(s) to {path}")
    return 0


def cmd_log_stats(args) -> int:
    stats = applog.conversion_stats(db_path=args.db, min_resolved=args.min_resolved)
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
    table = Table(title=f"Ranked JDs for {Path(args.resume).name} ({len(rows)} scored)",
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
        table.add_row(str(i), name, _fmt(vis), _fmt(hr), _fmt(mgr))
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
    report = ev.evaluate(items, ev.EXTRACTORS[args.extractor], name=args.extractor)
    if args.json:
        print(json.dumps(report.metrics(), indent=2))
    else:
        ev.print_report(report, show_misses=not args.brief)
    if args.write_baseline:
        Path(args.write_baseline).write_text(
            json.dumps(report.metrics(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote baseline to {args.write_baseline}")
    return 0


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
            print("\nNOW REVIEW EVERY BULLET. The bank is the truth constraint — anything")
            print("wrong here propagates into every tailored resume. Add bullets over time;")
            print("the bank should stay richer than any single application needs.")
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
        console.print(f"[yellow]Bank issue:[/yellow] {note}")
    for note in result.notes:
        console.print(f"[dim]Note: {note}[/dim]")

    if result.blocked:
        console.print(Panel(
            "\n".join(f"- {b}" for b in result.candidacy_blockers),
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
            "\n".join(f"- {g}" for g in result.honest_gaps),
            title="[bold]Honest gaps — JD terms your evidence bank doesn't cover[/bold]",
            border_style="yellow",
        ))
        console.print("[dim]These are NOT added — a keyword you can't defend is an interview "
                      "trap. Gain the experience, then add the bullet to your bank.[/dim]")

    if result.rewordings:
        lines = []
        for rw in result.rewordings:
            lines.append(f"[cyan]{rw['source']}[/cyan]: {rw['reason']}")
            lines.append(f"  [dim]-[/dim] {rw['original']}")
            lines.append(f"  [green]+[/green] {rw['rewrite']}\n")
        console.print(Panel("\n".join(lines).rstrip(),
                             title="Verified rewordings applied", border_style="green"))
    if result.rejected_rewrites:
        lines = [f"- suggested: {r['suggested']}\n  rejected: {r['reason']}" for r in result.rejected_rewrites]
        console.print(Panel("\n".join(lines),
                             title="Suggested but rejected (failed fact verification)",
                             border_style="red"))

    console.print(Panel(
        f"Resume text : {out_txt.resolve()}"
        + (f"\nWord format : {docx_path.resolve()}" if docx_path else "")
        + (f"\nPDF format  : {pdf_path.resolve()}" if pdf_path else "")
        + "\nEvery line traces to an evidence-bank bullet — check them against your bank "
          "before sending. Fill any [X] placeholders with your real numbers.",
        title="Output", border_style="bold",
    ))

    # ---- optionally log this application, same as `score --log`
    if args.log:
        if not args.company or not args.role:
            print("\n--log needs --company and --role", file=sys.stderr)
            return 1
        rep = result.report
        app_id = applog.log_application(
            company=args.company,
            role=args.role,
            ats_score=rep.ats_score if rep else None,
            visibility_score=rep.visibility.score if rep and rep.visibility else None,
            recruiter_score=rep.recruiter_score if rep else None,
            manager_score=rep.manager_score if rep else None,
            jd_text=jd_text,
            resume_version=f"tailored:{out_txt.name}",
            days_after_posting=args.days_after_posting,
            db_path=args.db,
        )
        console.print(f"[dim]Logged as application #{app_id}. Update it later with:[/dim]")
        console.print(f"[dim]  python cli.py log outcome {app_id} --status recruiter_call[/dim]")

    return 0


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
        console.print(f"[dim]PDF conversion failed ({e}) — the .txt/.docx outputs are fine.[/dim]")
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
    p_test.add_argument("--provider", choices=["openai", "ollama"])
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
    p_score.add_argument("--provider", choices=["openai", "ollama"],
                          help="Override the provider resolved from .env")
    p_score.add_argument("--offline", action="store_true",
                          help="Skip both LLM layers — fast, no Ollama needed")
    p_score.add_argument("--no-semantic", action="store_true", help="Skip the ATS semantic layer only")
    p_score.add_argument("--no-manager", action="store_true", help="Skip the manager evidence layer only")
    p_score.add_argument("--brief", action="store_true", help="Hide the full recruiter checklist table")
    p_score.add_argument("--json", action="store_true", help="Machine-readable output")
    p_score.add_argument("--out", help="Also write the JSON report here")
    p_score.add_argument("--log", action="store_true", help="Save this scoring to the application log")
    p_score.add_argument("--company", help="Company name (required with --log)")
    p_score.add_argument("--role", help="Role title (required with --log)")
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
    p_out.add_argument("id", type=int)
    p_out.add_argument("--status", required=True, choices=applog.OUTCOMES)
    p_out.add_argument("--notes")
    p_out.set_defaults(func=cmd_log_outcome)

    p_exp = log_sub.add_parser("export", help="Export the log to CSV (Power BI ready)")
    p_exp.add_argument("--out", default="applications.csv")
    p_exp.set_defaults(func=cmd_log_export)

    p_stats = log_sub.add_parser("stats", help="Conversion by score band (needs enough outcomes)")
    p_stats.add_argument("--min-resolved", type=int, default=20)
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
    p_batch.add_argument("--provider", choices=["openai", "ollama"])
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
    p_eval.set_defaults(func=cmd_eval)

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
    p_master.add_argument("--provider", choices=["openai", "ollama"])
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
                          help="Log the generated application to the application log")
    p_tailor.add_argument("--company", help="Company name (required with --log)")
    p_tailor.add_argument("--role", help="Role title (required with --log)")
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
    p_tailor.add_argument("--provider", choices=["openai", "ollama"])
    p_tailor.set_defaults(func=cmd_tailor)

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
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
