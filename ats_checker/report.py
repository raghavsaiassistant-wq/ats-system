"""Terminal rendering (rich) + JSON export for the three-layer report."""
from __future__ import annotations

import json

from rich.markup import escape

from .scorer import FullReport, band

BAND_COLOR = {
    "Strong": "green",
    "Workable": "yellow",
    "Weak": "red",
    "Very weak": "red",
    "Not scored": "dim",
}

STATUS_MARK = {
    "pass": ("[green]PASS[/green]", ""),
    "warn": ("[yellow]WARN[/yellow]", ""),
    "fail": ("[red]FAIL[/red]", ""),
    "skipped": ("[dim]SKIP[/dim]", "dim"),
}


def to_json(report: FullReport, indent: int = 2) -> str:
    return json.dumps(report.to_dict(), indent=indent, default=str)


def print_report(report: FullReport, show_checks: bool = True) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # ---- headline: three scores side by side
    head = Table(show_header=True, header_style="bold", title="Three-Layer Score")
    head.add_column("Layer")
    head.add_column("Score", justify="right")
    head.add_column("Band")
    head.add_column("What it measures")

    vis = report.visibility
    for label, score, meaning in (
        ("1. Search Visibility", vis.score if vis else None,
         "% of the recruiter searches this JD implies that would find you"),
        ("2. HR Screen", report.recruiter_score, "% of the JD's stated screening criteria you meet"),
        ("3. Manager Evidence", report.manager_score, "% evidence-strength score on the review rubric"),
    ):
        b = band(score)
        score_txt = f"[{BAND_COLOR[b]}]{score}/100[/{BAND_COLOR[b]}]" if score is not None else "[dim]—[/dim]"
        head.add_row(label, score_txt, f"[{BAND_COLOR[b]}]{b}[/{BAND_COLOR[b]}]", meaning)
    console.print(head)

    console.print(
        "[dim]These are percentages of things measured — not probabilities of passing. "
        "They don't predict selection, which depends on the rest of the applicant pool. "
        "Real passing rates need outcome data: see `log stats`.[/dim]\n"
    )

    # ---- HR sub-split: what you can fix vs what you just have to know
    rec = report.recruiter_result
    if rec and (rec.resume_pct is not None or rec.candidacy_pct is not None):
        split = Table(title="HR Screen — split by what you can actually change",
                       show_header=True, header_style="bold")
        split.add_column("Type")
        split.add_column("Met", justify="right")
        split.add_column("Meaning")
        if rec.resume_pct is not None:
            b = band(rec.resume_pct)
            split.add_row("Resume-fixable",
                           f"[{BAND_COLOR[b]}]{rec.resume_pct}%[/{BAND_COLOR[b]}]",
                           "You change these by editing the document — this is homework")
        if rec.candidacy_pct is not None:
            b = band(rec.candidacy_pct)
            split.add_row("Candidacy fit",
                           f"[{BAND_COLOR[b]}]{rec.candidacy_pct}%[/{BAND_COLOR[b]}]",
                           "Facts about you — no rewrite changes these, just go in knowing them")
        console.print(split)

    # ---- Layer 1 detail: parse gate, then every simulated search
    if vis:
        gate = Table(title="Layer 1a — Parse gate (can an ATS read the file?)",
                     show_header=True, header_style="bold")
        gate.add_column("", width=6)
        gate.add_column("Check", width=26)
        gate.add_column("Detail")
        for chk in vis.parse_checks:
            gate.add_row(STATUS_MARK.get(chk.status, ("?", ""))[0], escape(chk.name), escape(chk.detail))
        console.print(gate)
        console.print("[green]Parse gate: PASSED[/green]" if vis.parse_safe else
                      "[red]Parse gate: FAILED — fix these before anything else; an ATS may "
                      "index too little of your resume for any search to find it.[/red]")

        if vis.searches:
            st = Table(title=f"Layer 1b — Recruiter searches you'd appear in: "
                             f"{vis.matched_count} of {len(vis.searches)}",
                       show_header=True, header_style="bold")
            st.add_column("", width=6)
            st.add_column("Search", width=24)
            st.add_column("Query")
            st.add_column("Missing")
            for srch in vis.searches:
                st.add_row("[green]HIT[/green]" if srch.matched else "[red]MISS[/red]",
                           escape(srch.name), escape(srch.query), escape(", ".join(srch.missing)) or "—")
            console.print(st)
        for note in vis.notes:
            console.print(f"[dim]{escape(note)}[/dim]")

    jx = report.jd_extraction
    if jx and jx.get("source") == "llm+rules":
        rej, conf = jx.get("rejected") or [], jx.get("conflicts") or []
        console.print(
            f"[bold]JD read by:[/bold] LLM + rules{' (cached)' if jx.get('from_cache') else ''} — "
            f"{len(jx.get('llm_skills') or {})} LLM skills kept (each quotes its JD line), "
            f"{len(rej)} LLM item(s) dropped as unverifiable, {len(conf)} disagreement(s) with the rules")
        for r in rej[:5]:
            console.print(f"[dim]  dropped {escape(str(r['field']))}={escape(str(r['value']))}: "
                          f"{escape(str(r['reason']))}[/dim]")
        for c in conf[:5]:
            console.print(f"[dim]  {escape(str(c['field']))}: LLM {escape(str(c['llm']))} "
                          f"vs rules {escape(str(c['rules']))} (LLM's quoted value used)[/dim]")

    sem_txt = (f"{report.semantic_result.semantic_score}/100" if report.semantic_result.available
               else "not run")
    console.print(f"[bold]Layer 1c — LLM fit read:[/bold] {sem_txt}  [dim](a model's reading of "
                  "meaning-level fit — not something an ATS computes, so it's shown beside the "
                  "visibility score, never blended into it)[/dim]")
    console.print(f"[dim]Legacy composite 'ATS score' (deprecated, kept for old scripts): "
                  f"{report.ats_score}/100[/dim]\n")

    if report.parse_result.warnings:
        console.print(Panel("\n".join(f"- {escape(x)}" for x in report.parse_result.warnings),
                             title="Parsing / formatting warnings", border_style="yellow"))

    kw = report.keyword_result
    console.print(Panel(escape(", ".join(kw.matched_terms[:25])) or "(none)",
                         title=f"JD keyword coverage — matched ({len(kw.matched)})", border_style="green"))
    console.print(Panel(escape(", ".join(kw.missing_terms[:25])) or "(none)",
                         title=f"JD keyword coverage — missing ({len(kw.missing)})", border_style="red"))

    sem = report.semantic_result
    if sem.available:
        if sem.reworded_matches:
            console.print(Panel("\n".join(f"- {escape(s)}" for s in sem.reworded_matches),
                                 title="Experience you have but word differently than the JD",
                                 border_style="cyan"))
        if sem.gaps:
            console.print(Panel("\n".join(f"- {escape(s)}" for s in sem.gaps),
                                 title="Semantic gaps", border_style="red"))

    # ---- Layer 2 detail
    if rec:
        if rec.blockers:
            console.print(Panel("\n".join(f"- {escape(b)}" for b in rec.blockers),
                                 title="[bold red]Hard blockers — filtered on these first[/bold red]",
                                 border_style="red"))

        # The prep list: what HR will actually raise on the call
        if rec.expectations:
            derived = [e for e in rec.expectations if e.source == "derived"]
            standard = [e for e in rec.expectations if e.source == "standard"]
            lines = []
            if derived:
                lines.append("[bold]Specific to your application:[/bold]\n")
                for e in derived:
                    lines.append(f"[yellow]▸ {escape(e.topic)}[/yellow]")
                    lines.append(f"   [dim]{escape(e.why)}[/dim]")
                    lines.append(f"   [green]Prepare:[/green] {escape(e.prepare)}\n")
            if standard:
                lines.append("[bold]Asked on virtually every screen:[/bold]\n")
                for e in standard:
                    lines.append(f"[cyan]▸ {escape(e.topic)}[/cyan]")
                    lines.append(f"   [dim]{escape(e.why)}[/dim]")
                    lines.append(f"   [green]Prepare:[/green] {escape(e.prepare)}\n")
            console.print(Panel("\n".join(lines).rstrip(),
                                 title="What HR will expect from you",
                                 border_style="bold yellow"))

        if show_checks:
            table = Table(title="HR screen checklist", show_header=True, header_style="bold")
            table.add_column("", width=6)
            table.add_column("Type", width=10)
            table.add_column("Check", width=24)
            table.add_column("Detail")
            for check in rec.checks:
                mark, style = STATUS_MARK.get(check.status, ("?", ""))
                cat = ("[blue]RESUME[/blue]" if check.category == "resume"
                       else "[magenta]YOU[/magenta]")
                if check.status == "skipped":
                    cat = f"[dim]{'RESUME' if check.category == 'resume' else 'YOU'}[/dim]"
                detail = escape(check.detail)
                if check.jd_citation:
                    detail += f"\n[dim]JD: \"{escape(check.jd_citation[:100])}\"[/dim]"
                name = escape(check.name)
                table.add_row(mark, cat, f"[{style}]{name}[/{style}]" if style else name, detail)
            console.print(table)
            console.print("[dim]RESUME = fixable by editing.  YOU = a fact about you; "
                           "go in knowing it.[/dim]")

    # ---- Layer 3 detail
    mgr = report.manager_result
    if mgr and mgr.available:
        dims = Table(title="Layer 3 — Manager evidence rubric", show_header=True, header_style="bold")
        dims.add_column("Dimension")
        dims.add_column("Score", justify="right")
        labels = {
            "quantification": "Quantification (numbers/scale)",
            "outcome_focus": "Outcomes vs responsibilities",
            "evidence_backing": "Skills backed by real work",
            "scope_match": "Scope matches role level",
            "domain_relevance": "Day-to-day relevance",
            "credibility": "Credibility / defensibility",
        }
        for key, label in labels.items():
            if key not in mgr.dimensions:
                continue
            val = mgr.dimensions[key]
            if val is None:   # the model didn't score it: not a 0
                dims.add_row(label, "[dim]not scored[/dim]")
                continue
            color = "green" if val >= 70 else "yellow" if val >= 45 else "red"
            dims.add_row(label, f"[{color}]{val}/100[/{color}]")
        console.print(dims)
        if mgr.not_scored:
            console.print(f"[dim]The model didn't score {len(mgr.not_scored)} of {len(labels)} "
                          f"dimensions; the Layer 3 score is the weighted average of the ones "
                          f"it did score, not a penalty for the missing ones.[/dim]")

        if mgr.weak_bullets:
            lines = []
            for wb in mgr.weak_bullets:
                lines.append(f"[red]✗[/red] {escape(wb['bullet'])}")
                lines.append(f"   [dim]{escape(wb['problem'])}[/dim]")
                lines.append(f"   [green]→[/green] {escape(wb['rewrite'])}\n")
            console.print(Panel("\n".join(lines).rstrip(),
                                 title="Weakest bullets, with rewrites", border_style="yellow"))

        if mgr.interview_risks:
            console.print(Panel("\n".join(f"- {escape(r)}" for r in mgr.interview_risks),
                                 title="Claims a manager would probe — be ready to defend these",
                                 border_style="magenta"))
        if mgr.verdict:
            console.print(Panel(escape(mgr.verdict), title="Manager verdict", border_style="bold"))
    elif mgr and mgr.error:
        console.print(Panel(escape(mgr.error), title="Layer 3 — unavailable", border_style="dim"))

    if report.writing_review:
        findings = report.writing_review['findings']
        lines = [f"Line {f['line']}: {f['excerpt']}\n{f['explanation']}" for f in findings]
        lines.append('Local feedback only. No external AI detector was run; authorship is not established.')
        console.print(Panel(escape('\n\n'.join(lines)), title='Writing Review — separate from scores'))

    if sem.available and sem.recommendation:
        console.print(Panel(escape(sem.recommendation), title="Overall recommendation", border_style="bold"))

    for note in report.notes:
        console.print(f"[dim]Note: {escape(note)}[/dim]")


def print_stats(stats: dict) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    overall = stats.get("overall")
    console.print(Panel(
        f"Logged: {stats['total_logged']}   Resolved outcomes: {stats['resolved']}"
        + (f"\nReached a human overall: {escape(overall['rate'])}" if overall else ""),
        title="Application log", border_style="bold",
    ))

    if not stats["calibrated"]:
        console.print(f"[yellow]{escape(stats['message'])}[/yellow]")
        return

    def rate_table(title: str, first_col: str, buckets: dict) -> Table:
        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column(first_col)
        table.add_column("n", justify="right")
        table.add_column("Reached human", justify="right")
        table.add_column("Rate (95% CI)", justify="right")
        for name, bucket in buckets.items():
            table.add_row(escape(name), str(bucket["n"]), str(bucket["reached_human"]),
                          f"{bucket['conversion_pct']:.0f}% "
                          f"({bucket['ci_low']:.0f}\u2013{bucket['ci_high']:.0f}%)")
        return table

    for layer, bands in stats["bands"].items():
        if not bands:   # layer never scored (e.g. manager on offline runs)
            continue
        console.print(rate_table(f"{layer.title()} score → reached a human", "Score band", bands))

    def corr_reading(info: dict) -> str:
        text = info["strength"]
        if info["distinguishable_from_zero"] and info["correlation"] < 0:
            text += " — higher scores convert WORSE"
        return escape(text)

    def corr_cell(info: dict) -> str:
        color = ("dim" if not info["distinguishable_from_zero"] else
                 "green" if info["correlation"] > 0 else "red")
        return (f"[{color}]{info['correlation']:+.2f}[/{color}] "
                f"({info['ci_low']:+.2f} to {info['ci_high']:+.2f})")

    # Which layer actually predicts your outcomes (point-biserial r, Fisher-z CI)
    pred = stats.get("predictiveness") or {}
    if pred:
        corr = Table(title="Which layer predicts your outcomes",
                     show_header=True, header_style="bold")
        corr.add_column("Layer")
        corr.add_column("r (95% CI)", justify="right")
        corr.add_column("n", justify="right")
        corr.add_column("Reading")
        for layer, info in pred.items():
            corr.add_row(layer.title(), corr_cell(info), str(info["n"]), corr_reading(info))
        console.print(corr)

    comps = stats.get("component_predictiveness") or []
    if comps:
        ctable = Table(title="Which component predicts your outcomes (ranked)",
                       show_header=True, header_style="bold")
        ctable.add_column("Component")
        ctable.add_column("r (95% CI)", justify="right")
        ctable.add_column("n", justify="right")
        ctable.add_column("Reading")
        for info in comps:
            ctable.add_row(escape(info["component"]), corr_cell(info), str(info["n"]),
                           corr_reading(info))
        console.print(ctable)

    if pred or comps:
        console.print(
            "[dim]Correlation of each score with actually reaching a human "
            "(+1 = high scores always convert, 0 = the score tells you nothing, "
            "negative = high scores convert WORSE — investigate). A 95% interval that "
            "crosses 0 means the data can't yet tell it apart from no relationship; "
            "strength isn't graded below n=30.[/dim]\n"
        )

    timing = stats.get("apply_timing") or {}
    if timing:
        console.print(rate_table("When you applied → reached a human",
                                 "Applied after posting", timing))

    console.print(f"[dim]{escape(stats['message'])}[/dim]")
