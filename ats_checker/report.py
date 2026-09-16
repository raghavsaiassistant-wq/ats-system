"""Terminal rendering (rich) + JSON export for the three-layer report."""
from __future__ import annotations

import json

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

    for label, score, meaning in (
        ("1. ATS", report.ats_score, "% of machine-filter criteria met (parse + keyword + semantic)"),
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

    # ---- Layer 1 detail
    ats = Table(title="Layer 1 — ATS breakdown", show_header=True, header_style="bold")
    ats.add_column("Component")
    ats.add_column("Score", justify="right")
    ats.add_column("Weight", justify="right")
    c, w = report.ats_components, report.ats_weights
    ats.add_row("Formatting / parseability", f"{c['formatting']}/100", f"{w.get('formatting', 0):.0%}")
    ats.add_row("Keyword match", f"{c['keyword_match']}/100", f"{w.get('keyword', 0):.0%}")
    ats.add_row(
        "Semantic fit" + ("" if report.semantic_result.available else " (unavailable)"),
        f"{c['semantic_fit']}/100" if report.semantic_result.available else "—",
        f"{w.get('semantic', 0):.0%}",
    )
    console.print(ats)

    if report.parse_result.warnings:
        console.print(Panel("\n".join(f"- {x}" for x in report.parse_result.warnings),
                             title="Parsing / formatting warnings", border_style="yellow"))

    kw = report.keyword_result
    console.print(Panel(", ".join(kw.matched_terms[:25]) or "(none)",
                         title=f"Matched keywords ({len(kw.matched)})", border_style="green"))
    console.print(Panel(", ".join(kw.missing_terms[:25]) or "(none)",
                         title=f"Missing keywords ({len(kw.missing)})", border_style="red"))

    sem = report.semantic_result
    if sem.available:
        if sem.reworded_matches:
            console.print(Panel("\n".join(f"- {s}" for s in sem.reworded_matches),
                                 title="Experience you have but word differently than the JD",
                                 border_style="cyan"))
        if sem.gaps:
            console.print(Panel("\n".join(f"- {s}" for s in sem.gaps),
                                 title="Semantic gaps", border_style="red"))

    # ---- Layer 2 detail
    if rec:
        if rec.blockers:
            console.print(Panel("\n".join(f"- {b}" for b in rec.blockers),
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
                    lines.append(f"[yellow]▸ {e.topic}[/yellow]")
                    lines.append(f"   [dim]{e.why}[/dim]")
                    lines.append(f"   [green]Prepare:[/green] {e.prepare}\n")
            if standard:
                lines.append("[bold]Asked on virtually every screen:[/bold]\n")
                for e in standard:
                    lines.append(f"[cyan]▸ {e.topic}[/cyan]")
                    lines.append(f"   [dim]{e.why}[/dim]")
                    lines.append(f"   [green]Prepare:[/green] {e.prepare}\n")
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
                detail = check.detail
                if check.jd_citation:
                    detail += f"\n[dim]JD: \"{check.jd_citation[:100]}\"[/dim]"
                table.add_row(mark, cat,
                               f"[{style}]{check.name}[/{style}]" if style else check.name,
                               detail)
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
            val = mgr.dimensions.get(key)
            if val is None:
                continue
            color = "green" if val >= 70 else "yellow" if val >= 45 else "red"
            dims.add_row(label, f"[{color}]{val}/100[/{color}]")
        console.print(dims)

        if mgr.weak_bullets:
            lines = []
            for wb in mgr.weak_bullets:
                lines.append(f"[red]✗[/red] {wb['bullet']}")
                lines.append(f"   [dim]{wb['problem']}[/dim]")
                lines.append(f"   [green]→[/green] {wb['rewrite']}\n")
            console.print(Panel("\n".join(lines).rstrip(),
                                 title="Weakest bullets, with rewrites", border_style="yellow"))

        if mgr.interview_risks:
            console.print(Panel("\n".join(f"- {r}" for r in mgr.interview_risks),
                                 title="Claims a manager would probe — be ready to defend these",
                                 border_style="magenta"))
        if mgr.verdict:
            console.print(Panel(mgr.verdict, title="Manager verdict", border_style="bold"))
    elif mgr and mgr.error:
        console.print(Panel(mgr.error, title="Layer 3 — unavailable", border_style="dim"))

    if sem.available and sem.recommendation:
        console.print(Panel(sem.recommendation, title="Overall recommendation", border_style="bold"))

    for note in report.notes:
        console.print(f"[dim]Note: {note}[/dim]")


def print_stats(stats: dict) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print(Panel(
        f"Logged: {stats['total_logged']}   Resolved outcomes: {stats['resolved']}",
        title="Application log", border_style="bold",
    ))

    if not stats["calibrated"]:
        console.print(f"[yellow]{stats['message']}[/yellow]")
        return

    for layer, bands in stats["bands"].items():
        table = Table(title=f"{layer.title()} score → reached a human",
                       show_header=True, header_style="bold")
        table.add_column("Score band")
        table.add_column("Applications", justify="right")
        table.add_column("Reached human", justify="right")
        table.add_column("Rate", justify="right")
        for band_name, bucket in bands.items():
            table.add_row(band_name, str(bucket["n"]), str(bucket["reached_human"]),
                           f"{bucket['conversion_pct']}%")
        console.print(table)

    # Which layer actually predicts your outcomes (point-biserial r)
    pred = stats.get("predictiveness") or {}
    if pred:
        corr = Table(title="Which layer predicts your outcomes",
                     show_header=True, header_style="bold")
        corr.add_column("Layer")
        corr.add_column("Correlation (r)", justify="right")
        corr.add_column("n", justify="right")
        corr.add_column("Reading")
        for layer, info in pred.items():
            color = "green" if abs(info["correlation"]) >= 0.5 else (
                "yellow" if abs(info["correlation"]) >= 0.3 else "dim")
            corr.add_row(layer.title(),
                         f"[{color}]{info['correlation']:+.2f}[/{color}]",
                         str(info["n"]), info["reading"])
        console.print(corr)
        console.print(
            "[dim]Correlation of each score with actually reaching a human "
            "(+1 = high scores always convert, 0 = the score tells you nothing, "
            "negative = high scores convert WORSE — investigate).[/dim]\n"
        )

    # Apply timing
    timing = stats.get("apply_timing") or {}
    if timing:
        t_table = Table(title="When you applied → reached a human",
                        show_header=True, header_style="bold")
        t_table.add_column("Applied after posting")
        t_table.add_column("Applications", justify="right")
        t_table.add_column("Reached human", justify="right")
        t_table.add_column("Rate", justify="right")
        for bucket_key, bucket in timing.items():
            t_table.add_row(bucket_key, str(bucket["n"]),
                            str(bucket["reached_human"]), f"{bucket['conversion_pct']}%")
        console.print(t_table)

    console.print(f"[dim]{stats['message']}[/dim]")
