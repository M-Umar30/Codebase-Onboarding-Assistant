"""cli/main.py — `onboard` CLI: index a repo, ask questions against it.

Phase 3: `ask` runs the LangGraph pipeline (retrieve -> draft -> critic loop
-> synthesize). `--show-trace` renders the graph path taken — the plan, each
iteration's critic verdicts, the route chosen, and the loop count. That trace
is the interview demo, so it's rendered for readability.
"""

from __future__ import annotations

from pathlib import Path

import typer

from app.graph import MAX_CRITIC_ITERATIONS
from app.indexing.indexer import index_repo
from app.pipeline import ask_with_trace
from app.retrieval.store import RepoNotIndexedError
from app.schemas import CitationStatus, Route, Trace

app = typer.Typer(help="Codebase onboarding assistant.")

# Per-status colors for citation verdicts in the trace.
_STATUS_COLOR = {
    CitationStatus.VERIFIED: typer.colors.GREEN,
    CitationStatus.FABRICATED: typer.colors.RED,
    CitationStatus.WRONG_LOCATION: typer.colors.YELLOW,
    CitationStatus.UNSUPPORTED_CLAIM: typer.colors.YELLOW,
}

_ROUTE_COLOR = {
    Route.PROCEED: typer.colors.GREEN,
    Route.RE_RETRIEVE: typer.colors.CYAN,
    Route.REGENERATE: typer.colors.MAGENTA,
}


@app.command()
def index(
    path: Path = typer.Argument(..., help="Local path to the repo to index."),
    repo_id: str = typer.Option(
        None, help="Repo id to store chunks under; derived from the path if omitted."
    ),
) -> None:
    """Walk PATH, chunk it, embed, and store chunks in Postgres."""
    if not path.is_dir():
        typer.secho(f"Not a directory: {path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    result = index_repo(path, repo_id=repo_id)

    typer.echo(f"repo_id: {result.repo_id}")
    typer.echo(f"files indexed: {result.files_indexed}")
    typer.echo(f"chunks written: {result.chunks_written}")
    typer.echo(f"fallback-language files: {result.fallback_language_files}")


@app.command()
def ask(
    repo_id: str = typer.Argument(..., help="Repo id to query, as printed by `onboard index`."),
    question: str = typer.Argument(..., help="Question about the codebase."),
    show_trace: bool = typer.Option(
        False,
        "--show-trace",
        help="Print the graph path taken: plan, per-iteration critic verdicts, routes, loop count.",
    ),
) -> None:
    """Ask a question against an already-indexed repo."""
    try:
        response = ask_with_trace(repo_id, question)
    except RepoNotIndexedError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    final_answer = response.answer
    typer.echo(final_answer.answer_markdown)

    if final_answer.citations:
        typer.echo("\nCitations:")
        for citation in final_answer.citations:
            typer.echo(
                f"  [{citation.id}] {citation.file_path}:{citation.start_line}-{citation.end_line}"
            )

    if final_answer.unverified_notes:
        typer.echo("\nUnverified:")
        for note in final_answer.unverified_notes:
            typer.echo(f"  - {note}")

    if final_answer.confidence_caveat:
        typer.secho(f"\nCaveat: {final_answer.confidence_caveat}", fg=typer.colors.YELLOW)

    if show_trace:
        render_trace(response.trace)


def render_trace(trace: Trace) -> None:
    """Human-readable dump of the graph's execution — the demo surface.

    ASCII-only: the default Windows console (cp1252) can't encode box-drawing
    or arrow glyphs, and a demo that raises UnicodeEncodeError is worse than a
    plain one."""
    rule = "-" * 60
    typer.echo(f"\n{rule}")
    typer.secho(" trace", bold=True)
    typer.echo(rule)

    n_sub = len(trace.plan.sub_queries)
    noun = "sub-query" if n_sub == 1 else "sub-queries"
    typer.echo(f"Plan   {n_sub} {noun} (planner arrives in Phase 4)")
    for sub in trace.plan.sub_queries:
        typer.echo(f'       [{sub.id}] "{sub.query}"')

    models = " | ".join(f"{node}={model}" for node, model in trace.models_used.items())
    typer.echo(f"Models {models}")

    for it in trace.iterations:
        typer.echo("")
        typer.secho(
            f"Iteration {it.iteration} - {it.chunks_retrieved} chunks retrieved", bold=True
        )
        for verdict in it.critic.verdicts:
            color = _STATUS_COLOR.get(verdict.status, typer.colors.WHITE)
            label = verdict.status.value.ljust(17)
            typer.echo("  ", nl=False)
            typer.secho(f"[{verdict.citation_id}] {label}", fg=color, nl=False)
            typer.echo(f" - {verdict.reasoning}")

        route = it.critic.route
        typer.echo("  route -> ", nl=False)
        typer.secho(route.value, fg=_ROUTE_COLOR.get(route, typer.colors.WHITE))
        if it.critic.refined_queries:
            typer.echo(f"    refined: {', '.join(repr(q) for q in it.critic.refined_queries)}")
        if it.critic.regeneration_guidance:
            typer.echo(f"    guidance: {it.critic.regeneration_guidance}")
        typer.echo(f"    why: {it.critic.reasoning}")

    typer.echo("")
    exhausted = "yes" if trace.budget_exhausted else "no"
    footer = f"Loops used: {len(trace.iterations)}/{MAX_CRITIC_ITERATIONS} | budget exhausted: "
    typer.echo(footer, nl=False)
    typer.secho(exhausted, fg=typer.colors.RED if trace.budget_exhausted else typer.colors.GREEN)
    typer.echo(rule)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
