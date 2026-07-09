"""cli/main.py — `onboard` CLI: index a repo, ask questions against it.

Phase 1: plain function pipeline (retrieve -> draft -> synthesize). No
LangGraph, no critic loop, no planner — those arrive in later phases.
"""

from __future__ import annotations

from pathlib import Path

import typer

from app.indexing.indexer import index_repo
from app.pipeline import ask as run_ask

app = typer.Typer(help="Codebase onboarding assistant.")


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
) -> None:
    """Ask a question against an already-indexed repo."""
    final_answer = run_ask(repo_id, question)

    typer.echo(final_answer.answer_markdown)

    if final_answer.citations:
        typer.echo("\nCitations:")
        for citation in final_answer.citations:
            typer.echo(f"  [{citation.id}] {citation.file_path}:{citation.start_line}-{citation.end_line}")

    if final_answer.unverified_notes:
        typer.echo("\nUnverified:")
        for note in final_answer.unverified_notes:
            typer.echo(f"  - {note}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
