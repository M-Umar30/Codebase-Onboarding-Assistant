"""app/critic/mechanical.py — deterministic citation verification, zero LLM.

Checks each citation in a DraftAnswer against the actual filesystem and the
already-indexed chunk list: does the file exist, are the cited lines in
bounds, does the claimed symbol appear in the cited range, and does the
citation's location match evidence that was actually retrieved (not just
line numbers that happen to fall inside a real file)?

The hash check is containment-based, not exact-range equality: a citation
commonly cites a sub-range of a larger retrieved chunk (e.g. lines 10-25 of
a 60-line window) to support a specific claim, so the lookup finds the
*containing* indexed chunk and checks that chunk's own staleness, rather
than requiring the citation's own bounds to already be a chunk boundary.

hash_matches_index=False collapses two distinct causes — no containing chunk
at all (nothing was ever retrieved at this location) vs. a containing chunk
that's gone stale (file changed since indexing). MechanicalCheck is frozen
with no field to carry that distinction downstream; Phase 2 fixtures model
the dominant "never retrieved" case (see semantic.py's mapping).
"""

from __future__ import annotations

from pathlib import Path

from app.indexing.chunker import content_hash
from app.schemas import Citation, CodeChunk, DraftAnswer, MechanicalCheck


def read_cited_lines(
    repo_root: Path, file_path: str, start_line: int, end_line: int
) -> str | None:
    """The 1-indexed, inclusive line range's text, or None if the file is
    missing, unreadable, or the range is out of bounds. Shared with
    app.critic.semantic so both layers read identical content."""
    abs_path = repo_root / file_path
    try:
        text = abs_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    lines = text.splitlines()
    if start_line > len(lines) or end_line > len(lines):
        return None
    return "\n".join(lines[start_line - 1 : end_line])


def _find_containing_chunk(chunks: list[CodeChunk], citation: Citation) -> CodeChunk | None:
    """Smallest indexed chunk for this file whose range contains the
    citation's range, or None if no indexed chunk covers it at all."""
    candidates = [
        chunk
        for chunk in chunks
        if chunk.start_line <= citation.start_line and citation.end_line <= chunk.end_line
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.end_line - c.start_line)


def run_mechanical_checks(
    draft: DraftAnswer, repo_root: Path, index: list[CodeChunk]
) -> list[MechanicalCheck]:
    chunks_by_file: dict[str, list[CodeChunk]] = {}
    for chunk in index:
        chunks_by_file.setdefault(chunk.file_path, []).append(chunk)

    checks: list[MechanicalCheck] = []
    for citation in draft.citations:
        abs_path = repo_root / citation.file_path
        file_exists = False
        lines_in_bounds = False
        symbol_found: bool | None = None
        hash_matches_index = False

        try:
            file_exists = abs_path.is_file()
        except OSError:
            file_exists = False

        if file_exists:
            try:
                text = abs_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                file_exists = False
            else:
                lines = text.splitlines()
                total_lines = len(lines)
                lines_in_bounds = (
                    citation.start_line <= total_lines and citation.end_line <= total_lines
                )
                if lines_in_bounds:
                    cited_snippet = "\n".join(
                        lines[citation.start_line - 1 : citation.end_line]
                    )
                    if citation.symbol:
                        symbol_found = citation.symbol in cited_snippet

                    containing = _find_containing_chunk(
                        chunks_by_file.get(citation.file_path, []), citation
                    )
                    if containing is not None:
                        chunk_snippet = "\n".join(
                            lines[containing.start_line - 1 : containing.end_line]
                        )
                        hash_matches_index = (
                            content_hash(chunk_snippet) == containing.content_hash
                        )

        passed = (
            file_exists
            and lines_in_bounds
            and (symbol_found is not False)
            and hash_matches_index
        )

        checks.append(
            MechanicalCheck(
                citation_id=citation.id,
                file_exists=file_exists,
                lines_in_bounds=lines_in_bounds,
                symbol_found=symbol_found,
                hash_matches_index=hash_matches_index,
                passed=passed,
            )
        )
    return checks
