"""app/retrieval/store.py — ask-time loaders for the critic's mechanical layer.

The mechanical critic (app/critic/mechanical.py) needs two things the retriever
never returns: the repo's on-disk root (to read cited files) and the FULL
indexed chunk list (to check citation containment against everything that was
indexed, not just the top-k that came back). Both are loaded once per ask, from
the `repos` and `chunks` tables written at index time.
"""

from __future__ import annotations

from pathlib import Path

from app.schemas import ChunkKind, CodeChunk, Language


class RepoNotIndexedError(RuntimeError):
    """No usable index exists for a repo_id: either it has no `repos` row
    (indexed before Phase 3, or never indexed) or its recorded root path is
    gone from disk. Both mean the mechanical layer cannot verify citations,
    so this is raised at the boundary rather than letting every check fail as
    a false FABRICATED verdict."""


def load_repo_root(repo_id: str, conn) -> Path:
    row = conn.execute(
        "SELECT root_path FROM repos WHERE repo_id = %s", (repo_id,)
    ).fetchone()
    if row is None:
        raise RepoNotIndexedError(
            f"repo_id '{repo_id}' has no recorded root path. Re-run "
            f"`onboard index <path> --repo-id {repo_id}` "
            "(repos indexed before Phase 3 must be re-indexed once)."
        )
    root = Path(row[0])
    if not root.is_dir():
        raise RepoNotIndexedError(
            f"repo_id '{repo_id}' was indexed from '{root}', which no longer "
            "exists on disk. Re-index it before asking."
        )
    return root


def load_chunks(repo_id: str, conn) -> list[CodeChunk]:
    """Every indexed chunk for the repo, minus the embedding (the mechanical
    layer only needs line spans + content hashes)."""
    rows = conn.execute(
        """
        SELECT id, repo_id, file_path, start_line, end_line,
               language, kind, symbol, content, content_hash
        FROM chunks
        WHERE repo_id = %s
        """,
        (repo_id,),
    ).fetchall()
    return [
        CodeChunk(
            id=row[0],
            repo_id=row[1],
            file_path=row[2],
            start_line=row[3],
            end_line=row[4],
            language=Language(row[5]),
            kind=ChunkKind(row[6]),
            symbol=row[7],
            content=row[8],
            content_hash=row[9],
        )
        for row in rows
    ]
