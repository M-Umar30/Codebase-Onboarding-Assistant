"""app/indexing/indexer.py — Phase 1 indexing pipeline: walk -> chunk -> embed -> store.

No incremental re-index yet (Phase 6): every `index_repo` call re-embeds the
whole repo and replaces that repo_id's rows. `files_skipped_unchanged` is
therefore always 0 here — the field exists in IndexResponse for Phase 6 to
start reporting real numbers into.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config import Settings, get_settings
from app.db import get_connection
from app.embeddings import Embedder, get_embedder
from app.indexing.chunker import chunk_text
from app.indexing.walker import iter_source_files
from app.schemas import CodeChunk, IndexResponse

EMBED_BATCH_SIZE = 64


def derive_repo_id(source: str) -> str:
    """Default repo_id when the caller doesn't supply one: a slugified
    basename of the source path."""
    name = Path(source).resolve().name
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug or "repo"


def _insert_chunks(conn, chunks: list[CodeChunk], vectors: list[list[float]]) -> None:
    with conn.cursor() as cur:
        for chunk, vector in zip(chunks, vectors):
            cur.execute(
                """
                INSERT INTO chunks
                    (id, repo_id, file_path, start_line, end_line,
                     language, kind, symbol, content, content_hash, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    chunk.id,
                    chunk.repo_id,
                    chunk.file_path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.language.value,
                    chunk.kind.value,
                    chunk.symbol,
                    chunk.content,
                    chunk.content_hash,
                    vector,
                ),
            )


def index_repo(
    source: Path,
    repo_id: str | None = None,
    settings: Settings | None = None,
    conn=None,
    embedder: Embedder | None = None,
) -> IndexResponse:
    settings = settings or get_settings()
    repo_id = repo_id or derive_repo_id(str(source))
    embedder = embedder or get_embedder(settings)

    owns_conn = conn is None
    conn = conn or get_connection(settings)
    try:
        repo_root = Path(source).resolve()

        all_chunks: list[CodeChunk] = []
        files_indexed = 0
        for file_path in iter_source_files(repo_root):
            rel_path = file_path.relative_to(repo_root).as_posix()
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            file_chunks = chunk_text(repo_id, rel_path, text)
            if not file_chunks:
                continue
            all_chunks.extend(file_chunks)
            files_indexed += 1

        conn.execute("DELETE FROM chunks WHERE repo_id = %s", (repo_id,))

        chunks_written = 0
        for batch_start in range(0, len(all_chunks), EMBED_BATCH_SIZE):
            batch = all_chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
            vectors = embedder.embed_documents([c.content for c in batch])
            _insert_chunks(conn, batch, vectors)
            chunks_written += len(batch)

        return IndexResponse(
            repo_id=repo_id,
            files_indexed=files_indexed,
            files_skipped_unchanged=0,
            chunks_written=chunks_written,
            fallback_language_files=files_indexed,
        )
    finally:
        if owns_conn:
            conn.close()
