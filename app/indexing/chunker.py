"""app/indexing/chunker.py — naive fixed-size line-window chunking.

Phase 1 scope: no syntax awareness. Every chunk is Language.OTHER /
ChunkKind.BLOCK regardless of file type; tree-sitter chunking arrives in
Phase 4. Windows are WINDOW_SIZE lines with OVERLAP_SIZE lines shared
between consecutive windows, so a fact near a window boundary still shows
up whole in at least one chunk.
"""

from __future__ import annotations

import hashlib

from app.schemas import ChunkKind, CodeChunk, Language

WINDOW_SIZE = 60
OVERLAP_SIZE = 10

assert WINDOW_SIZE > OVERLAP_SIZE, "window must advance each step"

_STEP = WINDOW_SIZE - OVERLAP_SIZE


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def chunk_id(repo_id: str, file_path: str, start_line: int, hash_: str) -> str:
    raw = f"{repo_id}:{file_path}:{start_line}:{hash_}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def chunk_text(repo_id: str, file_path: str, text: str) -> list[CodeChunk]:
    """Split `text` (one file's content, repo-relative `file_path` with POSIX
    separators) into fixed-size overlapping line windows."""
    lines = text.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return []

    chunks: list[CodeChunk] = []
    start = 1
    while True:
        end = min(start + WINDOW_SIZE - 1, total_lines)
        window_content = "\n".join(lines[start - 1 : end])
        hash_ = content_hash(window_content)
        chunks.append(
            CodeChunk(
                id=chunk_id(repo_id, file_path, start, hash_),
                repo_id=repo_id,
                file_path=file_path,
                start_line=start,
                end_line=end,
                language=Language.OTHER,
                kind=ChunkKind.BLOCK,
                symbol=None,
                content=window_content,
                content_hash=hash_,
            )
        )
        if end >= total_lines:
            break
        start += _STEP

    return chunks
