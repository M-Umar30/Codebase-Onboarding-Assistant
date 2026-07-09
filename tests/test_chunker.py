"""app/indexing/chunker.py unit tests — fixed-size line-window edges.

No filesystem or DB involved: chunk_text operates on in-memory strings.
"""

from __future__ import annotations

from app.indexing.chunker import OVERLAP_SIZE, WINDOW_SIZE, chunk_text
from app.schemas import ChunkKind, Language


def _lines(n: int) -> str:
    return "\n".join(f"line {i}" for i in range(1, n + 1))


def test_empty_content_yields_no_chunks() -> None:
    assert chunk_text("repo", "empty.py", "") == []


def test_content_shorter_than_window_yields_single_chunk() -> None:
    chunks = chunk_text("repo", "small.py", _lines(5))
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 5


def test_content_exactly_window_size_yields_single_chunk() -> None:
    chunks = chunk_text("repo", "exact.py", _lines(WINDOW_SIZE))
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == WINDOW_SIZE


def test_content_one_line_over_window_yields_overlapping_second_chunk() -> None:
    total = WINDOW_SIZE + 1
    chunks = chunk_text("repo", "over.py", _lines(total))
    assert len(chunks) == 2

    assert chunks[0].start_line == 1
    assert chunks[0].end_line == WINDOW_SIZE

    step = WINDOW_SIZE - OVERLAP_SIZE
    assert chunks[1].start_line == 1 + step
    assert chunks[1].end_line == total

    # the overlap region is shared verbatim between consecutive chunks
    overlap_start = chunks[1].start_line
    overlap_end = chunks[0].end_line
    assert overlap_end - overlap_start + 1 == OVERLAP_SIZE


def test_last_chunk_never_exceeds_total_lines() -> None:
    total = WINDOW_SIZE * 3 + 7
    chunks = chunk_text("repo", "long.py", _lines(total))
    assert chunks[-1].end_line == total
    for chunk in chunks:
        assert chunk.end_line <= total
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_every_chunk_is_naive_language_and_block_kind() -> None:
    chunks = chunk_text("repo", "file.ts", _lines(10))
    assert len(chunks) == 1
    assert chunks[0].language is Language.OTHER
    assert chunks[0].kind is ChunkKind.BLOCK
    assert chunks[0].symbol is None


def test_chunk_id_is_deterministic_for_same_inputs() -> None:
    a = chunk_text("repo", "file.py", _lines(5))
    b = chunk_text("repo", "file.py", _lines(5))
    assert a[0].id == b[0].id
    assert a[0].content_hash == b[0].content_hash


def test_chunk_id_differs_by_repo_id() -> None:
    a = chunk_text("repo-a", "file.py", _lines(5))
    b = chunk_text("repo-b", "file.py", _lines(5))
    assert a[0].id != b[0].id
