"""app/retrieval/store.py tests. A hand-rolled fake connection stands in for
psycopg — no real Postgres, matching this project's no-network/no-DB fast-test
convention (test_retriever.py is the only DB-dependent suite).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.retrieval.store import RepoNotIndexedError, load_chunks, load_repo_root
from app.schemas import ChunkKind, Language


class _FakeResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Returns canned rows keyed by a substring of the SQL. Records the params
    each query was called with so tests can assert repo_id was threaded."""

    def __init__(self, repos_rows: list, chunks_rows: list) -> None:
        self._repos_rows = repos_rows
        self._chunks_rows = chunks_rows
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql: str, params=None) -> _FakeResult:
        self.calls.append((sql, params))
        if "FROM repos" in sql:
            return _FakeResult(self._repos_rows)
        if "FROM chunks" in sql:
            return _FakeResult(self._chunks_rows)
        raise AssertionError(f"unexpected SQL: {sql}")


class TestLoadRepoRoot:
    def test_returns_path_when_row_exists(self, tmp_path: Path) -> None:
        conn = _FakeConn(repos_rows=[(str(tmp_path),)], chunks_rows=[])

        root = load_repo_root("myrepo", conn)

        assert root == tmp_path
        assert conn.calls[0][1] == ("myrepo",)

    def test_missing_row_raises_with_reindex_hint(self) -> None:
        conn = _FakeConn(repos_rows=[], chunks_rows=[])

        with pytest.raises(RepoNotIndexedError, match="no recorded root path"):
            load_repo_root("ghost", conn)

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        gone = tmp_path / "was-here"  # never created
        conn = _FakeConn(repos_rows=[(str(gone),)], chunks_rows=[])

        with pytest.raises(RepoNotIndexedError, match="no longer"):
            load_repo_root("moved", conn)


class TestLoadChunks:
    def test_maps_rows_to_codechunks(self) -> None:
        row = (
            "chunk-id",
            "myrepo",
            "auth.py",
            1,
            30,
            "other",
            "block",
            None,
            "def authenticate(): ...",
            "deadbeef",
        )
        conn = _FakeConn(repos_rows=[], chunks_rows=[row])

        chunks = load_chunks("myrepo", conn)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.id == "chunk-id"
        assert chunk.repo_id == "myrepo"
        assert chunk.file_path == "auth.py"
        assert chunk.start_line == 1
        assert chunk.end_line == 30
        assert chunk.language is Language.OTHER
        assert chunk.kind is ChunkKind.BLOCK
        assert chunk.symbol is None
        assert chunk.content_hash == "deadbeef"
        assert conn.calls[0][1] == ("myrepo",)

    def test_empty_when_no_chunks(self) -> None:
        conn = _FakeConn(repos_rows=[], chunks_rows=[])

        assert load_chunks("empty", conn) == []
