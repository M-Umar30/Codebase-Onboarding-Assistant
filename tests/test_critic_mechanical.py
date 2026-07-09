"""app/critic/mechanical.py tests — pure, no LLM, no network. This is the
"fast subset" the Phase 2 task requires wired into pytest.
"""

from __future__ import annotations

from pathlib import Path

from app.critic.mechanical import read_cited_lines, run_mechanical_checks
from app.indexing.chunker import content_hash
from app.schemas import ChunkKind, Citation, CodeChunk, DraftAnswer, Language


def _write(tmp_path: Path, rel_path: str, lines: list[str]) -> Path:
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return file_path


def _citation(**overrides: object) -> Citation:
    defaults = dict(
        id=1,
        file_path="auth.py",
        start_line=1,
        end_line=5,
        symbol=None,
        claim="does something",
    )
    defaults.update(overrides)
    return Citation(**defaults)


def _chunk(**overrides: object) -> CodeChunk:
    defaults = dict(
        id="a" * 16,
        repo_id="repo-1",
        file_path="auth.py",
        start_line=1,
        end_line=10,
        language=Language.OTHER,
        kind=ChunkKind.BLOCK,
        symbol=None,
        content="placeholder",
        content_hash="placeholder-hash",
    )
    defaults.update(overrides)
    return CodeChunk(**defaults)


def _draft(citation: Citation) -> DraftAnswer:
    return DraftAnswer(answer_markdown="Something happens [1].", citations=[citation])


class TestReadCitedLines:
    def test_reads_inclusive_range(self, tmp_path: Path) -> None:
        _write(tmp_path, "auth.py", [f"line{i}" for i in range(1, 11)])
        assert read_cited_lines(tmp_path, "auth.py", 2, 4) == "line2\nline3\nline4"

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert read_cited_lines(tmp_path, "nope.py", 1, 5) is None

    def test_out_of_bounds_returns_none(self, tmp_path: Path) -> None:
        _write(tmp_path, "auth.py", ["only one line"])
        assert read_cited_lines(tmp_path, "auth.py", 1, 5) is None


class TestFileAndLineChecks:
    def test_file_does_not_exist(self, tmp_path: Path) -> None:
        citation = _citation(file_path="missing.py")
        checks = run_mechanical_checks(_draft(citation), tmp_path, index=[])
        assert checks[0].file_exists is False
        assert checks[0].passed is False

    def test_lines_out_of_bounds(self, tmp_path: Path) -> None:
        _write(tmp_path, "auth.py", ["a", "b", "c"])
        citation = _citation(start_line=1, end_line=10)
        checks = run_mechanical_checks(_draft(citation), tmp_path, index=[])
        assert checks[0].file_exists is True
        assert checks[0].lines_in_bounds is False
        assert checks[0].passed is False


class TestContainment:
    """Regression coverage for the bug caught in review: a citation to a
    sub-range of a larger indexed chunk must mechanically pass — it must not
    be branded fabricated just because its own bounds aren't a chunk
    boundary. This is exactly how a real drafter cites evidence (a claim
    about part of a 60-line retrieved window, not the whole window)."""

    def test_sub_range_within_larger_chunk_passes(self, tmp_path: Path) -> None:
        lines = [f"line{i}" for i in range(1, 21)]  # 20-line file
        _write(tmp_path, "auth.py", lines)
        chunk_content = "\n".join(lines)  # one chunk spans the whole file
        index = [
            _chunk(
                file_path="auth.py",
                start_line=1,
                end_line=20,
                content_hash=content_hash(chunk_content),
            )
        ]

        citation = _citation(start_line=10, end_line=12)  # sub-range within that chunk
        checks = run_mechanical_checks(_draft(citation), tmp_path, index=index)

        assert checks[0].hash_matches_index is True
        assert checks[0].passed is True

    def test_range_spanning_no_single_chunk_fails(self, tmp_path: Path) -> None:
        lines = [f"line{i}" for i in range(1, 21)]
        _write(tmp_path, "auth.py", lines)
        index = [
            _chunk(
                file_path="auth.py",
                start_line=1,
                end_line=10,
                content_hash=content_hash("\n".join(lines[0:10])),
            ),
            _chunk(
                file_path="auth.py",
                start_line=11,
                end_line=20,
                content_hash=content_hash("\n".join(lines[10:20])),
            ),
        ]
        # straddles both chunks, contained fully in neither
        citation = _citation(start_line=8, end_line=13)
        checks = run_mechanical_checks(_draft(citation), tmp_path, index=index)

        assert checks[0].hash_matches_index is False
        assert checks[0].passed is False

    def test_stale_containing_chunk_fails(self, tmp_path: Path) -> None:
        lines = [f"line{i}" for i in range(1, 21)]
        _write(tmp_path, "auth.py", lines)
        index = [
            _chunk(
                file_path="auth.py",
                start_line=1,
                end_line=20,
                content_hash="stale-hash-from-before-the-file-changed",
            )
        ]

        citation = _citation(start_line=10, end_line=12)
        checks = run_mechanical_checks(_draft(citation), tmp_path, index=index)

        assert checks[0].hash_matches_index is False
        assert checks[0].passed is False

    def test_tightest_containing_chunk_is_used(self, tmp_path: Path) -> None:
        lines = [f"line{i}" for i in range(1, 21)]
        _write(tmp_path, "auth.py", lines)
        narrow_chunk_content = "\n".join(lines[9:12])  # lines 10-12
        index = [
            _chunk(
                file_path="auth.py",
                start_line=1,
                end_line=20,
                content_hash="wrong-hash-for-the-wide-stale-chunk",
            ),
            _chunk(
                file_path="auth.py",
                start_line=10,
                end_line=12,
                content_hash=content_hash(narrow_chunk_content),
            ),
        ]
        citation = _citation(start_line=10, end_line=12)
        checks = run_mechanical_checks(_draft(citation), tmp_path, index=index)

        # the tightest containing chunk (10-12, fresh) is used, not the wide stale one
        assert checks[0].hash_matches_index is True
        assert checks[0].passed is True


class TestSymbol:
    def test_symbol_none_skips_check(self, tmp_path: Path) -> None:
        lines = ["def foo():", "    return 1"]
        _write(tmp_path, "auth.py", lines)
        content = "\n".join(lines)
        index = [_chunk(file_path="auth.py", start_line=1, end_line=2, content_hash=content_hash(content))]
        citation = _citation(start_line=1, end_line=2, symbol=None)
        checks = run_mechanical_checks(_draft(citation), tmp_path, index=index)
        assert checks[0].symbol_found is None
        assert checks[0].passed is True

    def test_symbol_present_passes(self, tmp_path: Path) -> None:
        lines = ["def foo():", "    return 1"]
        _write(tmp_path, "auth.py", lines)
        content = "\n".join(lines)
        index = [_chunk(file_path="auth.py", start_line=1, end_line=2, content_hash=content_hash(content))]
        citation = _citation(start_line=1, end_line=2, symbol="foo")
        checks = run_mechanical_checks(_draft(citation), tmp_path, index=index)
        assert checks[0].symbol_found is True
        assert checks[0].passed is True

    def test_symbol_absent_fails_passed_only(self, tmp_path: Path) -> None:
        lines = ["def foo():", "    return 1"]
        _write(tmp_path, "auth.py", lines)
        content = "\n".join(lines)
        index = [_chunk(file_path="auth.py", start_line=1, end_line=2, content_hash=content_hash(content))]
        citation = _citation(start_line=1, end_line=2, symbol="bar")
        checks = run_mechanical_checks(_draft(citation), tmp_path, index=index)
        assert checks[0].symbol_found is False
        assert checks[0].file_exists is True
        assert checks[0].lines_in_bounds is True
        assert checks[0].hash_matches_index is True
        assert checks[0].passed is False
