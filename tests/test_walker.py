"""app/indexing/walker.py tests — .gitignore honored (root + nested),
hardcoded vendored/generated dir skips, and binary-file skipping.
"""

from __future__ import annotations

from pathlib import Path

from app.indexing.walker import iter_source_files


def _write(path: Path, content: str = "content\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _relative_results(repo_root: Path) -> set[str]:
    return {p.relative_to(repo_root).as_posix() for p in iter_source_files(repo_root)}


def test_root_gitignore_excludes_matching_files(tmp_path: Path) -> None:
    _write(tmp_path / ".gitignore", "ignored.py\nignored_dir/\n")
    _write(tmp_path / "kept.py")
    _write(tmp_path / "ignored.py")
    _write(tmp_path / "ignored_dir" / "also_kept_but_ignored.py")

    results = _relative_results(tmp_path)

    assert "kept.py" in results
    assert "ignored.py" not in results
    assert "ignored_dir/also_kept_but_ignored.py" not in results


def test_nested_gitignore_only_applies_under_its_directory(tmp_path: Path) -> None:
    _write(tmp_path / "top_level_secret.py")  # not ignored: pattern is scoped to subdir/
    _write(tmp_path / "subdir" / ".gitignore", "secret.py\n")
    _write(tmp_path / "subdir" / "secret.py")
    _write(tmp_path / "subdir" / "public.py")

    results = _relative_results(tmp_path)

    assert "top_level_secret.py" in results
    assert "subdir/secret.py" not in results
    assert "subdir/public.py" in results


def test_hardcoded_vendor_dirs_are_skipped_even_without_gitignore(tmp_path: Path) -> None:
    _write(tmp_path / "node_modules" / "pkg" / "index.js")
    _write(tmp_path / ".git" / "config")
    _write(tmp_path / "__pycache__" / "mod.cpython-312.pyc")
    _write(tmp_path / ".venv" / "Lib" / "site.py")
    _write(tmp_path / "dist" / "bundle.js")
    _write(tmp_path / "build" / "out.js")
    _write(tmp_path / "src" / "app.py")

    results = _relative_results(tmp_path)

    assert results == {"src/app.py"}


def test_binary_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02\x03garbage")
    _write(tmp_path / "text.py")

    results = _relative_results(tmp_path)

    assert "text.py" in results
    assert "binary.bin" not in results
