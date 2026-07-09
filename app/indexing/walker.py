"""app/indexing/walker.py — repo file discovery for Phase 1's naive indexer.

Honors nested .gitignore files (git semantics: a directory's .gitignore
only governs paths beneath it) plus a hardcoded skip list for
vendored/generated directories that repos often omit from .gitignore
entirely (e.g. a checked-out node_modules, or this project's own .venv).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pathspec

HARD_SKIP_DIR_NAMES = {
    "node_modules",
    ".git",
    "dist",
    "build",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    ".env",
}

# First-bytes binary sniff: presence of a NUL byte is the classic
# git/Perl heuristic for "don't treat this as text".
_BINARY_SNIFF_BYTES = 8192


def _load_gitignore_spec(dir_path: Path) -> pathspec.GitIgnoreSpec | None:
    gitignore_path = dir_path / ".gitignore"
    if not gitignore_path.is_file():
        return None
    lines = gitignore_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return pathspec.GitIgnoreSpec.from_lines(lines)


def _matches_any_spec(
    path: Path, is_dir: bool, specs: list[tuple[Path, pathspec.GitIgnoreSpec]]
) -> bool:
    for spec_dir, spec in specs:
        try:
            rel = path.relative_to(spec_dir)
        except ValueError:
            continue
        posix_rel = rel.as_posix() + ("/" if is_dir else "")
        if spec.match_file(posix_rel):
            return True
    return False


def _is_binary(file_path: Path) -> bool:
    try:
        with file_path.open("rb") as f:
            chunk = f.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return True
    return b"\x00" in chunk


def iter_source_files(repo_root: Path) -> Iterator[Path]:
    """Yield text file paths under repo_root, honoring .gitignore and
    skipping vendored/generated/binary content. Paths are absolute."""
    repo_root = repo_root.resolve()
    specs: list[tuple[Path, pathspec.GitIgnoreSpec]] = []

    root_spec = _load_gitignore_spec(repo_root)
    if root_spec:
        specs.append((repo_root, root_spec))

    for dirpath, dirnames, filenames in os.walk(repo_root):
        current_dir = Path(dirpath)

        dirnames[:] = sorted(d for d in dirnames if d not in HARD_SKIP_DIR_NAMES)

        if current_dir != repo_root:
            local_spec = _load_gitignore_spec(current_dir)
            if local_spec:
                specs.append((current_dir, local_spec))

        dirnames[:] = [
            d
            for d in dirnames
            if not _matches_any_spec(current_dir / d, is_dir=True, specs=specs)
        ]

        for filename in sorted(filenames):
            file_path = current_dir / filename
            if _matches_any_spec(file_path, is_dir=False, specs=specs):
                continue
            if _is_binary(file_path):
                continue
            yield file_path
