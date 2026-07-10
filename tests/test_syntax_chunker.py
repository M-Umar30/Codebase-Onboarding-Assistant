"""app/indexing/syntax_chunker.py unit tests over committed fixture files.

Exercises the boundary cases that make code chunking different from prose:
qualified symbols, nested classes, decorated defs (top-level and in-class),
TS arrow-function consts, oversized-def splitting, and — the invariant the
mechanical critic depends on — that every line of a parsed file lands in
exactly one chunk (no gaps, no overlaps).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.indexing.syntax_chunker import MAX_CHUNK_LINES, chunk_file, detect_language
from app.schemas import ChunkKind, CodeChunk, Language

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> tuple[str, list[CodeChunk]]:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return text, chunk_file("repo", name, text)


def _by_symbol(chunks: list[CodeChunk], symbol: str) -> CodeChunk:
    matches = [c for c in chunks if c.symbol == symbol]
    assert len(matches) == 1, f"expected exactly one chunk for {symbol!r}, got {len(matches)}"
    return matches[0]


# --------------------------------------------------------------- language

def test_detect_language() -> None:
    assert detect_language("a/b/mod.py") is Language.PYTHON
    assert detect_language("mod.ts") is Language.TYPESCRIPT
    assert detect_language("component.tsx") is Language.TYPESCRIPT
    assert detect_language("data.json") is Language.OTHER
    assert detect_language("README.md") is Language.OTHER
    assert detect_language("Makefile") is Language.OTHER


# --------------------------------------------------------------- coverage

@pytest.mark.parametrize("name", ["sample.py", "sample.ts"])
def test_chunks_tile_every_line_with_no_gap_or_overlap(name: str) -> None:
    text, chunks = _load(name)
    total = len(text.splitlines())
    spans = sorted((c.start_line, c.end_line) for c in chunks)

    cursor = 1
    for lo, hi in spans:
        assert lo == cursor, f"gap or overlap at line {cursor} (chunk starts at {lo})"
        assert hi >= lo
        cursor = hi + 1
    assert cursor == total + 1, "chunks do not reach the last line"


@pytest.mark.parametrize("name", ["sample.py", "sample.ts"])
def test_content_matches_line_slice_and_hash(name: str) -> None:
    text, chunks = _load(name)
    lines = text.splitlines()
    for c in chunks:
        expected = "\n".join(lines[c.start_line - 1 : c.end_line])
        assert c.content == expected
        assert c.content_hash == hashlib.sha256(expected.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("name,language", [("sample.py", Language.PYTHON), ("sample.ts", Language.TYPESCRIPT)])
def test_parsed_files_are_never_tagged_other(name: str, language: Language) -> None:
    _text, chunks = _load(name)
    assert chunks
    assert all(c.language is language for c in chunks)


@pytest.mark.parametrize("name", ["sample.py", "sample.ts"])
def test_no_chunk_exceeds_max_lines(name: str) -> None:
    _text, chunks = _load(name)
    for c in chunks:
        assert c.end_line - c.start_line + 1 <= MAX_CHUNK_LINES


# --------------------------------------------------------------- module header

@pytest.mark.parametrize("name", ["sample.py", "sample.ts"])
def test_module_header_is_first_and_carries_imports(name: str) -> None:
    _text, chunks = _load(name)
    ordered = sorted(chunks, key=lambda c: c.start_line)
    header = ordered[0]
    assert header.kind is ChunkKind.MODULE_HEADER
    assert header.symbol is None
    assert header.start_line == 1
    assert "import" in header.content


# --------------------------------------------------------------- python symbols

def test_python_top_level_function_and_decorator_span() -> None:
    _text, chunks = _load("sample.py")
    plain = _by_symbol(chunks, "plain_function")
    assert plain.kind is ChunkKind.FUNCTION

    decorated = _by_symbol(chunks, "top_level_decorated")
    assert decorated.kind is ChunkKind.FUNCTION
    assert "@staticmethod" in decorated.content  # decorator included in the span


def test_python_class_and_qualified_methods() -> None:
    _text, chunks = _load("sample.py")
    service = _by_symbol(chunks, "Service")
    assert service.kind is ChunkKind.CLASS

    init = _by_symbol(chunks, "Service.__init__")
    assert init.kind is ChunkKind.METHOD

    teardown = _by_symbol(chunks, "Service.teardown")
    assert teardown.kind is ChunkKind.METHOD


def test_python_decorated_method_includes_decorator() -> None:
    _text, chunks = _load("sample.py")
    label = _by_symbol(chunks, "Service.label")
    assert label.kind is ChunkKind.METHOD
    assert "@property" in label.content


def test_python_nested_class_symbols_are_fully_qualified() -> None:
    _text, chunks = _load("sample.py")
    inner = _by_symbol(chunks, "Service.Config")
    assert inner.kind is ChunkKind.CLASS

    inner_method = _by_symbol(chunks, "Service.Config.describe")
    assert inner_method.kind is ChunkKind.METHOD


def test_python_module_level_wiring_is_block_not_dropped() -> None:
    _text, chunks = _load("sample.py")
    # The `registry[...] = Service` wiring between the class and `oversized`
    # must live in a BLOCK chunk (symbol=None), or a citation to it would have
    # no containing chunk and be judged fabricated.
    wiring = [c for c in chunks if "registry" in c.content and c.kind is ChunkKind.BLOCK]
    assert wiring, "module-level wiring landed in no BLOCK chunk"
    assert all(c.symbol is None for c in wiring)

    trailing = [c for c in chunks if "TRAILING_EXPORT" in c.content]
    assert trailing and trailing[0].kind is ChunkKind.BLOCK


def test_python_oversized_function_splits_at_nested_boundary() -> None:
    _text, chunks = _load("sample.py")
    outer = _by_symbol(chunks, "oversized")
    assert outer.kind is ChunkKind.FUNCTION
    # The nested def was hoisted into its own chunk with a qualified symbol.
    nested = _by_symbol(chunks, "oversized.nested_helper")
    assert nested.kind is ChunkKind.FUNCTION
    assert "nested_helper" in nested.content


# --------------------------------------------------------------- typescript symbols

def test_ts_arrow_const_is_a_named_function() -> None:
    _text, chunks = _load("sample.ts")
    handler = _by_symbol(chunks, "makeHandler")
    assert handler.kind is ChunkKind.FUNCTION
    assert "=>" in handler.content

    exported = _by_symbol(chunks, "exportedFunction")
    assert exported.kind is ChunkKind.FUNCTION
    assert "export function" in exported.content  # export wrapper included


def test_ts_class_methods_are_qualified() -> None:
    _text, chunks = _load("sample.ts")
    assert _by_symbol(chunks, "Service").kind is ChunkKind.CLASS
    assert _by_symbol(chunks, "Service.greet").kind is ChunkKind.METHOD
    assert _by_symbol(chunks, "Service.create").kind is ChunkKind.METHOD
    assert _by_symbol(chunks, "Service.constructor").kind is ChunkKind.METHOD


def test_ts_trailing_default_export_is_covered() -> None:
    _text, chunks = _load("sample.ts")
    trailing = [c for c in chunks if "export default" in c.content]
    assert trailing and trailing[0].kind is ChunkKind.BLOCK


# --------------------------------------------------------------- fallback

def test_unsupported_extension_falls_back_to_line_chunks() -> None:
    chunks = chunk_file("repo", "data.json", '{"a": 1}\n{"b": 2}\n')
    assert chunks
    assert all(c.language is Language.OTHER and c.kind is ChunkKind.BLOCK for c in chunks)


def test_unparseable_python_falls_back_to_line_chunks() -> None:
    broken = "def oops(:\n    this is not valid python (((\n"
    chunks = chunk_file("repo", "broken.py", broken)
    assert chunks
    assert all(c.language is Language.OTHER for c in chunks)


def test_empty_file_yields_no_chunks() -> None:
    assert chunk_file("repo", "empty.py", "") == []
