"""app/indexing/syntax_chunker.py — Phase 4 tree-sitter chunking.

Python and TS/TSX files are chunked on function/class/method boundaries with
qualified symbol names; a leading MODULE_HEADER chunk carries imports +
docstring + module constants. Everything else (unsupported languages, or a
supported file that fails to parse) falls back to the naive line-window
chunker in app/indexing/chunker.py (Language.OTHER / ChunkKind.BLOCK), which
the indexer counts into IndexResponse.fallback_language_files.

Coverage invariant: for a supported file that parses, the emitted chunks tile
line 1..total_lines with no gap and no overlap. This matters beyond
tidiness — the mechanical critic (app/critic/mechanical.py) verifies a
citation by finding the chunk that *contains* its lines and re-hashing them.
A line that belonged to no chunk would make a citation to real code look
fabricated, so interstitial/trailing module-level code (route registrations,
`app = FastAPI()`, bottom-of-file exports) and class-level code between
methods are emitted as BLOCK chunks rather than dropped.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from tree_sitter import Language, Node, Parser

import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript

from app.indexing.chunker import chunk_id, chunk_text, content_hash
from app.schemas import ChunkKind, CodeChunk, Language as Lang

# A definition longer than this many lines is split at its nested boundaries
# (nested functions/methods become their own chunks); a def with no nested
# boundary that is still oversized is line-window split so no chunk is unbounded.
MAX_CHUNK_LINES = 120

LANGUAGE_BY_EXT = {
    ".py": Lang.PYTHON,
    ".ts": Lang.TYPESCRIPT,
    ".tsx": Lang.TYPESCRIPT,
}

_PY_LANGUAGE = Language(tspython.language())
_TS_LANGUAGE = Language(tstypescript.language_typescript())
_TSX_LANGUAGE = Language(tstypescript.language_tsx())


def detect_language(file_path: str) -> Lang:
    """Language from the file extension; anything unlisted is OTHER (line fallback)."""
    return LANGUAGE_BY_EXT.get(PurePosixPath(file_path).suffix.lower(), Lang.OTHER)


def _parser_for(language: Lang, file_path: str) -> Parser:
    if language is Lang.PYTHON:
        return Parser(_PY_LANGUAGE)
    if PurePosixPath(file_path).suffix.lower() == ".tsx":
        return Parser(_TSX_LANGUAGE)
    return Parser(_TS_LANGUAGE)


def chunk_file(repo_id: str, file_path: str, text: str) -> list[CodeChunk]:
    """Chunk one file. Dispatches on extension: Python/TS/TSX go through
    tree-sitter, everything else (and any parse failure) uses the line-window
    fallback, which tags chunks Language.OTHER."""
    language = detect_language(file_path)
    if language is Lang.OTHER:
        return chunk_text(repo_id, file_path, text)

    lines = text.splitlines()
    if not lines:
        return []

    try:
        parser = _parser_for(language, file_path)
        tree = parser.parse(text.encode("utf-8"))
        if tree.root_node.has_error:
            # A syntactically broken supported file: prefer complete, if coarse,
            # coverage over a partial syntax tree. Falls back (counted as OTHER).
            return chunk_text(repo_id, file_path, text)

        builder = _ChunkBuilder(repo_id, file_path, language, lines)
        builder.partition(
            tree.root_node,
            lo=1,
            hi=len(lines),
            scope="",
            leading_kind=ChunkKind.MODULE_HEADER,
            leading_symbol=None,
            container_is_class=False,
        )
        return builder.chunks
    except Exception:
        # tree-sitter should not raise on valid input, but never let an indexer
        # run die on one odd file — fall back and keep going.
        return chunk_text(repo_id, file_path, text)


# --------------------------------------------------------------------------- #
# Node classification
# --------------------------------------------------------------------------- #

# Container node whose *body children* hold the nested definitions.
_PY_BODY_FIELD = "body"
_TS_CLASS_BODY = {"class_body"}
_TS_STATEMENT_BLOCK = {"statement_block"}


class _Def:
    """A definition child of a container: its decorator-inclusive line span,
    kind, unqualified name, and the node to recurse into."""

    __slots__ = ("lo", "hi", "kind", "name", "body_node", "always_split")

    def __init__(
        self,
        lo: int,
        hi: int,
        kind: ChunkKind,
        name: str | None,
        body_node: Node | None,
        always_split: bool,
    ) -> None:
        self.lo = lo
        self.hi = hi
        self.kind = kind
        self.name = name
        self.body_node = body_node  # container to recurse into when split
        self.always_split = always_split  # classes always emit per-method chunks


def _line_span(node: Node) -> tuple[int, int]:
    """1-indexed inclusive line span. A node ending at column 0 of a line
    actually ends on the previous line (tree-sitter's exclusive end point)."""
    start = node.start_point[0] + 1
    end_row, end_col = node.end_point
    end = end_row if end_col == 0 and end_row > 0 else end_row + 1
    return start, end


def _named_child_of_type(node: Node, types: set[str]) -> Node | None:
    for child in node.named_children:
        if child.type in types:
            return child
    return None


def _identifier_text(node: Node | None, field: str = "name") -> str | None:
    if node is None:
        return None
    named = node.child_by_field_name(field)
    if named is not None:
        return named.text.decode("utf-8")
    # Fall back to the first identifier-ish child.
    for child in node.named_children:
        if child.type in {"identifier", "type_identifier", "property_identifier"}:
            return child.text.decode("utf-8")
    return None


_PY_FUNC = "function_definition"
_PY_CLASS = "class_definition"
_FUNC_VALUE_TYPES = {"arrow_function", "function_expression"}


def _classify(node: Node, container_is_class: bool) -> _Def | None:
    """Map a container's body child to a _Def, or None if it isn't a definition
    (imports, module-level statements, class fields — these become interstitial
    MODULE_HEADER/BLOCK regions so every line still lands in a chunk)."""
    t = node.type

    # --- Python ---
    if t == "decorated_definition":
        lo, hi = _line_span(node)  # decorator-inclusive span
        inner = node.child_by_field_name("definition") or _named_child_of_type(
            node, {_PY_FUNC, _PY_CLASS}
        )
        if inner is None:
            return None
        if inner.type == _PY_CLASS:
            return _Def(lo, hi, ChunkKind.CLASS, _identifier_text(inner), inner, True)
        kind = ChunkKind.METHOD if container_is_class else ChunkKind.FUNCTION
        return _Def(lo, hi, kind, _identifier_text(inner), inner, False)
    if t == _PY_FUNC:
        lo, hi = _line_span(node)
        kind = ChunkKind.METHOD if container_is_class else ChunkKind.FUNCTION
        return _Def(lo, hi, kind, _identifier_text(node), node, False)
    if t == _PY_CLASS:
        lo, hi = _line_span(node)
        return _Def(lo, hi, ChunkKind.CLASS, _identifier_text(node), node, True)

    # --- TypeScript / TSX ---
    if t == "export_statement":
        decl = node.child_by_field_name("declaration")
        if decl is None:
            return None  # `export { ... }` / `export default expr` — not a def
        inner = _classify(decl, container_is_class)
        if inner is None:
            return None
        inner.lo, inner.hi = _line_span(node)  # widen span to include `export`
        return inner
    if t in {"function_declaration", "generator_function_declaration"}:
        lo, hi = _line_span(node)
        kind = ChunkKind.METHOD if container_is_class else ChunkKind.FUNCTION
        return _Def(lo, hi, kind, _identifier_text(node), node, False)
    if t in {"class_declaration", "abstract_class_declaration"}:
        lo, hi = _line_span(node)
        return _Def(lo, hi, ChunkKind.CLASS, _identifier_text(node), node, True)
    if t == "method_definition":
        lo, hi = _line_span(node)
        return _Def(lo, hi, ChunkKind.METHOD, _identifier_text(node), node, False)
    if t in {"lexical_declaration", "variable_declaration"}:
        declarator = _named_child_of_type(node, {"variable_declarator"})
        if declarator is None:
            return None
        value = declarator.child_by_field_name("value")
        if value is None or value.type not in _FUNC_VALUE_TYPES:
            return None  # `const X = 1` — module data, not a def
        lo, hi = _line_span(node)
        return _Def(lo, hi, ChunkKind.FUNCTION, _identifier_text(declarator), value, False)
    if t == "public_field_definition" and container_is_class:
        value = node.child_by_field_name("value")
        if value is not None and value.type in _FUNC_VALUE_TYPES:
            lo, hi = _line_span(node)
            return _Def(lo, hi, ChunkKind.METHOD, _identifier_text(node), value, False)
        return None

    return None


def _body_of(node: Node) -> Node:
    """The node whose named children hold the nested definitions."""
    if node.type in {"module", "program"}:
        return node
    body = node.child_by_field_name("body")
    if body is not None:
        return body
    return _named_child_of_type(node, {"statement_block", "class_body", "block"}) or node


class _ChunkBuilder:
    def __init__(self, repo_id: str, file_path: str, language: Lang, lines: list[str]) -> None:
        self.repo_id = repo_id
        self.file_path = file_path
        self.language = language
        self.lines = lines
        self.chunks: list[CodeChunk] = []

    def _append(self, lo: int, hi: int, kind: ChunkKind, symbol: str | None) -> None:
        content = "\n".join(self.lines[lo - 1 : hi])
        hash_ = content_hash(content)
        self.chunks.append(
            CodeChunk(
                id=chunk_id(self.repo_id, self.file_path, lo, hash_),
                repo_id=self.repo_id,
                file_path=self.file_path,
                start_line=lo,
                end_line=hi,
                language=self.language,
                kind=kind,
                symbol=symbol,
                content=content,
                content_hash=hash_,
            )
        )

    def _emit(self, lo: int, hi: int, kind: ChunkKind, symbol: str | None) -> None:
        """Emit region [lo, hi] as one chunk, or several non-overlapping window
        pieces if it exceeds MAX_CHUNK_LINES (head keeps the kind/symbol,
        continuations are BLOCK). Non-overlapping keeps the coverage tiling exact."""
        if hi < lo:
            return
        cursor = lo
        first = True
        while cursor <= hi:
            end = min(cursor + MAX_CHUNK_LINES - 1, hi)
            self._append(
                cursor, end, kind if first else ChunkKind.BLOCK, symbol if first else None
            )
            first = False
            cursor = end + 1

    def partition(
        self,
        container: Node,
        lo: int,
        hi: int,
        scope: str,
        leading_kind: ChunkKind,
        leading_symbol: str | None,
        container_is_class: bool,
    ) -> None:
        """Tile [lo, hi] with the definitions found directly inside `container`,
        filling the leading region with `leading_kind` and every other
        interstitial/trailing region with BLOCK."""
        body = _body_of(container)
        defs = [
            d for d in (_classify(c, container_is_class) for c in body.named_children) if d is not None
        ]
        defs.sort(key=lambda d: d.lo)

        cursor = lo
        first_region = True
        for d in defs:
            d_lo = max(d.lo, lo)
            d_hi = min(d.hi, hi)
            if d_hi < d_lo or d_lo < cursor:
                continue  # overlapping/degenerate span (defensive) — skip
            if d_lo > cursor:
                if first_region:
                    self._emit(cursor, d_lo - 1, leading_kind, leading_symbol)
                else:
                    self._emit(cursor, d_lo - 1, ChunkKind.BLOCK, None)
            first_region = False
            self._emit_def(d, d_lo, d_hi, scope)
            cursor = d_hi + 1

        if cursor <= hi:
            # No defs at all -> the whole container is its leading region
            # (module with only imports/constants, class with only fields).
            kind = leading_kind if first_region else ChunkKind.BLOCK
            symbol = leading_symbol if first_region else None
            self._emit(cursor, hi, kind, symbol)

    def _emit_def(self, d: _Def, lo: int, hi: int, scope: str) -> None:
        if d.name and scope:
            qualified: str | None = f"{scope}.{d.name}"
        else:
            qualified = d.name or scope or None
        oversized = (hi - lo + 1) > MAX_CHUNK_LINES
        if d.body_node is not None and (d.always_split or oversized):
            # Classes always split into per-method chunks; functions only when
            # oversized (split at nested boundaries).
            self.partition(
                d.body_node,
                lo=lo,
                hi=hi,
                scope=qualified or scope,
                leading_kind=d.kind,
                leading_symbol=qualified,
                container_is_class=(d.kind is ChunkKind.CLASS),
            )
        else:
            self._emit(lo, hi, d.kind, qualified)
