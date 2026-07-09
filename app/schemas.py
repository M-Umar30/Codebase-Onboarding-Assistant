"""app/schemas.py — FROZEN.

The contract between every phase of the build. Do not add, remove, or rename
fields without a human-approved edit to CLAUDE.md.

Conventions (load-bearing, referenced by validators and the critic):
- Line numbers are 1-indexed, inclusive on both ends.
- Citations are referenced in answer text as inline markers: [1], [2], ...
  matching Citation.id. The drafter MUST emit markers; the synthesizer
  renumbers after unverified citations are dropped.
- Embeddings and FTS vectors live in the database only — never in these
  models. These schemas are the app-level contract, not the storage layout.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------- enums


class Language(str, Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    OTHER = "other"  # line-based fallback chunking; logged at index time


class ChunkKind(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE_HEADER = "module_header"  # imports + docstring + top-level context
    BLOCK = "block"  # fallback for Language.OTHER or oversized splits


class CitationStatus(str, Enum):
    VERIFIED = "verified"
    WRONG_LOCATION = "wrong_location"      # file/lines exist but symbol/claim isn't there
    UNSUPPORTED_CLAIM = "unsupported_claim"  # code is real, claim misstates it
    FABRICATED = "fabricated"              # file/lines/symbol don't exist at all


class Route(str, Enum):
    PROCEED = "proceed"        # ship to synthesizer (it drops unverified citations)
    RE_RETRIEVE = "re_retrieve"  # evidence gap — refined_queries required
    REGENERATE = "regenerate"    # evidence fine, draft misused it — guidance required


# ------------------------------------------------------------- indexing


class CodeChunk(BaseModel):
    """One indexed unit of code. Produced by the chunker, stored in Postgres."""

    id: str  # deterministic: sha256(repo_id + file_path + start_line + content_hash)[:16]
    repo_id: str
    file_path: str  # repo-relative, POSIX separators
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    language: Language
    kind: ChunkKind
    symbol: str | None = None  # qualified name, e.g. "AuthService.refresh_token"
    content: str
    content_hash: str  # sha256 of content; drives incremental re-index

    @model_validator(mode="after")
    def _line_order(self) -> "CodeChunk":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class RetrievedChunk(BaseModel):
    """A chunk plus retrieval provenance. Retriever output; drafter input."""

    chunk: CodeChunk
    dense_score: float | None = None    # None when dense side didn't return it
    lexical_score: float | None = None  # None when lexical side didn't return it
    fused_score: float                  # RRF score; sort key
    sub_query_id: int                   # which SubQuery pulled this in


# -------------------------------------------------------------- planner


class SubQuery(BaseModel):
    id: int = Field(ge=1)
    query: str
    rationale: str  # one line: why this sub-question serves the user's question


class Plan(BaseModel):
    """Planner output (structured LLM call)."""

    decomposed: bool  # False => single sub-query mirroring the raw question
    sub_queries: list[SubQuery] = Field(min_length=1, max_length=4)
    reasoning: str


# -------------------------------------------------------------- drafter


class Citation(BaseModel):
    """A claim tied to a specific code location. The `claim` field is what
    makes the critic's job checkable — it states what the code supposedly does."""

    id: int = Field(ge=1)  # matches inline [n] markers in the answer text
    file_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    symbol: str | None = None
    claim: str  # one sentence: what this code does, per the draft

    @model_validator(mode="after")
    def _line_order(self) -> "Citation":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class DraftAnswer(BaseModel):
    """Drafter output (structured LLM call). Every factual statement about the
    codebase must carry an inline [n] marker with a matching Citation."""

    answer_markdown: str
    citations: list[Citation] = Field(min_length=1)


# --------------------------------------------------------------- critic


class MechanicalCheck(BaseModel):
    """Deterministic verification result for one citation. Zero LLM involvement."""

    citation_id: int
    file_exists: bool
    lines_in_bounds: bool
    symbol_found: bool | None = None  # None when citation has no symbol
    hash_matches_index: bool
    passed: bool  # conjunction of the above (symbol_found treated as True if None)


class CitationVerdict(BaseModel):
    """Per-citation judgment. Mechanical failures are mapped to a status
    without an LLM call; only mechanically-valid citations get the semantic check."""

    citation_id: int
    status: CitationStatus
    checked_semantically: bool  # False => status derived from mechanical layer alone
    reasoning: str


class CriticVerdict(BaseModel):
    """Critic node output. `route` is THE agentic decision — a LangGraph
    conditional edge reads it directly. Iteration budget is enforced by the
    graph as a safety rail, never by this schema."""

    verdicts: list[CitationVerdict] = Field(min_length=1)
    route: Route
    refined_queries: list[str] = Field(default_factory=list)  # required for RE_RETRIEVE
    regeneration_guidance: str | None = None  # required for REGENERATE
    reasoning: str

    @model_validator(mode="after")
    def _route_payloads(self) -> "CriticVerdict":
        if self.route is Route.RE_RETRIEVE and not self.refined_queries:
            raise ValueError("re_retrieve requires at least one refined query")
        if self.route is Route.REGENERATE and not self.regeneration_guidance:
            raise ValueError("regenerate requires regeneration_guidance")
        return self


# ------------------------------------------------------------ synthesis


class FinalAnswer(BaseModel):
    """Synthesizer output. Citations here are verified-only, renumbered 1..n.
    Claims that lost their citation are either dropped or surfaced in
    unverified_notes — never silently kept."""

    answer_markdown: str
    citations: list[Citation]
    unverified_notes: list[str] = Field(default_factory=list)
    confidence_caveat: str | None = None  # set when budget exhausted (partial answer)


# ---------------------------------------------------------------- trace


class IterationTrace(BaseModel):
    iteration: int = Field(ge=1)
    chunks_retrieved: int
    critic: CriticVerdict


class Trace(BaseModel):
    """Full graph-execution record. Rendered by `--show-trace`; returned by the
    API when requested. This is the interview demo artifact."""

    plan: Plan
    iterations: list[IterationTrace]
    budget_exhausted: bool
    models_used: dict[str, str]  # node name -> model identifier (pinned in config)


# ------------------------------------------------------------ API + CLI


class IndexRequest(BaseModel):
    source: str  # local path or git URL
    repo_id: str | None = None  # default: derived from source


class IndexResponse(BaseModel):
    repo_id: str
    files_indexed: int
    files_skipped_unchanged: int  # the incremental re-index receipt
    chunks_written: int
    fallback_language_files: int  # files chunked line-based (Language.OTHER)


class AskRequest(BaseModel):
    repo_id: str
    question: str
    include_trace: bool = False


class AskResponse(BaseModel):
    answer: FinalAnswer
    trace: Trace | None = None