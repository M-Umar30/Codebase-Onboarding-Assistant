"""Smoke tests for app/schemas.py (frozen). Confirms every schema imports
and constructs, and exercises the CriticVerdict route validators — the
mechanism the critic's conditional edge in LangGraph will depend on.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    AskRequest,
    AskResponse,
    ChunkKind,
    Citation,
    CitationStatus,
    CitationVerdict,
    CodeChunk,
    CriticVerdict,
    DraftAnswer,
    FinalAnswer,
    IndexRequest,
    IndexResponse,
    IterationTrace,
    Language,
    MechanicalCheck,
    Plan,
    RetrievedChunk,
    Route,
    SubQuery,
    Trace,
)


def _citation(**overrides: object) -> Citation:
    defaults = dict(
        id=1,
        file_path="app/auth.py",
        start_line=10,
        end_line=20,
        symbol="AuthService.refresh_token",
        claim="Validates the refresh token signature.",
    )
    defaults.update(overrides)
    return Citation(**defaults)


def _code_chunk(**overrides: object) -> CodeChunk:
    defaults = dict(
        id="a" * 16,
        repo_id="repo-1",
        file_path="app/auth.py",
        start_line=10,
        end_line=20,
        language=Language.PYTHON,
        kind=ChunkKind.FUNCTION,
        symbol="AuthService.refresh_token",
        content="def refresh_token(): ...",
        content_hash="deadbeef",
    )
    defaults.update(overrides)
    return CodeChunk(**defaults)


def _citation_verdict(**overrides: object) -> CitationVerdict:
    defaults = dict(
        citation_id=1,
        status=CitationStatus.VERIFIED,
        checked_semantically=True,
        reasoning="Code matches the claim.",
    )
    defaults.update(overrides)
    return CitationVerdict(**defaults)


class TestCodeChunkAndRetrieval:
    def test_code_chunk_constructs(self) -> None:
        chunk = _code_chunk()
        assert chunk.language is Language.PYTHON
        assert chunk.kind is ChunkKind.FUNCTION

    def test_code_chunk_rejects_inverted_lines(self) -> None:
        with pytest.raises(ValidationError):
            _code_chunk(start_line=20, end_line=10)

    def test_retrieved_chunk_constructs(self) -> None:
        retrieved = RetrievedChunk(
            chunk=_code_chunk(),
            dense_score=0.8,
            lexical_score=None,
            fused_score=0.5,
            sub_query_id=1,
        )
        assert retrieved.lexical_score is None


class TestPlanner:
    def test_plan_constructs(self) -> None:
        plan = Plan(
            decomposed=True,
            sub_queries=[SubQuery(id=1, query="where are routes defined?", rationale="entry point")],
            reasoning="Question spans multiple concerns.",
        )
        assert len(plan.sub_queries) == 1

    def test_plan_requires_at_least_one_sub_query(self) -> None:
        with pytest.raises(ValidationError):
            Plan(decomposed=False, sub_queries=[], reasoning="n/a")


class TestDrafter:
    def test_citation_rejects_inverted_lines(self) -> None:
        with pytest.raises(ValidationError):
            _citation(start_line=20, end_line=10)

    def test_draft_answer_requires_citations(self) -> None:
        with pytest.raises(ValidationError):
            DraftAnswer(answer_markdown="Auth works via JWT.", citations=[])

    def test_draft_answer_constructs(self) -> None:
        draft = DraftAnswer(answer_markdown="Auth works via JWT [1].", citations=[_citation()])
        assert draft.citations[0].id == 1


class TestCritic:
    def test_mechanical_check_constructs(self) -> None:
        check = MechanicalCheck(
            citation_id=1,
            file_exists=True,
            lines_in_bounds=True,
            symbol_found=True,
            hash_matches_index=True,
            passed=True,
        )
        assert check.passed

    def test_proceed_route_requires_no_extra_payload(self) -> None:
        verdict = CriticVerdict(
            verdicts=[_citation_verdict()],
            route=Route.PROCEED,
            reasoning="All citations verified.",
        )
        assert verdict.route is Route.PROCEED
        assert verdict.refined_queries == []
        assert verdict.regeneration_guidance is None

    def test_re_retrieve_requires_refined_queries(self) -> None:
        with pytest.raises(ValidationError):
            CriticVerdict(
                verdicts=[_citation_verdict(status=CitationStatus.FABRICATED, checked_semantically=False)],
                route=Route.RE_RETRIEVE,
                refined_queries=[],
                reasoning="Evidence gap.",
            )

    def test_re_retrieve_with_refined_queries_constructs(self) -> None:
        verdict = CriticVerdict(
            verdicts=[_citation_verdict(status=CitationStatus.FABRICATED, checked_semantically=False)],
            route=Route.RE_RETRIEVE,
            refined_queries=["where is refresh_token validated?"],
            reasoning="Evidence gap.",
        )
        assert verdict.route is Route.RE_RETRIEVE

    def test_regenerate_requires_guidance(self) -> None:
        with pytest.raises(ValidationError):
            CriticVerdict(
                verdicts=[_citation_verdict(status=CitationStatus.UNSUPPORTED_CLAIM)],
                route=Route.REGENERATE,
                regeneration_guidance=None,
                reasoning="Evidence fine, draft misused it.",
            )

    def test_regenerate_with_guidance_constructs(self) -> None:
        verdict = CriticVerdict(
            verdicts=[_citation_verdict(status=CitationStatus.UNSUPPORTED_CLAIM)],
            route=Route.REGENERATE,
            regeneration_guidance="Re-read the cited function before restating its behavior.",
            reasoning="Evidence fine, draft misused it.",
        )
        assert verdict.route is Route.REGENERATE

    def test_verdicts_requires_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            CriticVerdict(verdicts=[], route=Route.PROCEED, reasoning="n/a")


class TestSynthesisAndTrace:
    def test_final_answer_constructs(self) -> None:
        answer = FinalAnswer(answer_markdown="Auth works via JWT [1].", citations=[_citation()])
        assert answer.unverified_notes == []

    def test_iteration_trace_and_trace_construct(self) -> None:
        plan = Plan(
            decomposed=False,
            sub_queries=[SubQuery(id=1, query="how does auth work?", rationale="direct lookup")],
            reasoning="Simple lookup, no decomposition needed.",
        )
        verdict = CriticVerdict(
            verdicts=[_citation_verdict()], route=Route.PROCEED, reasoning="All verified."
        )
        iteration = IterationTrace(iteration=1, chunks_retrieved=5, critic=verdict)
        trace = Trace(
            plan=plan,
            iterations=[iteration],
            budget_exhausted=False,
            models_used={"planner": "llama-3.3-70b-versatile"},
        )
        assert trace.iterations[0].critic.route is Route.PROCEED


class TestApiAndCli:
    def test_index_request_and_response_construct(self) -> None:
        req = IndexRequest(source="https://github.com/example/repo")
        resp = IndexResponse(
            repo_id="repo-1",
            files_indexed=10,
            files_skipped_unchanged=2,
            chunks_written=42,
            fallback_language_files=1,
        )
        assert req.repo_id is None
        assert resp.chunks_written == 42

    def test_ask_request_and_response_construct(self) -> None:
        req = AskRequest(repo_id="repo-1", question="how does auth work here?")
        resp = AskResponse(
            answer=FinalAnswer(answer_markdown="Auth works via JWT [1].", citations=[_citation()]),
            trace=None,
        )
        assert req.include_trace is False
        assert resp.trace is None
