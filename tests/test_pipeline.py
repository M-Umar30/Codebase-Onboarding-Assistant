"""app/pipeline.py test — orchestration only, LLM nodes mocked out.

No network call happens: retrieve/draft_answer/synthesize_answer are
monkeypatched to canned outputs, so this test verifies that `ask()` wires
question/repo_id/settings through each stage correctly and returns the
synthesizer's result, not that any particular model produces good text.
"""

from __future__ import annotations

import pytest

from app import pipeline
from app.config import Settings
from app.schemas import (
    ChunkKind,
    Citation,
    CodeChunk,
    DraftAnswer,
    FinalAnswer,
    Language,
    RetrievedChunk,
)

REPO_ID = "mock-repo"
QUESTION = "how does auth work here?"


def _retrieved_chunk() -> RetrievedChunk:
    chunk = CodeChunk(
        id="abc123",
        repo_id=REPO_ID,
        file_path="auth/middleware.py",
        start_line=10,
        end_line=25,
        language=Language.OTHER,
        kind=ChunkKind.BLOCK,
        symbol=None,
        content="def authenticate(request): ...",
        content_hash="hash",
    )
    return RetrievedChunk(chunk=chunk, dense_score=0.9, lexical_score=None, fused_score=0.9, sub_query_id=1)


def _draft_answer() -> DraftAnswer:
    return DraftAnswer(
        answer_markdown="Auth is handled by middleware [1].",
        citations=[
            Citation(
                id=1,
                file_path="auth/middleware.py",
                start_line=10,
                end_line=25,
                symbol=None,
                claim="Validates the request's auth token.",
            )
        ],
    )


def _final_answer() -> FinalAnswer:
    return FinalAnswer(
        answer_markdown="Auth is handled by middleware [1].",
        citations=[
            Citation(
                id=1,
                file_path="auth/middleware.py",
                start_line=10,
                end_line=25,
                symbol=None,
                claim="Validates the request's auth token.",
            )
        ],
        unverified_notes=[],
        confidence_caveat=None,
    )


def test_ask_wires_retrieve_draft_and_synthesize(monkeypatch: pytest.MonkeyPatch, test_settings: Settings) -> None:
    retrieve_calls = []
    draft_calls = []
    synthesize_calls = []

    retrieved = [_retrieved_chunk()]
    draft = _draft_answer()
    final = _final_answer()

    def fake_retrieve(repo_id, question, settings=None):
        retrieve_calls.append((repo_id, question, settings))
        return retrieved

    def fake_draft_answer(question, retrieved_chunks, settings=None):
        draft_calls.append((question, retrieved_chunks, settings))
        return draft

    def fake_synthesize_answer(question, draft_arg, settings=None):
        synthesize_calls.append((question, draft_arg, settings))
        return final

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "draft_answer", fake_draft_answer)
    monkeypatch.setattr(pipeline, "synthesize_answer", fake_synthesize_answer)

    result = pipeline.ask(REPO_ID, QUESTION, settings=test_settings)

    assert result is final
    assert retrieve_calls == [(REPO_ID, QUESTION, test_settings)]
    assert draft_calls == [(QUESTION, retrieved, test_settings)]
    assert synthesize_calls == [(QUESTION, draft, test_settings)]
