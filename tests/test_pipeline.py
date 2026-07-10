"""app/pipeline.py test — the Phase-3 façade.

pipeline.ask() is now a thin wrapper over graph.ask_with_trace: it forwards
(repo_id, question, settings) and returns just the .answer. The graph itself is
covered by test_graph.py; here we only assert the façade wiring, with
ask_with_trace monkeypatched so no graph/DB/network runs.
"""

from __future__ import annotations

import pytest

from app import pipeline
from app.config import Settings
from app.schemas import AskResponse, Citation, FinalAnswer, Plan, SubQuery, Trace

REPO_ID = "mock-repo"
QUESTION = "how does auth work here?"


def _final_answer() -> FinalAnswer:
    return FinalAnswer(
        answer_markdown="Auth is handled by middleware [1].",
        citations=[
            Citation(
                id=1,
                file_path="auth/middleware.py",
                start_line=10,
                end_line=25,
                claim="Validates the request's auth token.",
            )
        ],
    )


def _ask_response() -> AskResponse:
    return AskResponse(
        answer=_final_answer(),
        trace=Trace(
            plan=Plan(
                decomposed=False,
                sub_queries=[SubQuery(id=1, query=QUESTION, rationale="r")],
                reasoning="r",
            ),
            iterations=[],
            budget_exhausted=False,
            models_used={},
        ),
    )


def test_ask_forwards_to_graph_and_returns_answer(
    monkeypatch: pytest.MonkeyPatch, test_settings: Settings
) -> None:
    calls = []
    response = _ask_response()

    def fake_ask_with_trace(repo_id, question, settings=None, conn=None, embedder=None):
        calls.append((repo_id, question, settings))
        return response

    monkeypatch.setattr(pipeline, "ask_with_trace", fake_ask_with_trace)

    result = pipeline.ask(REPO_ID, QUESTION, settings=test_settings)

    assert result is response.answer
    assert calls == [(REPO_ID, QUESTION, test_settings)]
