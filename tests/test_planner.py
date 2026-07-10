"""app/nodes/planner.py tests — decompose-or-skip with a mocked LLM.

Only the LLM boundary (get_chat_model) is stubbed; the Python-owned logic —
id assignment, the skip/decompose collapse rules, the 1..4 clamp, and the
retry-once-then-fail path — runs for real.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import app.nodes.planner as planner_module
from app.config import Settings
from app.nodes.planner import _PlanDecision, _SubQueryDraft, plan_question

QUESTION = "how does auth work here?"


class _FakeStructuredModel:
    """Pops a scripted item per invoke; an Exception item is raised (to drive
    the retry path), anything else is returned as the structured output."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.invocations = 0

    def invoke(self, messages: list):
        self.invocations += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeChatModel:
    def __init__(self, structured: _FakeStructuredModel) -> None:
        self._structured = structured

    def with_structured_output(self, schema: object) -> _FakeStructuredModel:
        return self._structured


def _patch(monkeypatch: pytest.MonkeyPatch, responses: list) -> _FakeStructuredModel:
    fake = _FakeStructuredModel(responses)
    monkeypatch.setattr(planner_module, "get_chat_model", lambda node, settings: _FakeChatModel(fake))
    return fake


def _draft(query: str) -> _SubQueryDraft:
    return _SubQueryDraft(query=query, rationale=f"because {query}")


def _settings() -> Settings:
    return Settings(_env_file=None, openai_api_key="k", groq_api_key="k")


def _validation_error() -> ValidationError:
    try:
        _PlanDecision.model_validate({})  # missing required fields
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def test_skip_mirrors_question_as_single_sub_query(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [_PlanDecision(decomposed=False, sub_queries=[], reasoning="simple lookup")])

    plan = plan_question(QUESTION, settings=_settings())

    assert plan.decomposed is False
    assert len(plan.sub_queries) == 1
    assert plan.sub_queries[0].id == 1
    assert plan.sub_queries[0].query == QUESTION
    assert plan.reasoning == "simple lookup"


def test_decompose_assigns_sequential_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(
        monkeypatch,
        [
            _PlanDecision(
                decomposed=True,
                sub_queries=[_draft("routes"), _draft("middleware"), _draft("token validation")],
                reasoning="three facets",
            )
        ],
    )

    plan = plan_question(QUESTION, settings=_settings())

    assert plan.decomposed is True
    assert [s.id for s in plan.sub_queries] == [1, 2, 3]
    assert [s.query for s in plan.sub_queries] == ["routes", "middleware", "token validation"]


def test_decompose_is_clamped_to_four_sub_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(
        monkeypatch,
        [
            _PlanDecision(
                decomposed=True,
                sub_queries=[_draft(f"q{i}") for i in range(6)],
                reasoning="too many",
            )
        ],
    )

    plan = plan_question(QUESTION, settings=_settings())

    assert len(plan.sub_queries) == 4
    assert [s.id for s in plan.sub_queries] == [1, 2, 3, 4]


def test_decompose_with_one_usable_query_collapses_to_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    # Model said decompose but only produced a single (or blank-padded) query —
    # not worth a multi-query fan-out, so mirror the question instead.
    _patch(
        monkeypatch,
        [
            _PlanDecision(
                decomposed=True,
                sub_queries=[_draft("only one"), _SubQueryDraft(query="   ", rationale="blank")],
                reasoning="thin decomposition",
            )
        ],
    )

    plan = plan_question(QUESTION, settings=_settings())

    assert plan.decomposed is False
    assert len(plan.sub_queries) == 1
    assert plan.sub_queries[0].query == QUESTION


def test_retries_once_on_invalid_output(monkeypatch: pytest.MonkeyPatch) -> None:
    good = _PlanDecision(decomposed=False, sub_queries=[], reasoning="ok on retry")
    fake = _patch(monkeypatch, [_validation_error(), good])

    plan = plan_question(QUESTION, settings=_settings())

    assert fake.invocations == 2
    assert plan.reasoning == "ok on retry"


def test_second_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _patch(monkeypatch, [_validation_error(), _validation_error()])

    with pytest.raises(ValidationError):
        plan_question(QUESTION, settings=_settings())

    assert fake.invocations == 2
