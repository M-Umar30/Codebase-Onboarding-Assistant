"""app/critic/route.py tests. The LLM call is monkeypatched — no real
network, matching this project's existing testing convention.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import app.critic.route as route_module
from app.config import Settings
from app.critic.route import _RouteDecision, decide_route
from app.schemas import CitationStatus, CitationVerdict, Route


def _verdict(**overrides: object) -> CitationVerdict:
    defaults = dict(
        citation_id=1, status=CitationStatus.VERIFIED, checked_semantically=True, reasoning="ok"
    )
    defaults.update(overrides)
    return CitationVerdict(**defaults)


class _FakeStructuredModel:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.invocations = 0

    def invoke(self, messages: list) -> _RouteDecision:
        self.invocations += 1
        return self._responses.pop(0)


class _FakeChatModel:
    def __init__(self, structured_model: _FakeStructuredModel) -> None:
        self._structured_model = structured_model

    def with_structured_output(self, schema: object) -> _FakeStructuredModel:
        return self._structured_model


class TestDecideRoute:
    def test_verdicts_are_spliced_in_not_llm_generated(
        self, monkeypatch: pytest.MonkeyPatch, test_settings: Settings
    ) -> None:
        verdicts = [_verdict(citation_id=1), _verdict(citation_id=2)]
        decision = _RouteDecision(route=Route.PROCEED, reasoning="all good")
        fake_model = _FakeStructuredModel([decision])
        monkeypatch.setattr(
            route_module, "get_chat_model", lambda node, settings: _FakeChatModel(fake_model)
        )

        result = decide_route("how does auth work?", verdicts, settings=test_settings)

        assert result.verdicts == verdicts  # identical to input — never LLM-generated
        assert result.route is Route.PROCEED

    def test_retry_once_then_succeeds(self, monkeypatch: pytest.MonkeyPatch, test_settings: Settings) -> None:
        verdicts = [_verdict(citation_id=1, status=CitationStatus.FABRICATED, checked_semantically=False)]
        # bad_decision constructs fine on its own (_RouteDecision has no cross-field
        # validator) but fails when spliced into CriticVerdict: RE_RETRIEVE requires
        # refined_queries, which is empty here. That's the frozen schema validator
        # acting as the actual enforcement.
        bad_decision = _RouteDecision(route=Route.RE_RETRIEVE, refined_queries=[], reasoning="evidence gap")
        good_decision = _RouteDecision(
            route=Route.RE_RETRIEVE, refined_queries=["narrower query"], reasoning="evidence gap"
        )
        fake_model = _FakeStructuredModel([bad_decision, good_decision])
        monkeypatch.setattr(
            route_module, "get_chat_model", lambda node, settings: _FakeChatModel(fake_model)
        )

        result = decide_route("how does auth work?", verdicts, settings=test_settings)

        assert fake_model.invocations == 2
        assert result.route is Route.RE_RETRIEVE
        assert result.refined_queries == ["narrower query"]

    def test_two_consecutive_failures_propagate(
        self, monkeypatch: pytest.MonkeyPatch, test_settings: Settings
    ) -> None:
        verdicts = [_verdict(citation_id=1, status=CitationStatus.FABRICATED, checked_semantically=False)]
        bad_decision = _RouteDecision(route=Route.RE_RETRIEVE, refined_queries=[], reasoning="evidence gap")
        fake_model = _FakeStructuredModel([bad_decision, bad_decision])
        monkeypatch.setattr(
            route_module, "get_chat_model", lambda node, settings: _FakeChatModel(fake_model)
        )

        with pytest.raises(ValidationError):
            decide_route("how does auth work?", verdicts, settings=test_settings)
