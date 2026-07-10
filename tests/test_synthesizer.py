"""app/nodes/synthesizer.py tests — the LLM call is monkeypatched (no network),
matching the project convention. These verify the Python-owned guarantees:
verified-only citations, deterministic notes/caveat, and the post-validation
retry-once-then-fail-loudly path.
"""

from __future__ import annotations

import pytest

import app.nodes.synthesizer as synth_module
from app.config import Settings
from app.nodes.synthesizer import BUDGET_CAVEAT, ZERO_VERIFIED_ANSWER, _SynthesizedAnswer, synthesize_answer
from app.schemas import Citation, CitationStatus, CitationVerdict, DraftAnswer

QUESTION = "how does auth work here?"


def _citation(**overrides: object) -> Citation:
    defaults = dict(
        id=1, file_path="auth.py", start_line=1, end_line=10, symbol=None, claim="validates the token"
    )
    defaults.update(overrides)
    return Citation(**defaults)


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

    def invoke(self, messages: list) -> _SynthesizedAnswer:
        self.invocations += 1
        return self._responses.pop(0)


class _FakeChatModel:
    def __init__(self, structured_model: _FakeStructuredModel) -> None:
        self._structured_model = structured_model

    def with_structured_output(self, schema: object) -> _FakeStructuredModel:
        return self._structured_model


def _patch_model(monkeypatch: pytest.MonkeyPatch, fake_model: _FakeStructuredModel) -> None:
    monkeypatch.setattr(
        synth_module, "get_chat_model", lambda node, settings: _FakeChatModel(fake_model)
    )


class TestSynthesizeAnswer:
    def test_drops_unverified_and_records_deterministic_note(
        self, monkeypatch: pytest.MonkeyPatch, test_settings: Settings
    ) -> None:
        c1 = _citation(id=1, claim="validates the token")
        c2 = _citation(id=2, file_path="ghost.py", start_line=5, end_line=8, claim="rate limits requests")
        draft = DraftAnswer(answer_markdown="Auth validates [1] and rate limits [2].", citations=[c1, c2])
        verdicts = [
            _verdict(citation_id=1, status=CitationStatus.VERIFIED),
            _verdict(citation_id=2, status=CitationStatus.FABRICATED, checked_semantically=False),
        ]
        # LLM keeps only c1, renumbered to 1.
        response = _SynthesizedAnswer(
            answer_markdown="Auth validates the token [1].",
            citations=[_citation(id=1)],
        )
        fake_model = _FakeStructuredModel([response])
        _patch_model(monkeypatch, fake_model)

        final = synthesize_answer(QUESTION, draft, verdicts, settings=test_settings)

        assert [c.id for c in final.citations] == [1]
        assert final.citations[0].file_path == "auth.py"
        assert final.unverified_notes == [
            'Could not verify: "rate limits requests" (cited ghost.py:5-8; fabricated)'
        ]
        assert final.confidence_caveat is None

    def test_budget_exhausted_sets_caveat_deterministically(
        self, monkeypatch: pytest.MonkeyPatch, test_settings: Settings
    ) -> None:
        c1 = _citation(id=1)
        draft = DraftAnswer(answer_markdown="Auth validates [1].", citations=[c1])
        verdicts = [_verdict(citation_id=1, status=CitationStatus.VERIFIED)]
        response = _SynthesizedAnswer(answer_markdown="Auth validates [1].", citations=[_citation(id=1)])
        _patch_model(monkeypatch, _FakeStructuredModel([response]))

        final = synthesize_answer(QUESTION, draft, verdicts, budget_exhausted=True, settings=test_settings)

        assert final.confidence_caveat == BUDGET_CAVEAT

    def test_zero_verified_short_circuits_without_llm(
        self, monkeypatch: pytest.MonkeyPatch, test_settings: Settings
    ) -> None:
        c1 = _citation(id=1, file_path="ghost.py")
        draft = DraftAnswer(answer_markdown="Auth does X [1].", citations=[c1])
        verdicts = [_verdict(citation_id=1, status=CitationStatus.FABRICATED, checked_semantically=False)]
        fake_model = _FakeStructuredModel([])  # must never be popped
        _patch_model(monkeypatch, fake_model)

        final = synthesize_answer(QUESTION, draft, verdicts, settings=test_settings)

        assert fake_model.invocations == 0
        assert final.citations == []
        assert final.answer_markdown == ZERO_VERIFIED_ANSWER
        assert len(final.unverified_notes) == 1

    def test_citing_unverified_location_triggers_retry_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, test_settings: Settings
    ) -> None:
        c1 = _citation(id=1)
        draft = DraftAnswer(answer_markdown="Auth validates [1].", citations=[c1])
        verdicts = [_verdict(citation_id=1, status=CitationStatus.VERIFIED)]
        bad = _SynthesizedAnswer(
            answer_markdown="Auth does X [1].",
            citations=[_citation(id=1, file_path="not-verified.py")],  # location not in verified set
        )
        good = _SynthesizedAnswer(answer_markdown="Auth validates [1].", citations=[_citation(id=1)])
        fake_model = _FakeStructuredModel([bad, good])
        _patch_model(monkeypatch, fake_model)

        final = synthesize_answer(QUESTION, draft, verdicts, settings=test_settings)

        assert fake_model.invocations == 2
        assert final.citations[0].file_path == "auth.py"

    def test_marker_id_mismatch_triggers_retry(
        self, monkeypatch: pytest.MonkeyPatch, test_settings: Settings
    ) -> None:
        c1 = _citation(id=1)
        draft = DraftAnswer(answer_markdown="Auth validates [1].", citations=[c1])
        verdicts = [_verdict(citation_id=1, status=CitationStatus.VERIFIED)]
        # answer references [2] but only citation id 1 exists -> marker/id mismatch
        bad = _SynthesizedAnswer(answer_markdown="Auth validates [2].", citations=[_citation(id=1)])
        good = _SynthesizedAnswer(answer_markdown="Auth validates [1].", citations=[_citation(id=1)])
        fake_model = _FakeStructuredModel([bad, good])
        _patch_model(monkeypatch, fake_model)

        final = synthesize_answer(QUESTION, draft, verdicts, settings=test_settings)

        assert fake_model.invocations == 2

    def test_two_consecutive_bad_outputs_propagate(
        self, monkeypatch: pytest.MonkeyPatch, test_settings: Settings
    ) -> None:
        c1 = _citation(id=1)
        draft = DraftAnswer(answer_markdown="Auth validates [1].", citations=[c1])
        verdicts = [_verdict(citation_id=1, status=CitationStatus.VERIFIED)]
        bad = _SynthesizedAnswer(
            answer_markdown="X [1].", citations=[_citation(id=1, file_path="not-verified.py")]
        )
        fake_model = _FakeStructuredModel([bad, bad])
        _patch_model(monkeypatch, fake_model)

        with pytest.raises(ValueError):
            synthesize_answer(QUESTION, draft, verdicts, settings=test_settings)
