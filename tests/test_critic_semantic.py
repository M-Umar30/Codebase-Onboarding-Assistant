"""app/critic/semantic.py tests. The deterministic mapping is pure (no LLM).
The batched LLM call is monkeypatched — matching this project's existing
test_pipeline.py/test_llm.py convention of no real network calls in pytest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.critic.semantic as semantic_module
from app.config import Settings
from app.critic.semantic import _SemanticBatch, _map_mechanical_failure, _validate_batch, run_semantic_checks
from app.schemas import Citation, CitationStatus, CitationVerdict, DraftAnswer, MechanicalCheck


def _check(**overrides: object) -> MechanicalCheck:
    defaults = dict(
        citation_id=1,
        file_exists=True,
        lines_in_bounds=True,
        symbol_found=None,
        hash_matches_index=True,
        passed=True,
    )
    defaults.update(overrides)
    return MechanicalCheck(**defaults)


def _citation(**overrides: object) -> Citation:
    defaults = dict(
        id=1, file_path="auth.py", start_line=1, end_line=2, symbol=None, claim="does something"
    )
    defaults.update(overrides)
    return Citation(**defaults)


def _verdict(**overrides: object) -> CitationVerdict:
    defaults = dict(
        citation_id=1, status=CitationStatus.VERIFIED, checked_semantically=True, reasoning="ok"
    )
    defaults.update(overrides)
    return CitationVerdict(**defaults)


class TestMapMechanicalFailure:
    def test_missing_file_is_fabricated(self) -> None:
        check = _check(file_exists=False, lines_in_bounds=False, hash_matches_index=False, passed=False)
        assert _map_mechanical_failure(check) == CitationStatus.FABRICATED

    def test_out_of_bounds_is_fabricated(self) -> None:
        check = _check(lines_in_bounds=False, hash_matches_index=False, passed=False)
        assert _map_mechanical_failure(check) == CitationStatus.FABRICATED

    def test_no_containing_chunk_is_fabricated(self) -> None:
        check = _check(hash_matches_index=False, passed=False)
        assert _map_mechanical_failure(check) == CitationStatus.FABRICATED

    def test_symbol_not_found_is_wrong_location(self) -> None:
        check = _check(symbol_found=False, passed=False)
        assert _map_mechanical_failure(check) == CitationStatus.WRONG_LOCATION

    def test_all_passing_returns_none(self) -> None:
        assert _map_mechanical_failure(_check()) is None


class TestValidateBatch:
    def test_matching_ids_and_statuses_passes(self) -> None:
        batch = _SemanticBatch(verdicts=[_verdict(citation_id=1), _verdict(citation_id=2)])
        _validate_batch(batch, expected_citation_ids={1, 2})  # no raise

    def test_dropped_citation_raises(self) -> None:
        batch = _SemanticBatch(verdicts=[_verdict(citation_id=1)])
        with pytest.raises(ValueError):
            _validate_batch(batch, expected_citation_ids={1, 2})

    def test_extra_citation_raises(self) -> None:
        batch = _SemanticBatch(verdicts=[_verdict(citation_id=1), _verdict(citation_id=2)])
        with pytest.raises(ValueError):
            _validate_batch(batch, expected_citation_ids={1})

    def test_out_of_domain_status_raises(self) -> None:
        batch = _SemanticBatch(verdicts=[_verdict(citation_id=1, status=CitationStatus.FABRICATED)])
        with pytest.raises(ValueError):
            _validate_batch(batch, expected_citation_ids={1})

    def test_not_checked_semantically_raises(self) -> None:
        batch = _SemanticBatch(verdicts=[_verdict(citation_id=1, checked_semantically=False)])
        with pytest.raises(ValueError):
            _validate_batch(batch, expected_citation_ids={1})


class _FakeStructuredModel:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.invocations = 0

    def invoke(self, messages: list) -> _SemanticBatch:
        self.invocations += 1
        return self._responses.pop(0)


class _FakeChatModel:
    def __init__(self, structured_model: _FakeStructuredModel) -> None:
        self._structured_model = structured_model

    def with_structured_output(self, schema: object) -> _FakeStructuredModel:
        return self._structured_model


class TestRunSemanticChecksOrchestration:
    def test_mechanically_failed_citations_skip_the_llm(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, test_settings: Settings
    ) -> None:
        def fail_if_called(*args: object, **kwargs: object) -> None:
            raise AssertionError("get_chat_model must not be called with no mechanically-passed citations")

        monkeypatch.setattr(semantic_module, "get_chat_model", fail_if_called)

        citation = _citation(id=1)
        draft = DraftAnswer(answer_markdown="x [1].", citations=[citation])
        checks = [_check(citation_id=1, file_exists=False, hash_matches_index=False, passed=False)]

        verdicts = run_semantic_checks(draft, checks, tmp_path, settings=test_settings)

        assert len(verdicts) == 1
        assert verdicts[0].status == CitationStatus.FABRICATED
        assert verdicts[0].checked_semantically is False

    def test_dropped_citation_triggers_retry_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, test_settings: Settings
    ) -> None:
        (tmp_path / "auth.py").write_text("line1\nline2\n", encoding="utf-8")

        citation = _citation(id=1, file_path="auth.py", start_line=1, end_line=2)
        draft = DraftAnswer(answer_markdown="x [1].", citations=[citation])
        checks = [_check(citation_id=1, passed=True)]

        bad_batch = _SemanticBatch(verdicts=[])  # dropped citation 1
        good_batch = _SemanticBatch(verdicts=[_verdict(citation_id=1, status=CitationStatus.VERIFIED)])
        fake_model = _FakeStructuredModel([bad_batch, good_batch])
        monkeypatch.setattr(
            semantic_module, "get_chat_model", lambda node, settings: _FakeChatModel(fake_model)
        )

        verdicts = run_semantic_checks(draft, checks, tmp_path, settings=test_settings)

        assert fake_model.invocations == 2
        assert verdicts[0].status == CitationStatus.VERIFIED

    def test_two_consecutive_failures_propagate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, test_settings: Settings
    ) -> None:
        (tmp_path / "auth.py").write_text("line1\nline2\n", encoding="utf-8")

        citation = _citation(id=1, file_path="auth.py", start_line=1, end_line=2)
        draft = DraftAnswer(answer_markdown="x [1].", citations=[citation])
        checks = [_check(citation_id=1, passed=True)]

        bad_batch = _SemanticBatch(verdicts=[])
        fake_model = _FakeStructuredModel([bad_batch, bad_batch])
        monkeypatch.setattr(
            semantic_module, "get_chat_model", lambda node, settings: _FakeChatModel(fake_model)
        )

        with pytest.raises(ValueError):
            run_semantic_checks(draft, checks, tmp_path, settings=test_settings)
