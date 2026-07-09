"""app/critic/semantic.py — LLM semantic verification + deterministic
mechanical-failure mapping.

Mechanically-failed citations never reach the LLM: their CitationStatus is
derived deterministically from MechanicalCheck (see _map_mechanical_failure).
Only mechanically-passed citations get a real semantic judgment, batched into
one structured call. The batch response is validated for completeness (every
requested citation_id present, no extras) and status domain (only
verified/unsupported_claim) — both folded into the same retry-once path as
malformed structured output, since a dropped citation is a well-formed but
semantically incomplete response and must trigger the same recovery as a
schema-validation failure.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings
from app.critic.mechanical import read_cited_lines
from app.llm import get_chat_model
from app.prompts import build_semantic_prompt
from app.schemas import Citation, CitationStatus, CitationVerdict, DraftAnswer, MechanicalCheck

_SEMANTIC_ALLOWED_STATUSES = {CitationStatus.VERIFIED, CitationStatus.UNSUPPORTED_CLAIM}


class _SemanticBatch(BaseModel):
    """Internal wrapper to bind with_structured_output for the batched
    semantic call. Not part of the frozen app.schemas contract."""

    verdicts: list[CitationVerdict]


def _map_mechanical_failure(check: MechanicalCheck) -> CitationStatus | None:
    """None means the citation passed mechanically and needs a semantic
    check; otherwise the deterministic status for a mechanical failure."""
    if not check.file_exists:
        return CitationStatus.FABRICATED
    if not check.lines_in_bounds:
        return CitationStatus.FABRICATED
    if not check.hash_matches_index:
        return CitationStatus.FABRICATED
    if check.symbol_found is False:
        return CitationStatus.WRONG_LOCATION
    return None


def _mechanical_failure_reasoning(check: MechanicalCheck) -> str:
    if not check.file_exists:
        return "Cited file does not exist in the repo."
    if not check.lines_in_bounds:
        return "Cited line range is out of bounds for the file."
    if not check.hash_matches_index:
        return (
            "No indexed chunk covers this citation's line range "
            "(or the covering chunk is stale)."
        )
    return "Cited symbol was not found within the cited line range."


def _validate_batch(batch: _SemanticBatch, expected_citation_ids: set[int]) -> None:
    actual_ids = {verdict.citation_id for verdict in batch.verdicts}
    if actual_ids != expected_citation_ids:
        raise ValueError(
            f"Expected verdicts for citation_ids {sorted(expected_citation_ids)}, "
            f"got {sorted(actual_ids)}"
        )
    bad_status = [v.citation_id for v in batch.verdicts if v.status not in _SEMANTIC_ALLOWED_STATUSES]
    if bad_status:
        raise ValueError(
            "Semantic layer must only return verified/unsupported_claim; "
            f"citation_ids {bad_status} returned a different status"
        )
    not_checked = [v.citation_id for v in batch.verdicts if not v.checked_semantically]
    if not_checked:
        raise ValueError(
            f"checked_semantically must be true for LLM-produced verdicts; "
            f"citation_ids {not_checked} returned false"
        )


def run_semantic_checks(
    draft: DraftAnswer,
    mechanical_checks: list[MechanicalCheck],
    repo_root: Path,
    settings: Settings | None = None,
) -> list[CitationVerdict]:
    settings = settings or get_settings()
    checks_by_id = {check.citation_id: check for check in mechanical_checks}

    verdicts: list[CitationVerdict] = []
    passed_citations: list[Citation] = []
    for citation in draft.citations:
        check = checks_by_id[citation.id]
        if check.passed:
            passed_citations.append(citation)
            continue
        status = _map_mechanical_failure(check)
        assert status is not None  # check.passed is False => a failure status always exists
        verdicts.append(
            CitationVerdict(
                citation_id=citation.id,
                status=status,
                checked_semantically=False,
                reasoning=_mechanical_failure_reasoning(check),
            )
        )

    if passed_citations:
        citations_with_content = [
            (
                citation,
                read_cited_lines(
                    repo_root, citation.file_path, citation.start_line, citation.end_line
                )
                or "",
            )
            for citation in passed_citations
        ]
        expected_ids = {citation.id for citation in passed_citations}
        model = get_chat_model("critic", settings).with_structured_output(_SemanticBatch)
        system_prompt, user_prompt = build_semantic_prompt(citations_with_content)
        messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

        def _invoke_and_validate() -> _SemanticBatch:
            batch = model.invoke(messages)
            _validate_batch(batch, expected_ids)
            return batch

        try:
            batch = _invoke_and_validate()
        except (ValidationError, ValueError) as first_error:
            messages.append(
                HumanMessage(
                    f"Your previous output was invalid: {first_error}\n"
                    f"Re-emit a corrected response covering exactly citation_ids "
                    f"{sorted(expected_ids)}, with status only 'verified' or "
                    f"'unsupported_claim' and checked_semantically true for each."
                )
            )
            batch = _invoke_and_validate()  # second failure propagates, fail loudly

        verdicts.extend(batch.verdicts)

    return sorted(verdicts, key=lambda v: v.citation_id)
