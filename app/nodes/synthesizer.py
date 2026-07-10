"""app/nodes/synthesizer.py — final answer from verified citations only.

The critic decides which citations are trustworthy; this node enforces that
decision. Mirroring app/critic/route.py's splice pattern, the LLM is given a
narrow job (rewrite the prose and renumber the surviving citations) and every
trust-sensitive field is owned by Python, never the model:

- verified vs dropped is partitioned here from the critic's verdicts;
- unverified_notes is built deterministically from the dropped citations;
- confidence_caveat is set deterministically when the graph exhausted its
  iteration budget;
- the LLM's structured output targets a local wrapper WITHOUT the notes/caveat
  fields, so it physically cannot write them.

A post-validation pass guarantees the final citations are a subset of the
verified set and that inline [n] markers line up 1..n — folded into the same
retry-once-then-fail-loudly path the critic's semantic/route layers use.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings
from app.llm import get_chat_model
from app.prompts import build_synthesizer_prompt
from app.schemas import Citation, CitationStatus, CitationVerdict, DraftAnswer, FinalAnswer

_MARKER_RE = re.compile(r"\[(\d+)\]")

BUDGET_CAVEAT = (
    "Verification budget exhausted (the critic hit its iteration limit); "
    "unverified claims were dropped and this answer may be incomplete."
)

ZERO_VERIFIED_ANSWER = (
    "I could not verify any of the evidence found for this question, so I "
    "won't state an answer. See the unverified notes for what was checked "
    "and rejected."
)


class _SynthesizedAnswer(BaseModel):
    """Internal wrapper to bind with_structured_output. Deliberately excludes
    unverified_notes and confidence_caveat — those are Python-owned, so the LLM
    has no field to write them into. Not part of the frozen app.schemas."""

    answer_markdown: str
    citations: list[Citation]


def _unverified_note(citation: Citation, verdict: CitationVerdict) -> str:
    return (
        f'Could not verify: "{citation.claim}" '
        f"(cited {citation.file_path}:{citation.start_line}-{citation.end_line}; "
        f"{verdict.status.value})"
    )


def _validate_synthesis(result: _SynthesizedAnswer, verified: list[Citation]) -> None:
    """Raise ValueError if the LLM's output isn't a clean renumbering of the
    verified citations. Trust-sensitive, so enforced in Python not the prompt."""
    verified_locations = {
        (c.file_path, c.start_line, c.end_line) for c in verified
    }
    unknown = [
        c
        for c in result.citations
        if (c.file_path, c.start_line, c.end_line) not in verified_locations
    ]
    if unknown:
        raise ValueError(
            "Final answer cites locations that were not verified: "
            f"{[(c.file_path, c.start_line, c.end_line) for c in unknown]}"
        )

    ids = [c.id for c in result.citations]
    expected = list(range(1, len(result.citations) + 1))
    if ids != expected:
        raise ValueError(
            f"Final citation ids must be 1..n in order; got {ids}, expected {expected}"
        )

    marker_ids = {int(m) for m in _MARKER_RE.findall(result.answer_markdown)}
    citation_ids = set(ids)
    if marker_ids != citation_ids:
        raise ValueError(
            f"Inline [n] markers {sorted(marker_ids)} must match citation ids "
            f"{sorted(citation_ids)} exactly (both directions)."
        )


def synthesize_answer(
    question: str,
    draft: DraftAnswer,
    verdicts: list[CitationVerdict],
    budget_exhausted: bool = False,
    settings: Settings | None = None,
) -> FinalAnswer:
    settings = settings or get_settings()

    verdict_by_id = {v.citation_id: v for v in verdicts}
    verified: list[Citation] = []
    dropped: list[tuple[Citation, CitationVerdict]] = []
    for citation in draft.citations:
        verdict = verdict_by_id.get(citation.id)
        if verdict is not None and verdict.status is CitationStatus.VERIFIED:
            verified.append(citation)
        elif verdict is not None:
            dropped.append((citation, verdict))

    unverified_notes = [_unverified_note(c, v) for c, v in dropped]
    confidence_caveat = BUDGET_CAVEAT if budget_exhausted else None

    # No verified evidence: saying "I don't know" beats an unverified guess
    # (README policy). No LLM call — FinalAnswer.citations allows empty.
    if not verified:
        return FinalAnswer(
            answer_markdown=ZERO_VERIFIED_ANSWER,
            citations=[],
            unverified_notes=unverified_notes,
            confidence_caveat=confidence_caveat,
        )

    model = get_chat_model("synthesizer", settings).with_structured_output(_SynthesizedAnswer)
    system_prompt, user_prompt = build_synthesizer_prompt(question, draft, verified, dropped)
    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

    def _invoke_and_validate() -> _SynthesizedAnswer:
        result = model.invoke(messages)
        _validate_synthesis(result, verified)
        return result

    try:
        result = _invoke_and_validate()
    except (ValidationError, ValueError) as first_error:
        messages.append(
            HumanMessage(
                f"Your previous output was invalid: {first_error}\n"
                "Re-emit the final answer using ONLY the verified citations, "
                "renumbered 1..n by first appearance, with matching inline [n] markers."
            )
        )
        result = _invoke_and_validate()  # second failure propagates, fail loudly

    return FinalAnswer(
        answer_markdown=result.answer_markdown,
        citations=result.citations,
        unverified_notes=unverified_notes,
        confidence_caveat=confidence_caveat,
    )
