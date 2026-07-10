"""Drafter prompt-builder tests — pure functions, no LLM. Verifies the
critic's regeneration_guidance is injected only when present."""

from __future__ import annotations

from app.prompts import build_drafter_prompt
from app.schemas import ChunkKind, CodeChunk, Language, RetrievedChunk


def _chunk() -> RetrievedChunk:
    chunk = CodeChunk(
        id="c1",
        repo_id="r",
        file_path="auth.py",
        start_line=1,
        end_line=10,
        language=Language.OTHER,
        kind=ChunkKind.BLOCK,
        symbol=None,
        content="def authenticate(): ...",
        content_hash="h",
    )
    return RetrievedChunk(chunk=chunk, dense_score=0.9, fused_score=0.9, sub_query_id=1)


def test_no_guidance_leaves_user_prompt_clean() -> None:
    _system, user = build_drafter_prompt("how does auth work?", [_chunk()])

    assert "reviewer" not in user.lower()
    assert "how does auth work?" in user


def test_guidance_is_injected_when_present() -> None:
    _system, user = build_drafter_prompt(
        "how does auth work?", [_chunk()], guidance="Cite the token check, not the router."
    )

    assert "Cite the token check, not the router." in user
    assert "reviewer" in user.lower()


def test_empty_guidance_is_treated_as_absent() -> None:
    _system, user = build_drafter_prompt("q", [_chunk()], guidance="")

    assert "reviewer" not in user.lower()
