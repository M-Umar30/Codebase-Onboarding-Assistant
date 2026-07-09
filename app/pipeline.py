"""app/pipeline.py — Phase 1 ask pipeline: retrieve -> draft -> synthesize.

Plain function composition, no LangGraph. That arrives in Phase 3 once
there's an actual conditional edge (the critic's route) to justify a graph.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.nodes.drafter import draft_answer
from app.nodes.synthesizer import synthesize_answer
from app.retrieval.retriever import retrieve
from app.schemas import FinalAnswer


def ask(repo_id: str, question: str, settings: Settings | None = None) -> FinalAnswer:
    settings = settings or get_settings()
    retrieved_chunks = retrieve(repo_id, question, settings=settings)
    draft = draft_answer(question, retrieved_chunks, settings=settings)
    return synthesize_answer(question, draft, settings=settings)
