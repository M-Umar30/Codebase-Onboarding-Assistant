"""app/pipeline.py — the ask entry point.

Phase 3 replaced the straight-through composition with a LangGraph graph (see
app/graph.py). This module is now a thin façade: `ask()` keeps the Phase-1
signature for callers that only want the answer, and `ask_with_trace` is
re-exported for the CLI/API surfaces that also want the execution trace.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.graph import ask_with_trace
from app.schemas import FinalAnswer

__all__ = ["ask", "ask_with_trace"]


def ask(repo_id: str, question: str, settings: Settings | None = None) -> FinalAnswer:
    settings = settings or get_settings()
    return ask_with_trace(repo_id, question, settings=settings).answer
