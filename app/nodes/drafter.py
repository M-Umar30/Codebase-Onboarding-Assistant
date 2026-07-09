"""app/nodes/drafter.py — single structured LLM call producing DraftAnswer.

Phase 1: no planner, no critic loop. The question is retrieved against
directly, and the drafter runs exactly once per `ask` call.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings, get_settings
from app.llm import get_chat_model
from app.prompts import build_drafter_prompt
from app.schemas import DraftAnswer, RetrievedChunk


def draft_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    settings: Settings | None = None,
) -> DraftAnswer:
    settings = settings or get_settings()
    system_prompt, user_prompt = build_drafter_prompt(question, retrieved_chunks)

    model = get_chat_model("drafter", settings).with_structured_output(DraftAnswer)
    return model.invoke([SystemMessage(system_prompt), HumanMessage(user_prompt)])
