"""app/nodes/synthesizer.py — single structured LLM call producing FinalAnswer.

Phase 1: no critic yet, so every draft citation passes through unchanged;
this node only renumbers citations/markers into the final answer. Phase 3
wires this behind the critic's `proceed` route and starts actually dropping
unverified claims.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings, get_settings
from app.llm import get_chat_model
from app.prompts import build_synthesizer_prompt
from app.schemas import DraftAnswer, FinalAnswer


def synthesize_answer(
    question: str,
    draft: DraftAnswer,
    settings: Settings | None = None,
) -> FinalAnswer:
    settings = settings or get_settings()
    system_prompt, user_prompt = build_synthesizer_prompt(question, draft)

    model = get_chat_model("synthesizer", settings).with_structured_output(FinalAnswer)
    return model.invoke([SystemMessage(system_prompt), HumanMessage(user_prompt)])
