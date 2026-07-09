"""app/critic/route.py — routing verdict: the critic's core agentic decision.

The LLM only decides route + refined_queries/regeneration_guidance +
reasoning (_RouteDecision) — it never re-emits the verdicts list. The
already-known verdicts (from mechanical.py + semantic.py) are spliced in
when constructing the real CriticVerdict, so the frozen
CriticVerdict.model_validator (the route/payload contract) is the actual
enforcement point, and the LLM has no way to silently drop a citation's
verdict the way a batched re-emission call could.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from app.config import Settings, get_settings
from app.llm import get_chat_model
from app.prompts import build_route_prompt
from app.schemas import CitationVerdict, CriticVerdict, Route


class _RouteDecision(BaseModel):
    """Internal wrapper to bind with_structured_output for the route call.
    Deliberately excludes `verdicts` — see module docstring."""

    route: Route
    refined_queries: list[str] = Field(default_factory=list)
    regeneration_guidance: str | None = None
    reasoning: str


def decide_route(
    question: str,
    verdicts: list[CitationVerdict],
    settings: Settings | None = None,
) -> CriticVerdict:
    settings = settings or get_settings()
    model = get_chat_model("critic", settings).with_structured_output(_RouteDecision)
    system_prompt, user_prompt = build_route_prompt(question, verdicts)
    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

    def _invoke_and_build() -> CriticVerdict:
        decision = model.invoke(messages)
        return CriticVerdict(
            verdicts=verdicts,
            route=decision.route,
            refined_queries=decision.refined_queries,
            regeneration_guidance=decision.regeneration_guidance,
            reasoning=decision.reasoning,
        )

    try:
        return _invoke_and_build()
    except ValidationError as first_error:
        messages.append(
            HumanMessage(
                f"Your previous output was invalid: {first_error}\n"
                "Re-emit a corrected route decision."
            )
        )
        return _invoke_and_build()  # second failure propagates, fail loudly
