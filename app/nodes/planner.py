"""app/nodes/planner.py — the planner: decompose-or-skip as an LLM decision.

The model decides whether a question is a single-search lookup or a broad
question worth splitting into 2-4 retrieval sub-queries, and drafts those
sub-queries (_PlanDecision). Python owns the trust-sensitive part — the
SubQuery ids (1..n) are always assigned here, never taken from the model, and
the count is clamped to the frozen Plan.sub_queries bounds (1..4). This mirrors
the route/synthesizer nodes: internal wrapper for with_structured_output,
Python-owned fields, invoke -> validate -> retry once -> fail loudly.

Wired as the graph's entry node (app/graph.py): its Plan populates the trace
and its sub_queries drive the retriever's fan-out.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from app.config import Settings, get_settings
from app.llm import get_chat_model
from app.prompts import build_planner_prompt
from app.schemas import Plan, SubQuery

# Frozen Plan.sub_queries bound; the model's list is clamped to this.
_MAX_SUB_QUERIES = 4


class _SubQueryDraft(BaseModel):
    """One planned sub-query, minus the id (assigned in Python)."""

    query: str
    rationale: str


class _PlanDecision(BaseModel):
    """Internal wrapper binding with_structured_output for the planner call.
    Excludes SubQuery ids — see module docstring."""

    decomposed: bool
    sub_queries: list[_SubQueryDraft] = Field(default_factory=list)
    reasoning: str


def plan_question(question: str, settings: Settings | None = None) -> Plan:
    settings = settings or get_settings()
    model = get_chat_model("planner", settings).with_structured_output(_PlanDecision)
    system_prompt, user_prompt = build_planner_prompt(question)
    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

    def _invoke_and_build() -> Plan:
        decision = model.invoke(messages)
        return _to_plan(question, decision)

    try:
        return _invoke_and_build()
    except ValidationError as first_error:
        messages.append(
            HumanMessage(
                f"Your previous output was invalid: {first_error}\n"
                "Re-emit a corrected plan decision."
            )
        )
        return _invoke_and_build()  # second failure propagates, fail loudly


def _to_plan(question: str, decision: _PlanDecision) -> Plan:
    """Turn the model's decision into a validated Plan with Python-assigned ids.

    A skip (or a decompose that produced <2 usable sub-queries) collapses to a
    single sub-query mirroring the question verbatim; a decompose is truncated
    to the frozen 1..4 bound. Ids are always 1..n by position."""
    drafts = [d for d in decision.sub_queries if d.query.strip()]

    if not decision.decomposed or len(drafts) < 2:
        return Plan(
            decomposed=False,
            sub_queries=[
                SubQuery(
                    id=1,
                    query=question,
                    rationale="Single-search lookup: the question is used verbatim as the query.",
                )
            ],
            reasoning=decision.reasoning,
        )

    drafts = drafts[:_MAX_SUB_QUERIES]
    return Plan(
        decomposed=True,
        sub_queries=[
            SubQuery(id=i, query=d.query, rationale=d.rationale)
            for i, d in enumerate(drafts, start=1)
        ],
        reasoning=decision.reasoning,
    )
