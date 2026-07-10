"""app/graph.py — the Phase 3 LangGraph pipeline: retrieve -> draft -> critic,
with the critic's CriticVerdict.route driving a conditional edge back to
retrieve (re_retrieve) or draft (regenerate), or forward to synthesize
(proceed).

This is the project's core agentic claim: the model's routing verdict — not an
`if attempts < 3` — picks the next graph edge. The iteration budget
(MAX_CRITIC_ITERATIONS) is a graph-enforced safety rail only: the critic always
runs and its verdict is always recorded in the trace, but a non-PROCEED verdict
on the final allowed iteration is overruled and routed to synthesize with
budget_exhausted=True. The trace therefore shows what the model *wanted* even
when the graph capped it.

No planner yet (Phase 4): the entry query is the raw question. Trace.plan is a
single-sub-query placeholder so the frozen (non-optional) schema is satisfied.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.critic.mechanical import run_mechanical_checks
from app.critic.route import decide_route
from app.critic.semantic import run_semantic_checks
from app.db import get_connection
from app.embeddings import Embedder
from app.nodes.drafter import draft_answer
from app.nodes.synthesizer import synthesize_answer
from app.retrieval.retriever import DEFAULT_TOP_K, retrieve
from app.retrieval.store import load_chunks, load_repo_root
from app.schemas import (
    AskResponse,
    CodeChunk,
    CriticVerdict,
    DraftAnswer,
    FinalAnswer,
    IterationTrace,
    Plan,
    RetrievedChunk,
    Route,
    SubQuery,
    Trace,
)

# Max critic passes per ask (initial pass + up to 2 loop-backs). A safety rail,
# never a schema/prompt concern. Worst case is 3*(retrieve|draft -> critic) +
# synthesize ~= 10 supersteps, comfortably under LangGraph's default
# recursion_limit of 25 — keep that margin in mind if this is ever raised.
MAX_CRITIC_ITERATIONS = 3

# Nodes named for the graph. Note: the Route enum values ("re_retrieve",
# "regenerate") are NOT node names — the conditional edge maps them explicitly.
_RETRIEVE = "retrieve"
_DRAFT = "draft"
_CRITIC = "critic"
_SYNTHESIZE = "synthesize"


class GraphState(TypedDict):
    """The graph runs strictly sequentially (no parallel branches), so plain
    last-write-wins reducers are correct; the critic node appends to
    `iterations` explicitly. If parallel retrieval is ever added, `iterations`
    and `retrieved` would need real merge reducers."""

    question: str
    repo_id: str
    queries: list[str]  # current retrieval queries; [question] at entry, overwritten by refined_queries
    retrieved: list[RetrievedChunk]
    draft: DraftAnswer | None
    guidance: str | None  # pending regeneration_guidance; consumed (cleared) by the draft node
    critic_verdict: CriticVerdict | None  # latest verdict; the conditional edge reads this
    iteration: int  # critic passes completed so far
    iterations: list[IterationTrace]
    budget_exhausted: bool
    final: FinalAnswer | None


def _build_graph(
    settings: Settings,
    conn,
    embedder: Embedder | None,
    repo_root: Path,
    index: list[CodeChunk],
):
    """Compile the graph with dependencies captured in closures. Nodes call
    retrieve/draft_answer/etc. as module globals (not the captured names) so
    tests monkeypatch app.graph.<name> exactly like test_pipeline.py did."""

    def retrieve_node(state: GraphState) -> dict:
        # One retrieve() per query; dedupe by chunk id keeping the max fused
        # score; cap at DEFAULT_TOP_K so drafter context stays bounded across
        # loops. (Scores from different query vectors are only roughly
        # comparable — acceptable at Phase 3, revisited with real RRF in 5.)
        merged: dict[str, RetrievedChunk] = {}
        for query in state["queries"]:
            for rc in retrieve(
                state["repo_id"], query, settings=settings, conn=conn, embedder=embedder
            ):
                existing = merged.get(rc.chunk.id)
                if existing is None or rc.fused_score > existing.fused_score:
                    merged[rc.chunk.id] = rc
        top = sorted(merged.values(), key=lambda rc: rc.fused_score, reverse=True)
        return {"retrieved": top[:DEFAULT_TOP_K]}

    def draft_node(state: GraphState) -> dict:
        draft = draft_answer(
            state["question"], state["retrieved"], settings=settings, guidance=state["guidance"]
        )
        # Clear guidance after one use: a later re_retrieve loop must not draft
        # with stale guidance from an earlier regenerate.
        return {"draft": draft, "guidance": None}

    def critic_node(state: GraphState) -> dict:
        draft = state["draft"]
        checks = run_mechanical_checks(draft, repo_root, index)
        verdicts = run_semantic_checks(draft, checks, repo_root, settings=settings)
        verdict = decide_route(state["question"], verdicts, settings=settings)

        iteration = state["iteration"] + 1
        entry = IterationTrace(
            iteration=iteration, chunks_retrieved=len(state["retrieved"]), critic=verdict
        )
        budget_exhausted = (
            iteration >= MAX_CRITIC_ITERATIONS and verdict.route is not Route.PROCEED
        )
        updates: dict = {
            "critic_verdict": verdict,
            "iteration": iteration,
            "iterations": state["iterations"] + [entry],
            "budget_exhausted": budget_exhausted,
        }
        if verdict.route is Route.RE_RETRIEVE:
            updates["queries"] = verdict.refined_queries
        elif verdict.route is Route.REGENERATE:
            updates["guidance"] = verdict.regeneration_guidance
        return updates

    def synthesize_node(state: GraphState) -> dict:
        final = synthesize_answer(
            state["question"],
            state["draft"],
            state["critic_verdict"].verdicts,
            budget_exhausted=state["budget_exhausted"],
            settings=settings,
        )
        return {"final": final}

    builder = StateGraph(GraphState)
    builder.add_node(_RETRIEVE, retrieve_node)
    builder.add_node(_DRAFT, draft_node)
    builder.add_node(_CRITIC, critic_node)
    builder.add_node(_SYNTHESIZE, synthesize_node)

    builder.add_edge(START, _RETRIEVE)
    builder.add_edge(_RETRIEVE, _DRAFT)
    builder.add_edge(_DRAFT, _CRITIC)
    builder.add_conditional_edges(
        _CRITIC,
        _route_after_critic,
        {_RETRIEVE: _RETRIEVE, _DRAFT: _DRAFT, _SYNTHESIZE: _SYNTHESIZE},
    )
    builder.add_edge(_SYNTHESIZE, END)
    return builder.compile()


def _route_after_critic(state: GraphState) -> str:
    """The conditional edge — the agentic decision point. Returns a node name,
    mapped explicitly in add_conditional_edges (Route values != node names)."""
    verdict = state["critic_verdict"]
    if verdict.route is Route.PROCEED:
        return _SYNTHESIZE
    if state["budget_exhausted"]:  # set by critic_node when the final pass still wants to loop
        return _SYNTHESIZE
    if verdict.route is Route.RE_RETRIEVE:
        return _RETRIEVE
    return _DRAFT  # Route.REGENERATE


def _placeholder_plan(question: str) -> Plan:
    """No planner until Phase 4 — the question is the single retrieval query.
    Fills Trace.plan (non-optional in the frozen schema)."""
    return Plan(
        decomposed=False,
        sub_queries=[
            SubQuery(
                id=1,
                query=question,
                rationale="No planner until Phase 4; the question is used verbatim as the single query.",
            )
        ],
        reasoning="Planner not yet built (Phase 4). The raw question is the sole retrieval query.",
    )


def ask_with_trace(
    repo_id: str,
    question: str,
    settings: Settings | None = None,
    conn=None,
    embedder: Embedder | None = None,
) -> AskResponse:
    """Run the full graph and return the final answer plus its execution trace.

    repo_root + the chunk index are loaded once here (not per iteration —
    nothing writes chunks during an ask). Raises RepoNotIndexedError if the
    repo has no persisted root (indexed before Phase 3)."""
    settings = settings or get_settings()
    owns_conn = conn is None
    conn = conn or get_connection(settings)
    try:
        repo_root = load_repo_root(repo_id, conn)
        index = load_chunks(repo_id, conn)
        graph = _build_graph(settings, conn, embedder, repo_root, index)
        result = graph.invoke(
            {
                "question": question,
                "repo_id": repo_id,
                "queries": [question],
                "retrieved": [],
                "draft": None,
                "guidance": None,
                "critic_verdict": None,
                "iteration": 0,
                "iterations": [],
                "budget_exhausted": False,
                "final": None,
            }
        )
    finally:
        if owns_conn:
            conn.close()

    trace = Trace(
        plan=_placeholder_plan(question),
        iterations=result["iterations"],
        budget_exhausted=result["budget_exhausted"],
        models_used={
            node: settings.node_config(node).model
            for node in ("drafter", "critic", "synthesizer")
        },
    )
    return AskResponse(answer=result["final"], trace=trace)
