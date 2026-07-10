"""app/graph.py tests — the LangGraph loop with a scripted critic.

Every boundary the graph touches is monkeypatched at app.graph.* (module
globals, so the closures resolve the fakes at call time): load_repo_root /
load_chunks (no DB), retrieve / draft_answer / synthesize_answer (recording
fakes), and the three critic functions. decide_route pops a scripted list of
CriticVerdicts, so each test forces an exact route sequence and asserts the
path taken and the resulting Trace. A sentinel conn keeps get_connection from
ever running.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.graph as graph_module
from app.config import Settings
from app.graph import MAX_CRITIC_ITERATIONS, ask_with_trace
from app.retrieval.store import RepoNotIndexedError
from app.schemas import (
    ChunkKind,
    Citation,
    CitationStatus,
    CitationVerdict,
    CodeChunk,
    CriticVerdict,
    DraftAnswer,
    FinalAnswer,
    Language,
    Plan,
    Route,
    RetrievedChunk,
    SubQuery,
)

REPO_ID = "mock-repo"
QUESTION = "how does auth work here?"


def _plan(*queries: str) -> Plan:
    """A Plan whose sub-queries are `queries` (defaults to the raw question)."""
    subs = queries or (QUESTION,)
    return Plan(
        decomposed=len(subs) > 1,
        sub_queries=[SubQuery(id=i, query=q, rationale="test") for i, q in enumerate(subs, start=1)],
        reasoning="scripted plan",
    )


# --------------------------------------------------------------- builders


def _chunk(chunk_id: str, file_path: str = "auth.py") -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        repo_id=REPO_ID,
        file_path=file_path,
        start_line=1,
        end_line=10,
        language=Language.OTHER,
        kind=ChunkKind.BLOCK,
        symbol=None,
        content="def authenticate(): ...",
        content_hash="h",
    )


def _retrieved(chunk_id: str, score: float, file_path: str = "auth.py") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_chunk(chunk_id, file_path), dense_score=score, fused_score=score, sub_query_id=1
    )


def _draft() -> DraftAnswer:
    return DraftAnswer(
        answer_markdown="Auth validates the token [1].",
        citations=[Citation(id=1, file_path="auth.py", start_line=1, end_line=10, claim="validates token")],
    )


def _final() -> FinalAnswer:
    return FinalAnswer(
        answer_markdown="Auth validates the token [1].",
        citations=[Citation(id=1, file_path="auth.py", start_line=1, end_line=10, claim="validates token")],
    )


def _verdict(route: Route, **overrides: object) -> CriticVerdict:
    cv = CitationVerdict(
        citation_id=1, status=CitationStatus.VERIFIED, checked_semantically=True, reasoning="ok"
    )
    payload: dict = dict(verdicts=[cv], route=route, reasoning="scripted")
    payload.update(overrides)
    return CriticVerdict(**payload)


# ----- import Language after use in _chunk (kept simple; real import below) -----
from app.schemas import Language  # noqa: E402


class _ScriptedRoute:
    """decide_route stand-in: pops a pre-scripted CriticVerdict each call and
    records the questions/verdicts it was asked to route."""

    def __init__(self, verdicts: list[CriticVerdict]) -> None:
        self._verdicts = list(verdicts)
        self.calls = 0

    def __call__(self, question, verdicts, settings=None) -> CriticVerdict:
        self.calls += 1
        return self._verdicts.pop(0)


class _Harness:
    """Installs all graph-boundary fakes and records calls."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        route_script: list[CriticVerdict],
        plan: Plan | None = None,
    ) -> None:
        self.retrieve_calls: list[tuple[str, str]] = []
        self.retrieve_sub_query_ids: list[int] = []
        self.draft_calls: list[tuple[str, list, str | None]] = []
        self.synth_calls: list[tuple[str, list, bool]] = []
        self.route = _ScriptedRoute(route_script)

        # Planner: return a scripted Plan (default mirrors the raw question) so
        # the graph entry is deterministic and no LLM runs.
        plan = plan or _plan()
        monkeypatch.setattr(graph_module, "plan_question", lambda question, settings=None: plan)

        # No DB.
        monkeypatch.setattr(graph_module, "load_repo_root", lambda repo_id, conn: Path("/fake/root"))
        monkeypatch.setattr(graph_module, "load_chunks", lambda repo_id, conn: [])

        # Retrieval: one chunk per query, id/score keyed off the query so
        # dedupe/sort are observable. sub_query_id is threaded through by the
        # graph's fan-out and recorded here.
        def fake_retrieve(repo_id, query, settings=None, conn=None, embedder=None, sub_query_id=1):
            self.retrieve_calls.append((repo_id, query))
            self.retrieve_sub_query_ids.append(sub_query_id)
            return self._chunks_for_query(query, sub_query_id)

        monkeypatch.setattr(graph_module, "retrieve", fake_retrieve)

        def fake_draft(question, retrieved, settings=None, guidance=None):
            self.draft_calls.append((question, list(retrieved), guidance))
            return _draft()

        monkeypatch.setattr(graph_module, "draft_answer", fake_draft)

        # The critic's mechanical/semantic layers are stubbed to no-ops here;
        # only decide_route's scripted verdict matters for routing tests.
        monkeypatch.setattr(graph_module, "run_mechanical_checks", lambda draft, root, index: [])
        monkeypatch.setattr(
            graph_module, "run_semantic_checks", lambda draft, checks, root, settings=None: []
        )
        monkeypatch.setattr(graph_module, "decide_route", self.route)

        def fake_synth(question, draft, verdicts, budget_exhausted=False, settings=None):
            self.synth_calls.append((question, list(verdicts), budget_exhausted))
            return _final()

        monkeypatch.setattr(graph_module, "synthesize_answer", fake_synth)

    def _chunks_for_query(self, query: str, sub_query_id: int = 1) -> list[RetrievedChunk]:
        # Overlapping ids across queries so the dedupe path is exercised in the
        # re_retrieve test: "q1" -> {a,b}, "q2" -> {b,c}.
        table = {
            QUESTION: [_retrieved("a", 0.9)],
            "q1": [_retrieved("a", 0.5), _retrieved("b", 0.8)],
            "q2": [_retrieved("b", 0.6), _retrieved("c", 0.7)],
            "fix X": [_retrieved("a", 0.9)],
        }
        chunks = table.get(query, [_retrieved("a", 0.9)])
        return [rc.model_copy(update={"sub_query_id": sub_query_id}) for rc in chunks]


def _run(monkeypatch, route_script):
    harness = _Harness(monkeypatch, route_script)
    settings = Settings(_env_file=None, openai_api_key="k", groq_api_key="k")
    response = ask_with_trace(REPO_ID, QUESTION, settings=settings, conn=object(), embedder=None)
    return harness, response


# ------------------------------------------------------------------ tests


class TestRouting:
    def test_proceed_ships_on_first_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        harness, response = _run(monkeypatch, [_verdict(Route.PROCEED)])

        assert harness.retrieve_calls == [(REPO_ID, QUESTION)]
        assert len(harness.draft_calls) == 1
        assert len(harness.synth_calls) == 1
        assert harness.synth_calls[0][2] is False  # budget_exhausted
        assert response.answer.answer_markdown == _final().answer_markdown

        trace = response.trace
        assert len(trace.iterations) == 1
        assert trace.iterations[0].critic.route is Route.PROCEED
        assert trace.iterations[0].chunks_retrieved == 1
        assert trace.budget_exhausted is False
        assert trace.plan.sub_queries[0].query == QUESTION
        assert trace.models_used == {
            "planner": "llama-3.3-70b-versatile",
            "drafter": "llama-3.3-70b-versatile",
            "critic": "gpt-4o-mini",
            "synthesizer": "gpt-4o-mini",
        }

    def test_re_retrieve_uses_refined_queries_and_dedupes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        script = [
            _verdict(Route.RE_RETRIEVE, refined_queries=["q1", "q2"]),
            _verdict(Route.PROCEED),
        ]
        harness, response = _run(monkeypatch, script)

        # initial question, then q1 + q2 on the loop
        assert harness.retrieve_calls == [(REPO_ID, QUESTION), (REPO_ID, "q1"), (REPO_ID, "q2")]
        # second draft got the merged/deduped chunks: {a,b,c}, sorted by score desc
        # retrieve_node rebuilds from state["sub_queries"] each call (the critic's
        # refined queries), so the loop sees only q1+q2 (not the original
        # question). Merge keeps max score per id: a=0.5, b=max(0.8,0.6)=0.8,
        # c=0.7 -> sorted desc.
        second_draft_chunks = harness.draft_calls[1][1]
        ids = [rc.chunk.id for rc in second_draft_chunks]
        assert ids == ["b", "c", "a"]
        assert len(ids) <= 8
        routes = [it.critic.route for it in response.trace.iterations]
        assert routes == [Route.RE_RETRIEVE, Route.PROCEED]

    def test_regenerate_injects_guidance_without_reretrieving(self, monkeypatch: pytest.MonkeyPatch) -> None:
        script = [
            _verdict(Route.REGENERATE, regeneration_guidance="fix X"),
            _verdict(Route.PROCEED),
        ]
        harness, response = _run(monkeypatch, script)

        assert harness.retrieve_calls == [(REPO_ID, QUESTION)]  # no re-retrieval
        # draft called twice: first with no guidance, second with the critic's guidance
        assert harness.draft_calls[0][2] is None
        assert harness.draft_calls[1][2] == "fix X"
        routes = [it.critic.route for it in response.trace.iterations]
        assert routes == [Route.REGENERATE, Route.PROCEED]

    def test_guidance_is_not_sticky_across_a_later_reretrieve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        script = [
            _verdict(Route.REGENERATE, regeneration_guidance="fix X"),
            _verdict(Route.RE_RETRIEVE, refined_queries=["q1"]),
            _verdict(Route.PROCEED),
        ]
        harness, _response = _run(monkeypatch, script)

        # draft 1: entry (None), draft 2: regenerate ("fix X"), draft 3: after
        # re_retrieve — guidance must have been cleared (None), not "fix X".
        guidances = [call[2] for call in harness.draft_calls]
        assert guidances == [None, "fix X", None]

    def test_budget_exhaustion_overrules_final_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        script = [
            _verdict(Route.RE_RETRIEVE, refined_queries=["q1"]),
            _verdict(Route.RE_RETRIEVE, refined_queries=["q2"]),
            _verdict(Route.RE_RETRIEVE, refined_queries=["q1"]),  # 3rd verdict: overruled
        ]
        harness, response = _run(monkeypatch, script)

        assert harness.route.calls == MAX_CRITIC_ITERATIONS  # exactly 3 critic passes
        # retrieve: initial + 2 loops = 3 calls (the 3rd verdict does NOT re-retrieve)
        assert len(harness.retrieve_calls) == 3
        assert harness.synth_calls[0][2] is True  # budget_exhausted passed to synthesizer
        trace = response.trace
        assert trace.budget_exhausted is True
        assert len(trace.iterations) == MAX_CRITIC_ITERATIONS
        # all three verdicts recorded, including the overruled last one
        assert all(it.critic.route is Route.RE_RETRIEVE for it in trace.iterations)

    def test_missing_repo_row_raises_before_any_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        harness = _Harness(monkeypatch, [_verdict(Route.PROCEED)])

        def boom(repo_id, conn):
            raise RepoNotIndexedError("no row")

        monkeypatch.setattr(graph_module, "load_repo_root", boom)
        settings = Settings(_env_file=None, openai_api_key="k", groq_api_key="k")

        with pytest.raises(RepoNotIndexedError):
            ask_with_trace(REPO_ID, QUESTION, settings=settings, conn=object())

        assert harness.retrieve_calls == []  # nothing ran


class TestPlannerFanOut:
    def test_retriever_fans_out_over_sub_queries_with_provenance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A decomposed plan of two sub-queries: the retriever is called once per
        # sub-query, tagged with that sub-query's id.
        harness = _Harness(monkeypatch, [_verdict(Route.PROCEED)], plan=_plan("q1", "q2"))
        settings = Settings(_env_file=None, openai_api_key="k", groq_api_key="k")

        response = ask_with_trace(REPO_ID, QUESTION, settings=settings, conn=object(), embedder=None)

        # one retrieve() per sub-query, each carrying its sub_query_id
        assert harness.retrieve_calls == [(REPO_ID, "q1"), (REPO_ID, "q2")]
        assert harness.retrieve_sub_query_ids == [1, 2]

        # merged chunks keep provenance: q1 -> {a,b}, q2 -> {b,c}; b is deduped
        # to the higher-scoring q1 copy (0.8 > 0.6), so it stays sub_query_id=1.
        by_id = {rc.chunk.id: rc for rc in harness.draft_calls[0][1]}
        assert by_id["a"].sub_query_id == 1
        assert by_id["b"].sub_query_id == 1
        assert by_id["c"].sub_query_id == 2

        # the plan carried into the trace is the decomposed one
        assert response.trace.plan.decomposed is True
        assert [s.id for s in response.trace.plan.sub_queries] == [1, 2]
