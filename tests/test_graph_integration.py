"""app/graph.py integration test — sabotaged retriever forces a real loop.

Unlike test_graph.py (which stubs the whole critic), this drives the REAL
mechanical/semantic/route/drafter/synthesizer bodies against a temp repo on
disk, stubbing only the LLM boundary in each module (get_chat_model) — no
network, no DB. A poisoned retriever hands the drafter wrong-file evidence on
the first pass; the real mechanical layer catches the resulting hallucinated
citation (FABRICATED), the router re_retrieves, and the second pass verifies.
The assertion that matters: the critic visibly triggered at least one loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.critic.route as route_module
import app.critic.semantic as semantic_module
import app.graph as graph_module
import app.nodes.drafter as drafter_module
import app.nodes.synthesizer as synth_module
from app.config import Settings
from app.critic.route import _RouteDecision
from app.critic.semantic import _SemanticBatch
from app.indexing.chunker import chunk_text
from app.nodes.synthesizer import _SynthesizedAnswer
from app.schemas import (
    Citation,
    CitationStatus,
    CitationVerdict,
    DraftAnswer,
    Route,
    RetrievedChunk,
)

REPO_ID = "sabotage-repo"
QUESTION = "how is the auth token validated?"
REFINED_QUERY = "token validation in auth.py"

AUTH_PY = """\
import time


class TokenError(Exception):
    pass


def _decode(token):
    if not token or "." not in token:
        raise TokenError("malformed token")
    header, payload, signature = token.split(".")
    return header, payload, signature


def validate_token(token, now=None):
    now = now or int(time.time())
    header, payload, signature = _decode(token)
    if not signature:
        raise TokenError("missing signature")
    expiry = _read_expiry(payload)
    if expiry is not None and expiry < now:
        raise TokenError("token expired")
    return True


def _read_expiry(payload):
    if payload.isdigit():
        return int(payload)
    return None
"""

UNRELATED_PY = """\
import logging


def configure_logging(level="INFO"):
    logging.basicConfig(level=level)
    return logging.getLogger("app")


def log_request(logger, method, path):
    logger.info("%s %s", method, path)
"""


# ------------------------------------------------------- LLM-boundary fakes


class _FakeStructuredModel:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.invocations = 0

    def invoke(self, messages: list):
        self.invocations += 1
        return self._responses.pop(0)


class _FakeChatModel:
    def __init__(self, structured_model: _FakeStructuredModel) -> None:
        self._structured_model = structured_model

    def with_structured_output(self, schema: object) -> _FakeStructuredModel:
        return self._structured_model


def _patch_llm(monkeypatch, module, responses: list) -> _FakeStructuredModel:
    fake = _FakeStructuredModel(responses)
    monkeypatch.setattr(module, "get_chat_model", lambda node, settings: _FakeChatModel(fake))
    return fake


def test_sabotaged_retriever_triggers_at_least_one_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # --- real repo on disk + real index (each file < 60 lines => one chunk) ---
    (tmp_path / "auth.py").write_text(AUTH_PY, encoding="utf-8")
    (tmp_path / "unrelated.py").write_text(UNRELATED_PY, encoding="utf-8")
    index = chunk_text(REPO_ID, "auth.py", AUTH_PY) + chunk_text(REPO_ID, "unrelated.py", UNRELATED_PY)
    auth_chunk = next(c for c in index if c.file_path == "auth.py")
    unrelated_chunk = next(c for c in index if c.file_path == "unrelated.py")

    monkeypatch.setattr(graph_module, "load_repo_root", lambda repo_id, conn: tmp_path)
    monkeypatch.setattr(graph_module, "load_chunks", lambda repo_id, conn: index)

    # --- sabotaged retriever: wrong-file chunk first, right file afterwards ---
    retrieve_queries: list[str] = []

    def sabotaged_retrieve(repo_id, query, settings=None, conn=None, embedder=None):
        retrieve_queries.append(query)
        chunk = unrelated_chunk if len(retrieve_queries) == 1 else auth_chunk
        return [RetrievedChunk(chunk=chunk, dense_score=0.9, fused_score=0.9, sub_query_id=1)]

    monkeypatch.setattr(graph_module, "retrieve", sabotaged_retrieve)

    # --- drafter: pass 1 hallucinates a nonexistent file; pass 2 cites auth.py ---
    draft_bad = DraftAnswer(
        answer_markdown="The token is validated by the auth middleware [1].",
        citations=[
            Citation(
                id=1,
                file_path="auth/middleware.py",  # does not exist in the repo
                start_line=5,
                end_line=12,
                claim="Validates the request's auth token.",
            )
        ],
    )
    draft_good = DraftAnswer(
        answer_markdown="`validate_token` decodes the token and checks its expiry [1].",
        citations=[
            Citation(
                id=1,
                file_path="auth.py",
                start_line=1,
                end_line=25,  # within the file's 29 lines and its single chunk
                claim="validate_token decodes the token and rejects expired ones.",
            )
        ],
    )
    _patch_llm(monkeypatch, drafter_module, [draft_bad, draft_good])

    # --- semantic layer runs only on pass 2 (pass 1 fails mechanically, no LLM) ---
    semantic_fake = _patch_llm(
        monkeypatch,
        semantic_module,
        [
            _SemanticBatch(
                verdicts=[
                    CitationVerdict(
                        citation_id=1,
                        status=CitationStatus.VERIFIED,
                        checked_semantically=True,
                        reasoning="validate_token decodes and checks expiry.",
                    )
                ]
            )
        ],
    )

    # --- router: re_retrieve after the fabrication, then proceed ---
    _patch_llm(
        monkeypatch,
        route_module,
        [
            _RouteDecision(
                route=Route.RE_RETRIEVE,
                refined_queries=[REFINED_QUERY],
                reasoning="citation 1 is fabricated; need real evidence for token validation.",
            ),
            _RouteDecision(route=Route.PROCEED, reasoning="citation verified against auth.py."),
        ],
    )

    # --- synthesizer: keep the verified auth.py citation ---
    _patch_llm(
        monkeypatch,
        synth_module,
        [
            _SynthesizedAnswer(
                answer_markdown="`validate_token` decodes the token and rejects expired ones [1].",
                citations=[
                    Citation(
                        id=1,
                        file_path="auth.py",
                        start_line=1,
                        end_line=25,
                        claim="validate_token decodes the token and rejects expired ones.",
                    )
                ],
            )
        ],
    )

    settings = Settings(_env_file=None, openai_api_key="k", groq_api_key="k")
    response = graph_module.ask_with_trace(REPO_ID, QUESTION, settings=settings, conn=object())

    trace = response.trace
    # The loop visibly fired: two critic iterations.
    assert len(trace.iterations) == 2

    # Iteration 1's FABRICATED verdict came from the REAL mechanical layer
    # (checked_semantically False, reasoning about a missing file).
    iter1 = trace.iterations[0]
    assert iter1.critic.route is Route.RE_RETRIEVE
    v1 = iter1.critic.verdicts[0]
    assert v1.status is CitationStatus.FABRICATED
    assert v1.checked_semantically is False
    assert "does not exist" in v1.reasoning.lower()

    # The refined query reached the (sabotaged) retriever on the second pass.
    assert retrieve_queries == [QUESTION, REFINED_QUERY]

    # Iteration 2 verified via the semantic layer and proceeded.
    assert trace.iterations[1].critic.route is Route.PROCEED
    assert semantic_fake.invocations == 1

    # Final answer is grounded in the real file, budget was not exhausted.
    assert response.answer.citations[0].file_path == "auth.py"
    assert trace.budget_exhausted is False
