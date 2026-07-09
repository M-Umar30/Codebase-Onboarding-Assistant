"""app/retrieval/retriever.py test against a seeded Postgres+pgvector DB.

Requires the docker-compose Postgres to be running (see db/migrations for
the chunks table). No LLM call happens: the embedder is a stub returning
hand-built vectors, so only real pgvector cosine-similarity math is under
test, not an embedding model.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.config import Settings
from app.db import apply_migrations, get_connection
from app.indexing.chunker import chunk_text
from app.retrieval.retriever import retrieve
from app.schemas import CodeChunk

REPO_ID = "test-retriever-seeded"
DIM = 1536


def _basis_vector(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def _mixed_vector(index_a: int, index_b: int) -> list[float]:
    vector = [0.0] * DIM
    weight = 2**-0.5
    vector[index_a] = weight
    vector[index_b] = weight
    return vector


def _seed_chunk(conn: object, chunk: CodeChunk, vector: list[float]) -> None:
    conn.execute(
        """
        INSERT INTO chunks
            (id, repo_id, file_path, start_line, end_line,
             language, kind, symbol, content, content_hash, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            chunk.id,
            chunk.repo_id,
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
            chunk.language.value,
            chunk.kind.value,
            chunk.symbol,
            chunk.content,
            chunk.content_hash,
            vector,
        ),
    )


class _StubEmbedder:
    def __init__(self, query_vector: list[float]) -> None:
        self._query_vector = query_vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("seeding writes vectors directly in this test")

    def embed_query(self, text: str) -> list[float]:
        return self._query_vector


@pytest.fixture
def seeded_conn() -> Iterator[object]:
    settings = Settings(_env_file=None, openai_api_key="k", groq_api_key="k")
    conn = get_connection(settings)
    apply_migrations(conn)
    conn.execute("DELETE FROM chunks WHERE repo_id = %s", (REPO_ID,))

    close_chunk = chunk_text(REPO_ID, "close.py", "def close(): pass")
    orthogonal_chunk = chunk_text(REPO_ID, "orthogonal.py", "def far(): pass")
    mixed_chunk = chunk_text(REPO_ID, "mixed.py", "def mixed(): pass")

    all_chunks = close_chunk + orthogonal_chunk + mixed_chunk
    vectors = [
        _basis_vector(0),  # identical to the query vector -> score 1.0
        _basis_vector(1),  # orthogonal to the query vector -> score 0.0
        _mixed_vector(0, 1),  # partial overlap -> score ~0.7071
    ]
    for chunk, vector in zip(all_chunks, vectors):
        _seed_chunk(conn, chunk, vector)

    yield conn

    conn.execute("DELETE FROM chunks WHERE repo_id = %s", (REPO_ID,))
    conn.close()


def test_retrieve_orders_by_cosine_similarity_descending(seeded_conn: object) -> None:
    embedder = _StubEmbedder(query_vector=_basis_vector(0))

    results = retrieve(REPO_ID, "irrelevant text", top_k=3, conn=seeded_conn, embedder=embedder)

    assert [r.chunk.file_path for r in results] == ["close.py", "mixed.py", "orthogonal.py"]
    assert results[0].dense_score == pytest.approx(1.0, abs=1e-6)
    assert results[1].dense_score == pytest.approx(2**-0.5, abs=1e-6)
    assert results[2].dense_score == pytest.approx(0.0, abs=1e-6)


def test_retrieve_respects_top_k(seeded_conn: object) -> None:
    embedder = _StubEmbedder(query_vector=_basis_vector(0))

    results = retrieve(REPO_ID, "irrelevant text", top_k=1, conn=seeded_conn, embedder=embedder)

    assert len(results) == 1
    assert results[0].chunk.file_path == "close.py"


def test_retrieved_chunk_provenance_fields(seeded_conn: object) -> None:
    embedder = _StubEmbedder(query_vector=_basis_vector(0))

    results = retrieve(REPO_ID, "irrelevant text", top_k=1, conn=seeded_conn, embedder=embedder)
    result = results[0]

    assert result.lexical_score is None
    assert result.fused_score == result.dense_score
    assert result.sub_query_id == 1
    assert result.chunk.repo_id == REPO_ID


def test_retrieve_scopes_to_repo_id(seeded_conn: object) -> None:
    embedder = _StubEmbedder(query_vector=_basis_vector(0))

    results = retrieve("some-other-repo", "irrelevant text", top_k=5, conn=seeded_conn, embedder=embedder)

    assert results == []
