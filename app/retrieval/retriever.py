"""app/retrieval/retriever.py — dense-only retrieval.

pgvector cosine top-k, no lexical side and no real fusion yet: fused_score
is just dense_score. Phase 5 adds FTS and Reciprocal Rank Fusion. The graph
calls this once per planner sub-query, passing that sub-query's id so results
carry their provenance (RetrievedChunk.sub_query_id); it defaults to 1 for the
single-query / direct-call case.
"""

from __future__ import annotations

from pgvector import Vector

from app.config import Settings, get_settings
from app.db import get_connection
from app.embeddings import Embedder, get_embedder
from app.schemas import ChunkKind, CodeChunk, Language, RetrievedChunk

DEFAULT_TOP_K = 8


def retrieve(
    repo_id: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    settings: Settings | None = None,
    conn=None,
    embedder: Embedder | None = None,
    sub_query_id: int = 1,
) -> list[RetrievedChunk]:
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    query_vector = Vector(embedder.embed_query(query))

    owns_conn = conn is None
    conn = conn or get_connection(settings)
    try:
        rows = conn.execute(
            """
            SELECT id, repo_id, file_path, start_line, end_line, language, kind,
                   symbol, content, content_hash,
                   1 - (embedding <=> %(vector)s) AS score
            FROM chunks
            WHERE repo_id = %(repo_id)s
            ORDER BY embedding <=> %(vector)s
            LIMIT %(top_k)s
            """,
            {"vector": query_vector, "repo_id": repo_id, "top_k": top_k},
        ).fetchall()
    finally:
        if owns_conn:
            conn.close()

    retrieved: list[RetrievedChunk] = []
    for (
        chunk_id,
        chunk_repo_id,
        file_path,
        start_line,
        end_line,
        language,
        kind,
        symbol,
        content,
        content_hash,
        score,
    ) in rows:
        chunk = CodeChunk(
            id=chunk_id,
            repo_id=chunk_repo_id,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            language=Language(language),
            kind=ChunkKind(kind),
            symbol=symbol,
            content=content,
            content_hash=content_hash,
        )
        retrieved.append(
            RetrievedChunk(
                chunk=chunk,
                dense_score=float(score),
                lexical_score=None,
                fused_score=float(score),
                sub_query_id=sub_query_id,
            )
        )
    return retrieved
