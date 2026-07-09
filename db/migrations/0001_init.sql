-- 0001_init.sql — Phase 1: chunks table + pgvector dense index.
--
-- Embedding dimension is pinned to 1536 (OpenAI text-embedding-3-small, the
-- default embedder). Switching EMBEDDER_PROVIDER to "local"
-- (sentence-transformers/all-MiniLM-L6-v2, 384-dim) requires a new migration
-- to alter the column dimension and re-embed — that's out of scope for
-- Phase 1's naive pipeline.
--
-- Lexical (FTS/BM25) columns are deliberately absent: Phase 1 is dense-only.
-- Phase 5 adds the lexical side in its own migration.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    repo_id         TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    start_line      INTEGER NOT NULL,
    end_line        INTEGER NOT NULL,
    language        TEXT NOT NULL,
    kind            TEXT NOT NULL,
    symbol          TEXT,
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    embedding       VECTOR(1536) NOT NULL,
    CONSTRAINT chunks_line_order CHECK (end_line >= start_line)
);

CREATE INDEX IF NOT EXISTS chunks_repo_id_idx ON chunks (repo_id);

-- HNSW cosine index: fine at Phase-1 scale (single-repo demos), revisit
-- ivfflat/list tuning if corpora grow large enough to matter.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
