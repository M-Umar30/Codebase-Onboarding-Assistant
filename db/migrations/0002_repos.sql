-- 0002_repos.sql — Phase 3: record where each repo_id was indexed from.
--
-- The critic's mechanical layer reads source files off disk at ask time to
-- verify citations, but the frozen AskRequest carries only repo_id — no
-- root path. This table is the bridge: index_repo writes the resolved
-- on-disk root here, and ask-time (app/retrieval/store.py::load_repo_root)
-- reads it back. Repos indexed before Phase 3 have no row and must be
-- re-indexed once.

CREATE TABLE IF NOT EXISTS repos (
    repo_id     TEXT PRIMARY KEY,
    root_path   TEXT NOT NULL,
    indexed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
