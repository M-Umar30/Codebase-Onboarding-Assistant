# Codebase Onboarding Assistant

A multi-agent RAG system that answers questions about an unfamiliar codebase
("how does auth work here?") with verified file/line citations. Its critic
agent independently checks every citation against the actual code and
decides — via LLM judgment, not a retry counter — whether to re-retrieve,
regenerate, or ship the answer.

## Phase checklist

- [x] Phase 0 — Skeleton: repo layout, frozen schemas, Postgres+pgvector,
      config, LLM/embedder ports
- [x] Phase 1 — Dumb-but-working pipeline
- [ ] Phase 2 — Critic standalone + hallucination fixture set
- [ ] Phase 3 — Wire the critic loop into LangGraph
- [ ] Phase 4 — Planner + syntax-aware chunking
- [ ] Phase 5 — Hybrid retrieval + ablation
- [ ] Phase 6 — Ship (FastAPI, CLI polish, incremental re-index, Docker,
      eval table)
