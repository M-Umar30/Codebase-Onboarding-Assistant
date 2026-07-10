# Architecture & Decisions Log

Running record of implementation decisions, organized by phase. CLAUDE.md
holds the pre-build plan and locked decisions table; this doc captures what
actually got decided *during* each phase's build — especially anything
non-obvious that isn't self-evident from reading the code, and anything that
revised the original plan.

---

## Phase 0 — Skeleton

- **`app/schemas.py` is the frozen contract.** Every Pydantic model for
  every phase (including ones not yet wired up, like `Plan`/`SubQuery` for
  the not-yet-built planner) was agreed and locked before any node code.
  Conventions baked into the models themselves: line numbers are 1-indexed
  inclusive on both ends; citations reference inline `[n]` markers matching
  `Citation.id`; embeddings/FTS vectors live only in Postgres, never in these
  models.
- **`app/config.py`: per-node model assignment via `Settings.node_config(name)`.**
  `NodeName = Literal["planner", "drafter", "critic", "synthesizer"]`, each
  with its own `{node}_provider`/`{node}_model` env-overridable pair. Node
  code never branches on provider — it calls `get_settings().node_config(name)`
  indirectly through `app.llm.get_chat_model`.
- **`app/llm.py`: `get_chat_model(node_name)` provider factory**, reused from
  project #1's pattern. Branches on `provider` (`"openai"` → `ChatOpenAI`,
  `"groq"` → `ChatGroq`), raises on unknown provider. Structured output is
  bound at the call site via `.with_structured_output(SomeModel)`, not inside
  this factory.
- **`app/embeddings.py`: `Embedder` protocol** with `OpenAIEmbedder`
  (`text-embedding-3-small`, default) and `LocalEmbedder`
  (sentence-transformers fallback, no API credits needed for demos).
- **`app/db.py`: Postgres + pgvector connection/migration runner.**
  Autocommit connections (no multi-statement transactions to coordinate
  yet); a `schema_migrations` table tracks applied `.sql` files.
- **Testing convention (holds for every phase since):** tests never read the
  real `.env` — `tests/conftest.py`'s `test_settings` fixture builds
  `Settings` in-process with fake keys, so no real API key is required and
  no network call can happen by accident in `pytest`. Mocking is hand-rolled
  stub classes or `monkeypatch`, never a mocking library.

## Phase 1 — Dumb-but-working pipeline

- **`app/indexing/chunker.py`: naive fixed-size line-window chunking.**
  `WINDOW_SIZE = 60`, `OVERLAP_SIZE = 10`. Every chunk is
  `Language.OTHER` / `ChunkKind.BLOCK` / `symbol=None` regardless of file
  type — tree-sitter chunking is explicitly deferred to Phase 4. This is a
  documented scoping decision, not an oversight, and it has a real
  downstream consequence: **no chunk in Phase 1/2 ever has a real symbol
  value**, so any symbol-verification logic (see Phase 2 below) is
  necessarily a textual heuristic, not a field comparison.
- **`app/indexing/walker.py`: `.gitignore`-aware file discovery** (git
  semantics — a directory's `.gitignore` only governs paths beneath it) plus
  a hardcoded skip list for vendored/generated dirs that repos often don't
  gitignore (`node_modules`, `.git`, `dist`, `build`, `__pycache__`, `venv`,
  `.venv`, `env`), plus a NUL-byte binary sniff on the first 8KB.
- **`app/indexing/indexer.py`: full re-embed on every `index_repo` call.**
  No incremental re-index yet — `files_skipped_unchanged` is hardcoded to 0
  in `IndexResponse` (the field exists now so Phase 6 can start reporting
  real numbers into it without a schema change).
- **`app/retrieval/retriever.py`: dense-only pgvector cosine top-k.**
  `fused_score` is just `dense_score` and every result is attributed to
  `sub_query_id=1`, since there's no planner yet to decompose the question.
  Lexical/FTS and real RRF fusion arrive in Phase 5.
- **`app/nodes/drafter.py` / `app/nodes/synthesizer.py`: single structured
  LLM calls, no loop.** `get_chat_model(node).with_structured_output(Model)`
  is the pattern every subsequent structured call (critic included) copies.
  Phase 1's synthesizer trusts every draft citation as-is — no verification
  exists yet.
- **`app/pipeline.py`: plain function composition, no LangGraph.**
  `ask() = retrieve → draft → synthesize`. Deliberately not a graph yet —
  there's no actual conditional edge to justify one until the critic exists
  (Phase 3).
- **`cli/main.py`: Typer app**, `index` and `ask` subcommands, thin wrappers
  over `index_repo`/`pipeline.ask`. No logging module anywhere in the
  codebase; exception handling is minimal and only at genuine boundaries
  (e.g. `UnicodeDecodeError`/`OSError` on file read) — this convention holds
  through Phase 2.

## Phase 2 — Standalone critic

Built per CLAUDE.md's two-layer design (mechanical, zero-LLM; semantic,
LLM) plus the routing verdict — but **not wired into `app.pipeline`**;
measured standalone against a hand-built fixture set. Several of the
decisions below only exist because a review pass caught two real bugs in
the first draft of this design; both are called out.

- **Eval repos pinned**: `fastapi/fastapi` @ tag `0.139.0`
  (commit `cecd96d9c6c318e0df1c40cedbc2e953381ddfd3`) and `honojs/hono` @
  tag `v4.12.28` (commit `626b185d0e80fa1c2a3cc1a8945afd0df6aaf3ec`). Chosen
  over alternatives (e.g. `psf/requests` + `colinhacks/zod`) specifically
  because both have real, readable auth code (`fastapi/security/*`,
  `hono/middleware/{jwt,bearer-auth,basic-auth}`) matching the project's
  flagship demo question, "how does auth work here?"
- **Fixtures vendor trimmed excerpts, not full clones**
  (`evals/fixtures/repos/`). Real code from the pinned commits, trimmed to
  strip verbose docstrings/`Doc()` annotations/JSDoc example blocks —
  function/class logic kept verbatim. `evals/fixtures/repos/MANIFEST.md`
  records exact source path + commit per vendored file. This keeps fixtures
  reproducible offline (no live clone needed to run `pytest` or the eval
  script) and keeps chunk-alignment simple to hand-construct. Phase 5's
  retrieval-ablation eval needs a full checkout of these same pins if/when
  it wants one — that's a separate, later concern.
- **The critic's "repo index" is built via the existing `chunk_text()`, not
  Postgres.** `run_mechanical_checks(draft, repo_root, index: list[CodeChunk])`
  takes the index as a plain parameter rather than querying a DB itself.
  Production (Phase 3+) will build that list from Postgres; the eval script
  and tests build it by walking vendored/temp files through `chunk_text()`
  directly. This keeps the mechanical layer — and the fast pytest subset —
  free of any live-Postgres dependency, unlike `test_retriever.py`'s
  existing DB-dependent tests. No DB-loader helper was added in this phase
  since nothing in Phase 2 calls it.
- **Mechanical hash check is containment-based, not exact-range equality**
  (`app/critic/mechanical.py::_find_containing_chunk`). *Bug caught in
  review*: the first draft required a citation's own `(start_line,
  end_line)` to exactly equal an indexed chunk's boundary, which would have
  branded nearly every honest sub-range citation (e.g. citing lines 10–25 of
  a retrieved 60-line window to support one specific claim — exactly how a
  real drafter cites evidence) as `FABRICATED`. Fixed: look up the
  *containing* chunk (tightest match wins), and check that chunk's own
  current-vs-indexed hash — decoupling "was this location ever retrieved"
  (containment) from "has the evidence gone stale since indexing" (the hash
  comparison). `tests/test_critic_mechanical.py::TestContainment` is the
  regression suite for this specifically.
- **Known, accepted limitation**: `hash_matches_index=False` still collapses
  two distinct causes — "no containing chunk at all" (nothing was ever
  retrieved here — the dominant hallucination shape) vs. "a containing
  chunk exists but is stale" (file changed since indexing). `MechanicalCheck`
  is frozen with no field to carry that distinction downstream, so both
  collapse to the same boolean and both map to `FABRICATED`. Phase 2's
  fixtures model the dominant "never retrieved" case only; a future schema
  revision would be needed to separate staleness as its own signal.
- **Deterministic mechanical-failure → status mapping** lives in
  `app/critic/semantic.py::_map_mechanical_failure` (not mechanical.py),
  per the task's own module split: `file_exists=False` → `FABRICATED`;
  `lines_in_bounds=False` → `FABRICATED`; `hash_matches_index=False` →
  `FABRICATED`; `symbol_found=False` (only reachable once the first three
  pass) → `WRONG_LOCATION`. Everything else falls through to the semantic
  LLM layer.
- **Semantic layer batches one structured LLM call for all mechanically-
  passed citations**, with completeness enforcement. *Bug caught in
  review*: the first draft had no check that the LLM's batch response
  actually covered every citation it was asked about — a dropped citation
  wouldn't raise `ValidationError`, so the retry-once logic would never
  fire. Fixed: `_validate_batch()` checks the returned `citation_id` set
  exactly matches the requested set *and* every status is in
  `{VERIFIED, UNSUPPORTED_CLAIM}` *and* `checked_semantically` is true,
  raising a plain `ValueError` folded into the *same* retry-once path as
  malformed structured output (not a bare assert that crashes immediately —
  retry-once-then-fail-loudly applies to semantic completeness exactly like
  it applies to schema validity).
- **Route layer never lets the LLM re-emit citation verdicts.**
  `app/critic/route.py`'s structured call targets a small local
  `_RouteDecision` wrapper (route + refined_queries/regeneration_guidance +
  reasoning only) — no `verdicts` field. The already-known, already-computed
  `verdicts` list (from mechanical + semantic) is spliced in when
  constructing the real `CriticVerdict` in Python. This means the frozen
  `CriticVerdict.model_validator` (route/payload contract — already tested
  in `test_schemas.py`) is the *actual* enforcement mechanism exactly as
  CLAUDE.md intends ("schema validators must be the enforcement"), and there
  is no way for the LLM to silently drop a citation from `verdicts` the way
  a batched re-emission call could (this was designed *in response to*
  finding the same class of bug in the semantic layer — rather than adding
  a completeness guard to route.py too, the design sidesteps the risk
  entirely).
- **Retry-once-then-fail-loudly, consistently.** Both `semantic.py`'s batch
  call and `route.py`'s route call: catch the relevant error
  (`ValidationError`, plus `ValueError` for semantic's completeness check),
  append the error text as a follow-up `HumanMessage`, re-invoke once, let a
  second failure propagate unmodified. No silent patching of invalid LLM
  output anywhere in the critic.
- **Eval metrics report catch rate and status accuracy separately**
  (`evals/run_critic_eval.py`), not one conflated number. For the three
  "bad" categories: **catch rate** = fraction flagged as *anything other
  than* `VERIFIED` (the number that matters for "does a hallucination ever
  reach the user" — a fabricated citation mislabeled `wrong_location` still
  never ships) vs. **status accuracy** = fraction with the *exact* right
  label. For `verified`: **false-positive rate** = fraction incorrectly
  flagged as non-verified. Current fixture-set result: 100% catch rate,
  100% status accuracy, 0% false positives across all 32 fixtures — a
  clean score on a hand-built set, not proof of robustness against messier
  real-pipeline drafts (see "Open items" below).

## Phase 3 — Wire the loop

The critic (Phase 2, standalone) is now a LangGraph node whose
`CriticVerdict.route` drives a conditional edge. This is the project's core
agentic claim: model output, not an `if attempts < 3`, picks the next edge.

- **`app/graph.py`: the graph, with `GraphState` as a plain `TypedDict`.**
  Chosen over a Pydantic state model: there's no checkpointer (nothing to
  serialize) and the state holds frozen Pydantic models as *values*, which a
  TypedDict carries without re-validating on every merge. The graph is
  strictly sequential (no parallel branches), so default last-write-wins
  reducers are correct; `critic_node` appends to `iterations` explicitly
  (`state["iterations"] + [entry]`). If parallel retrieval is ever added,
  `iterations`/`retrieved` would need real merge reducers — noted at the
  `GraphState` definition.
- **Dependency injection via a closure factory, but nodes call collaborators
  as module globals.** `_build_graph(settings, conn, embedder, repo_root,
  index)` captures deps in closures, yet the node bodies call `retrieve` /
  `draft_answer` / `run_mechanical_checks` / … as `app.graph` globals. That
  keeps the existing test convention intact: graph tests monkeypatch
  `app.graph.<name>` exactly like Phase 1's `test_pipeline.py` did, and the
  integration test lets the real bodies run while patching only each module's
  `get_chat_model`.
- **Budget = 3 critic passes total, graph-enforced, and it OVERRULES the
  model.** `MAX_CRITIC_ITERATIONS = 3` (initial pass + up to 2 loop-backs).
  The critic always runs and its verdict is always recorded in the trace;
  when the final allowed pass still returns a non-PROCEED route, the graph
  overrules it — routes to synthesize with `budget_exhausted=True` — rather
  than deleting the verdict. So the trace shows *what the model wanted* even
  when capped. The budget lives only in the graph: never in the schema, never
  in the critic prompt.
- **Trace is assembled after `invoke()`, but its data is captured as the graph
  runs.** `critic_node` appends one `IterationTrace` (iteration number,
  `chunks_retrieved`, the full `CriticVerdict`) per pass; `ask_with_trace`
  wraps the accumulated list in the frozen `Trace`. `Trace.plan` is
  non-optional in the frozen schema but there's no planner until Phase 4, so
  a single-sub-query placeholder `Plan(decomposed=False, …)` fills it.
  `models_used` is sourced from `settings.node_config(node).model` for the
  three nodes that actually run (drafter/critic/synthesizer) — no "planner"
  key yet.
- **`guidance` is consumed exactly once.** `draft_node` returns
  `{"draft": …, "guidance": None}`, clearing the critic's
  `regeneration_guidance` after one use so a later `re_retrieve` loop doesn't
  redraft with stale guidance from an earlier `regenerate`. Regression:
  `test_graph.py::test_guidance_is_not_sticky_across_a_later_reretrieve`.
- **Multi-query re-retrieval merge.** On `re_retrieve`, `retrieve_node` runs
  one `retrieve()` per refined query, dedupes by `chunk.id` keeping the max
  `fused_score`, sorts desc, and caps at `DEFAULT_TOP_K` so drafter context
  stays bounded across loops. (Cosine scores from different query vectors are
  only roughly comparable — acceptable pre-RRF; revisited in Phase 5.)
- **Persistence gap closed: the `repos` table (migration 0002).** The
  mechanical layer needs the on-disk `repo_root` and the FULL chunk index at
  ask time, but the frozen `AskRequest` carries only `repo_id`. So
  `index_repo` now upserts `(repo_id, str(repo_root))` after the chunk write
  (a repos row implies a finished index), and `app/retrieval/store.py`'s
  `load_repo_root`/`load_chunks` read them back — loaded once per ask in
  `ask_with_trace`, not per iteration (nothing writes chunks during an ask).
  A missing repos row (indexed before Phase 3) or a vanished root path raises
  `RepoNotIndexedError` at the boundary with a re-index hint, rather than
  letting every mechanical check fail as a false `FABRICATED` verdict and
  burning three LLM loops on garbage.
- **Synthesizer hardening mirrors `route.py`'s splice pattern.** The LLM gets
  a narrow job (rewrite prose, renumber surviving citations into a local
  `_SynthesizedAnswer` wrapper without the notes/caveat fields), and every
  trust-sensitive field is Python-owned: verified/dropped partition from the
  verdicts, `unverified_notes` built deterministically from dropped citations,
  `confidence_caveat` set deterministically on budget exhaustion. A
  post-validation pass (`_validate_synthesis`) enforces that final citations
  are a subset of the verified set and that inline `[n]` markers line up
  1..n — folded into the same retry-once-then-fail-loudly path as the semantic
  and route layers. Zero verified citations short-circuits to an honest "I
  couldn't verify anything" answer with no LLM call (README policy: "I don't
  know" beats an unverified guess).
- **`--show-trace` is ASCII-only.** The trace is the interview demo, but the
  default Windows console is cp1252 and can't encode box-drawing/arrow glyphs;
  a demo that raises `UnicodeEncodeError` is worse than a plain one, so
  `render_trace` uses `-`/`->`/`|` rather than `─`/`→`/`·`.
- **`pipeline.ask()` survives as a façade.** It keeps the Phase-1 signature
  (`repo_id, question, settings -> FinalAnswer`) over `ask_with_trace(...).
  answer`, so nothing that only wants the answer had to change.
  `test_pipeline.py` was rewritten accordingly — its old monkeypatch targets
  (`pipeline.retrieve/draft_answer/synthesize_answer`) no longer exist once
  the composition moved into the graph; graph orchestration coverage lives in
  `test_graph.py`.

---

## Cross-cutting patterns (established Phase 0–1, held through Phase 3)

- `get_chat_model(node_name)` — every LLM call goes through this, never a
  direct SDK import at the node level.
- `settings = settings or get_settings()` + optional `conn=None`/
  `embedder=None` dependency injection on every public function that touches
  external state, so tests can substitute fakes without monkeypatching
  internals.
- No `logging` module usage anywhere. Exceptions are only caught at genuine
  boundaries (file I/O, LLM structured-output parsing) — never
  defensively "just in case."
- Structured LLM outputs are Pydantic models; validation failures are real
  failures, not silently coerced.

## Open items / known limitations to revisit

- `hash_matches_index` staleness-vs-never-retrieved collapse (Phase 2,
  above) — needs a schema change to separate if it starts mattering.
- Symbol verification is a literal substring match against cited text, not
  real symbol resolution — a consequence of Phase 1/2 chunks never
  populating `CodeChunk.symbol`. Expected to need rework once Phase 4's
  tree-sitter chunking gives citations real qualified symbol names.
- Eval fixtures vendor trimmed excerpts, not full repo checkouts (Phase 2).
  Phase 5's retrieval ablation needs its own full clone at the same pins.
- The critic's 100% eval scores are against a fixture set designed to be
  unambiguous (clearly-false claims, structurally clean containment cases).
  Phase 3's live pipeline wiring is the real test of whether the mechanical/
  semantic split holds up against messier, real drafter output.
