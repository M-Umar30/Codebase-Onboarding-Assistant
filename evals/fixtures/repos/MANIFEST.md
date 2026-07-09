# Vendored fixture repos

Trimmed excerpts of real code from two pinned public repos, used by
`evals/fixtures/*.json` and `evals/run_critic_eval.py`. Trimming removes
verbose docstrings / `Doc()` annotations / JSDoc example blocks to keep
excerpts short and chunk-alignment simple; function/class logic bodies are
kept verbatim from the source at the pinned commit. Not a full checkout —
Phase 5's retrieval-ablation eval does its own full clone of these same
pins if/when it needs one.

## fastapi/fastapi

Tag `0.139.0`, commit `cecd96d9c6c318e0df1c40cedbc2e953381ddfd3`.

| Vendored file | Source path | Trim notes |
|---|---|---|
| `fastapi/security/utils.py` | `fastapi/security/utils.py` | Verbatim, no trim (7 lines). |
| `fastapi/security/http_bearer.py` | `fastapi/security/http.py` | `HTTPBase` + `HTTPBearer` only; `HTTPBasic`/`HTTPDigest` dropped; `Doc()`/docstrings stripped. |
| `fastapi/security/http_basic.py` | `fastapi/security/http.py` | `HTTPBasic` only; `Doc()`/docstrings stripped. |
| `fastapi/security/oauth2_password_bearer.py` | `fastapi/security/oauth2.py` | `OAuth2` + `OAuth2PasswordBearer` only; `OAuth2PasswordRequestForm`/`OAuth2AuthorizationCodeBearer`/`SecurityScopes` dropped; `Doc()`/docstrings stripped. |
| `fastapi/exception_handlers.py` | `fastapi/exception_handlers.py` | Verbatim, no trim (non-auth, for category variety). |

## honojs/hono

Tag `v4.12.28`, commit `626b185d0e80fa1c2a3cc1a8945afd0df6aaf3ec`.

| Vendored file | Source path | Trim notes |
|---|---|---|
| `hono/middleware/basic_auth.ts` | `src/middleware/basic-auth/index.ts` | JSDoc example blocks stripped; logic verbatim. |
| `hono/middleware/bearer_auth.ts` | `src/middleware/bearer-auth/index.ts` | JSDoc + deprecated `*Message` option fields stripped; logic verbatim. |
| `hono/middleware/cors.ts` | `src/middleware/cors/index.ts` | JSDoc + the OPTIONS-preflight branch + `findAllowMethods` dropped for brevity; simple-request path verbatim (non-auth, for category variety). |
