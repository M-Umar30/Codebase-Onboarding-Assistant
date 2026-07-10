"""evals/smoke/run_smoke.py — end-to-end smoke set (eval #3 from CLAUDE.md).

Proves the whole pipeline on REAL codebases: it obtains full checkouts of two
pinned public repos (fastapi and hono, at the tags/commits recorded in
evals/fixtures/repos/MANIFEST.md), indexes each into Postgres, asks ~10
questions with known target files, and asserts every question's final
citations land in an expected file. This is the demo story and the honest
end-to-end signal — unlike the vendored critic fixtures, retrieval and the
planner's fan-out actually have to work here.

Scoping decision: indexing is pointed at each repo's *source package*
(`source_subdir` in questions.json — `fastapi/` and `src/`), not the whole
checkout. fastapi's checkout is ~1600 markdown translation files a developer
never onboards on; embedding them would be wasteful and would drown the source
in retrieval. Expected-file paths are relative to that subdir; the pinned
commit is still verified against the full checkout.

Not part of the pytest suite: it clones repos, embeds them (real OpenAI calls),
and runs the graph (real Groq + OpenAI calls). Run it directly:

    python evals/smoke/run_smoke.py

Requirements: docker Postgres up (docker compose up -d), OPENAI_API_KEY and
GROQ_API_KEY set, and network access for the first clone (cached afterwards
under evals/smoke/.cache/, which is gitignored). Point SMOKE_FASTAPI_PATH /
SMOKE_HONO_PATH at local checkouts to run offline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.db import apply_migrations, get_connection
from app.graph import ask_with_trace
from app.indexing.indexer import index_repo

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache"
QUESTIONS_PATH = HERE / "questions.json"


@dataclass
class RepoPin:
    repo_id: str
    name: str
    url: str
    tag: str
    commit: str
    path_env: str
    source_subdir: str = ""


@dataclass
class QuestionResult:
    id: str
    repo_id: str
    question: str
    decomposed: bool
    n_sub_queries: int
    expected_files: list[str]
    cited_files: list[str]
    passed: bool
    error: str | None = None


def _run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _resolve_checkout(pin: RepoPin) -> Path:
    """Return a local checkout at the pinned commit, cloning into the cache if
    needed. Fails loudly if the resolved HEAD isn't the pinned commit — the
    smoke numbers are only meaningful against the recorded pins."""
    override = os.environ.get(pin.path_env)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"{pin.path_env}={path} is not a directory")
    else:
        path = CACHE_DIR / pin.name
        if not path.exists():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            print(f"  cloning {pin.name} @ {pin.tag} ...")
            _run_git("clone", "--depth", "1", "--branch", pin.tag, pin.url, str(path))

    git_dir = path / ".git"
    if git_dir.exists():
        head = _run_git("-C", str(path), "rev-parse", "HEAD")
        if head != pin.commit:
            raise SystemExit(
                f"{pin.name} checkout is at {head}, expected pinned {pin.commit}. "
                f"Reproducibility guard: check out the pinned commit or clear the cache."
            )
    else:
        print(f"  WARNING: {path} is not a git checkout; cannot verify the pinned commit.")

    if pin.source_subdir:
        source = path / pin.source_subdir
        if not source.is_dir():
            raise SystemExit(f"source_subdir {source} not found in checkout")
        return source
    return path


def _cited_files(repo_id: str, question: str) -> tuple[list[str], bool, int]:
    response = ask_with_trace(repo_id, question)
    files = [c.file_path for c in response.answer.citations]
    plan = response.trace.plan
    return files, plan.decomposed, len(plan.sub_queries)


def _cleanup(conn, repo_id: str) -> None:
    conn.execute("DELETE FROM chunks WHERE repo_id = %s", (repo_id,))
    conn.execute("DELETE FROM repos WHERE repo_id = %s", (repo_id,))


def main() -> int:
    spec = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    pins = {
        repo_id: RepoPin(repo_id=repo_id, **cfg) for repo_id, cfg in spec["repos"].items()
    }
    questions = spec["questions"]

    settings = get_settings()
    if not settings.openai_api_key or not settings.groq_api_key:
        raise SystemExit("OPENAI_API_KEY and GROQ_API_KEY must be set for the smoke run.")

    conn = get_connection(settings)
    apply_migrations(conn)

    results: list[QuestionResult] = []
    try:
        for repo_id, pin in pins.items():
            print(f"\n[{repo_id}] resolving checkout ...")
            checkout = _resolve_checkout(pin)
            print(f"[{repo_id}] indexing {checkout} ...")
            index = index_repo(checkout, repo_id=repo_id, settings=settings, conn=conn)
            print(
                f"[{repo_id}] {index.files_indexed} files, {index.chunks_written} chunks, "
                f"{index.fallback_language_files} line-based fallback files"
            )

            for q in (x for x in questions if x["repo_id"] == repo_id):
                expected = q["expected_files"]
                # Isolate each question: a flaky LLM output (e.g. the synthesizer's
                # strict marker-parity check tripping) must not abort the whole run.
                try:
                    cited, decomposed, n_sub = _cited_files(repo_id, q["question"])
                    passed = bool(set(cited) & set(expected))
                    error = None
                except Exception as exc:  # noqa: BLE001 - eval harness, report don't crash
                    cited, decomposed, n_sub, passed = [], False, 0, False
                    error = f"{type(exc).__name__}: {exc}"
                results.append(
                    QuestionResult(
                        id=q["id"],
                        repo_id=repo_id,
                        question=q["question"],
                        decomposed=decomposed,
                        n_sub_queries=n_sub,
                        expected_files=expected,
                        cited_files=cited,
                        passed=passed,
                        error=error,
                    )
                )
                status = "PASS" if passed else ("ERROR" if error else "FAIL")
                plan_desc = f"{'decomp' if decomposed else 'skip'}/{n_sub}sq"
                print(f"  [{status}] {q['id']:<32} ({plan_desc})")
    finally:
        for repo_id in pins:
            _cleanup(conn, repo_id)
        conn.close()

    _print_report(results)
    return 0 if all(r.passed for r in results) else 1


def _print_report(results: list[QuestionResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    print("\n" + "=" * 72)
    print(f" smoke results: {passed}/{len(results)} passed")
    print("=" * 72)
    print(f"| {'id':<32} | {'plan':<10} | {'result':<6} | files cited |")
    print(f"|{'-'*34}|{'-'*12}|{'-'*8}|{'-'*13}|")
    for r in results:
        plan_desc = f"{'decomp' if r.decomposed else 'skip'}/{r.n_sub_queries}"
        print(f"| {r.id:<32} | {plan_desc:<10} | {'PASS' if r.passed else 'FAIL':<6} | {', '.join(r.cited_files) or '(none)'}")
    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for r in failures:
            if r.error:
                print(f"  [{r.id}] errored: {r.error}")
            else:
                print(f"  [{r.id}] expected one of {r.expected_files}, cited {r.cited_files or '(none)'}")


if __name__ == "__main__":
    sys.exit(main())
