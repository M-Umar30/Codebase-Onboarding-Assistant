"""evals/run_critic_eval.py — measures the critic's mechanical + semantic
layers against the hand-built fixture set in evals/fixtures/.

Requires a live OPENAI_API_KEY (app.critic.semantic makes real structured
LLM calls) — unlike the fast pytest subset, which only exercises
app.critic.mechanical. The "index" for each fixture's repo is built the same
way production will build it: app.indexing.chunker.chunk_text() over the
vendored files, not a database (see the Phase 2 plan's decision #2).

Usage: python evals/run_critic_eval.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.critic.mechanical import run_mechanical_checks
from app.critic.semantic import run_semantic_checks
from app.indexing.chunker import chunk_text
from app.indexing.walker import iter_source_files
from app.schemas import Citation, CitationStatus, CitationVerdict, CodeChunk, DraftAnswer

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPOS_DIR = FIXTURES_DIR / "repos"
CATEGORIES = ["fabricated", "wrong_location", "unsupported_claim", "verified"]


@dataclass
class FixtureResult:
    fixture_id: str
    citation_id: int
    expected: CitationStatus
    verdict: CitationVerdict


def _load_fixtures() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(FIXTURES_DIR.glob("*.json"))
    ]


def _build_index(repo_root: Path) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for file_path in iter_source_files(repo_root):
        rel_path = file_path.relative_to(repo_root).as_posix()
        text = file_path.read_text(encoding="utf-8")
        chunks.extend(chunk_text(repo_id=repo_root.name, file_path=rel_path, text=text))
    return chunks


def _draft_from_fixture(fixture: dict) -> DraftAnswer:
    return DraftAnswer(
        answer_markdown=fixture["draft"]["answer_markdown"],
        citations=[Citation(**c) for c in fixture["draft"]["citations"]],
    )


def run_fixture(fixture: dict, settings) -> list[FixtureResult]:
    repo_root = REPOS_DIR / fixture["repo"]
    index = _build_index(repo_root)
    draft = _draft_from_fixture(fixture)

    mechanical_checks = run_mechanical_checks(draft, repo_root, index)
    verdicts = run_semantic_checks(draft, mechanical_checks, repo_root, settings=settings)
    verdicts_by_id = {v.citation_id: v for v in verdicts}

    return [
        FixtureResult(
            fixture_id=fixture["id"],
            citation_id=expected["citation_id"],
            expected=CitationStatus(expected["status"]),
            verdict=verdicts_by_id[expected["citation_id"]],
        )
        for expected in fixture["expected"]
    ]


def _print_table(results_by_category: dict[str, list[FixtureResult]]) -> None:
    print("| category | n | catch rate | status accuracy | false-positive rate |")
    print("|---|---|---|---|---|")
    for category in CATEGORIES:
        results = results_by_category.get(category, [])
        n = len(results)
        if n == 0:
            print(f"| {category} | 0 | - | - | - |")
            continue
        if category == "verified":
            false_positives = sum(1 for r in results if r.verdict.status != CitationStatus.VERIFIED)
            fp_rate = false_positives / n
            print(f"| {category} | {n} | - | - | {fp_rate:.0%} |")
        else:
            caught = sum(1 for r in results if r.verdict.status != CitationStatus.VERIFIED)
            exact = sum(1 for r in results if r.verdict.status == r.expected)
            print(f"| {category} | {n} | {caught / n:.0%} | {exact / n:.0%} | - |")


def _print_mismatches(results_by_category: dict[str, list[FixtureResult]]) -> None:
    mismatches = [
        r for results in results_by_category.values() for r in results if r.verdict.status != r.expected
    ]
    if not mismatches:
        print("\nNo mismatches - every fixture's citation landed on its expected status.")
        return
    print(f"\n{len(mismatches)} mismatch(es):")
    for r in mismatches:
        print(
            f"  [{r.fixture_id}] citation {r.citation_id}: expected={r.expected.value} "
            f"actual={r.verdict.status.value} - {r.verdict.reasoning}"
        )


def main() -> None:
    settings = get_settings()
    fixtures = _load_fixtures()

    results_by_category: dict[str, list[FixtureResult]] = defaultdict(list)
    for fixture in fixtures:
        print(f"running {fixture['id']}...")
        results_by_category[fixture["category"]].extend(run_fixture(fixture, settings))

    print()
    _print_table(results_by_category)
    _print_mismatches(results_by_category)


if __name__ == "__main__":
    main()
