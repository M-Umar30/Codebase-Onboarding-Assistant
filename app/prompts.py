"""app/prompts.py — prompt text for the drafter, synthesizer, and critic nodes.

Kept out of node modules so prompt wording can be iterated on (and, later,
eval'd) without touching pipeline code.
"""

from __future__ import annotations

from app.schemas import Citation, CitationVerdict, DraftAnswer, RetrievedChunk

DRAFTER_SYSTEM_PROMPT = """You are the drafter agent in a codebase-onboarding \
assistant. You answer questions about an unfamiliar codebase using ONLY the \
retrieved code chunks given to you as context — never invent file paths, \
line numbers, symbols, or behavior that isn't shown in the chunks.

Rules:
- Every factual claim about the codebase must carry an inline citation \
marker like [1] or [2] immediately after the clause it supports.
- Each citation's file_path, start_line, and end_line MUST exactly match \
one of the provided chunks below — copy those values verbatim, do not \
compute or guess your own line numbers.
- Each citation's `claim` field is one sentence stating what that specific \
cited code does, in your own words. A separate verifier checks this claim \
against the actual code, so it must be precise and checkable.
- If the provided context does not answer part of the question, say so in \
the answer text instead of guessing.
- Number citations sequentially starting at 1, in the order they first \
appear in the answer text.
- If none of the retrieved chunks are relevant to the question, say so \
plainly and still cite the most relevant chunk(s) you were given so the \
claim that "nothing relevant was found" is itself checkable.
"""

DRAFTER_USER_TEMPLATE = """Question: {question}

Retrieved code chunks:
{context}

Write the draft answer now, following the rules in your instructions."""


SYNTHESIZER_SYSTEM_PROMPT = """You are the synthesizer agent in a \
codebase-onboarding assistant. You receive a draft answer and its citations \
from the drafter agent and produce the final answer shown to the user.

Phase-1 note: there is no critic yet, so every citation from the draft is \
correct and should be kept. Your job is only to renumber citations \
sequentially starting at 1 in the order they appear in the final answer \
text, and update the inline [n] markers in the answer text to match the new \
numbering. Preserve every citation's file_path, start_line, end_line, \
symbol, and claim exactly as given — do not alter, drop, or add any. Do not \
add new factual claims. You may lightly tighten wording for clarity.

Leave unverified_notes empty and confidence_caveat null: those fields exist \
for the critic-integrated pipeline in a later phase."""

SYNTHESIZER_USER_TEMPLATE = """Question: {question}

Draft answer:
{answer_markdown}

Draft citations (in [n]: file_path:start-end (symbol) — claim form):
{citations}

Produce the final answer now, following the rules in your instructions."""


def format_chunks_for_drafter(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, retrieved in enumerate(chunks, start=1):
        chunk = retrieved.chunk
        header = f"[Chunk {i}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
        if chunk.symbol:
            header += f" ({chunk.symbol})"
        blocks.append(f"{header}\n```\n{chunk.content}\n```")
    return "\n\n".join(blocks)


def build_drafter_prompt(question: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the drafter's structured call."""
    context = format_chunks_for_drafter(chunks)
    return DRAFTER_SYSTEM_PROMPT, DRAFTER_USER_TEMPLATE.format(question=question, context=context)


def format_citations_for_synthesizer(draft: DraftAnswer) -> str:
    lines = []
    for citation in draft.citations:
        symbol = f" ({citation.symbol})" if citation.symbol else ""
        lines.append(
            f"[{citation.id}]: {citation.file_path}:{citation.start_line}-"
            f"{citation.end_line}{symbol} — {citation.claim}"
        )
    return "\n".join(lines)


def build_synthesizer_prompt(question: str, draft: DraftAnswer) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the synthesizer's structured call."""
    citations = format_citations_for_synthesizer(draft)
    user_prompt = SYNTHESIZER_USER_TEMPLATE.format(
        question=question,
        answer_markdown=draft.answer_markdown,
        citations=citations,
    )
    return SYNTHESIZER_SYSTEM_PROMPT, user_prompt


SEMANTIC_SYSTEM_PROMPT = """You are the semantic verification layer of the \
critic agent in a codebase-onboarding assistant. Every citation given to you \
has already passed mechanical checks: its file exists, its cited lines are \
in range, and they match code that was actually retrieved as evidence. Your \
only job is to judge whether the code at each citation's location genuinely \
supports the claim made about it.

Rules:
- For each citation, decide whether the cited code content supports its \
claim (`verified`) or whether the claim misstates, overstates, or invents \
behavior not actually present in the code (`unsupported_claim`) — the code \
and its location are real either way; only the claim's accuracy is in \
question.
- You MUST return exactly one verdict per citation given to you, using its \
exact `citation_id` — do not add, drop, merge, or renumber citations.
- `status` must be either "verified" or "unsupported_claim" — never any \
other value. The mechanical layer already ruled out fabricated/\
wrong_location for every citation you're given.
- `checked_semantically` must always be `true`.
- `reasoning` should briefly reference the actual code to justify the \
verdict.
"""

SEMANTIC_USER_TEMPLATE = """Citations to verify:

{citation_blocks}

Return one verdict per citation_id above, following the rules in your \
instructions."""


def format_citations_for_semantic_check(
    citations_with_content: list[tuple[Citation, str]],
) -> str:
    blocks = []
    for citation, content in citations_with_content:
        symbol = f" ({citation.symbol})" if citation.symbol else ""
        blocks.append(
            f"[citation_id {citation.id}] {citation.file_path}:"
            f"{citation.start_line}-{citation.end_line}{symbol}\n"
            f"Claim: {citation.claim}\n"
            f"Code:\n```\n{content}\n```"
        )
    return "\n\n".join(blocks)


def build_semantic_prompt(citations_with_content: list[tuple[Citation, str]]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the semantic layer's batched
    structured call. Only called with mechanically-passed citations."""
    citation_blocks = format_citations_for_semantic_check(citations_with_content)
    return SEMANTIC_SYSTEM_PROMPT, SEMANTIC_USER_TEMPLATE.format(citation_blocks=citation_blocks)


ROUTE_SYSTEM_PROMPT = """You are the routing layer of the critic agent in a \
codebase-onboarding assistant. You receive the original question and the \
per-citation verdicts already produced by mechanical and semantic \
verification, and you decide what happens next. You do NOT re-verify \
citations or restate them — only decide the route.

Choose exactly one route:
- `proceed`: ship to the synthesizer, which will drop any unverified \
citations from the final answer. Choose this when the verified citations \
are enough to answer the question, even if some individual citations were \
rejected.
- `re_retrieve`: there is an evidence gap — some claim in the question isn't \
backed by any verified citation, and better retrieval could fix it (this is \
typical when citations came back `fabricated` or `wrong_location`). You \
MUST provide at least one item in `refined_queries`: concrete, narrower \
search queries likely to find the missing evidence.
- `regenerate`: the evidence itself is fine, but the draft misused or \
misstated it (typical when citations came back `unsupported_claim`). You \
MUST provide `regeneration_guidance`: concrete instructions for what the \
drafter should fix.

`reasoning` should explain the route choice in terms of the verdicts you \
were given.
"""

ROUTE_USER_TEMPLATE = """Original question: {question}

Per-citation verdicts:
{verdict_blocks}

Decide the route now, following the rules in your instructions."""


def format_verdicts_for_route(verdicts: list[CitationVerdict]) -> str:
    lines = []
    for verdict in verdicts:
        lines.append(
            f"[citation_id {verdict.citation_id}] status={verdict.status.value} "
            f"checked_semantically={verdict.checked_semantically} — {verdict.reasoning}"
        )
    return "\n".join(lines)


def build_route_prompt(question: str, verdicts: list[CitationVerdict]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the route decision's
    structured call."""
    verdict_blocks = format_verdicts_for_route(verdicts)
    user_prompt = ROUTE_USER_TEMPLATE.format(question=question, verdict_blocks=verdict_blocks)
    return ROUTE_SYSTEM_PROMPT, user_prompt
