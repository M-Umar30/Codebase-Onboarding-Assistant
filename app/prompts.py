"""app/prompts.py — prompt text for the drafter and synthesizer nodes.

Kept out of node modules so prompt wording can be iterated on (and, later,
eval'd) without touching pipeline code.
"""

from __future__ import annotations

from app.schemas import DraftAnswer, RetrievedChunk

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
