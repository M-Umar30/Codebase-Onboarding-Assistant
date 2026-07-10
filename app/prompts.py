"""app/prompts.py — prompt text for the drafter, synthesizer, and critic nodes.

Kept out of node modules so prompt wording can be iterated on (and, later,
eval'd) without touching pipeline code.
"""

from __future__ import annotations

from app.schemas import Citation, CitationVerdict, DraftAnswer, RetrievedChunk

PLANNER_SYSTEM_PROMPT = """You are the planner agent in a codebase-onboarding \
assistant. You receive one question about an unfamiliar codebase and decide how \
to retrieve evidence for it. Retrieval is dense + lexical search over code \
chunks, so sub-queries should read like search queries a developer would type — \
concrete identifiers, symbols, and behaviors — not restatements of the question.

Decide between two modes:
- SKIP decomposition (`decomposed = false`) for a narrow lookup that maps to a \
single search: "where is `refresh_token` validated", "what does `CORSMiddleware` \
do", a single-symbol or single-file question. Return exactly ONE sub-query.
- DECOMPOSE (`decomposed = true`) for a broad question that spans several parts \
of the codebase: "how does auth work here?" touches routes, middleware, and \
token validation. Break it into 2-4 focused sub-queries, each targeting one \
facet, so retrieval can gather evidence for every part.

Rules:
- Prefer skipping. Only decompose when the question genuinely has multiple \
distinct facets that separate searches would serve better than one.
- Each sub-query must be independently searchable and name concrete things \
(function names, class names, concepts) likely to appear in the code.
- `reasoning` briefly explains the decompose-or-skip choice.
- Do not answer the question or invent file names — you only plan retrieval.
"""

PLANNER_USER_TEMPLATE = """Question: {question}

Decide whether to decompose, and emit the sub-queries, following the rules in \
your instructions."""


def build_planner_prompt(question: str) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the planner's structured call.
    The planner's output is validated and its sub-query ids are assigned in
    Python (app/nodes/planner.py) — the prompt only shapes the decision."""
    return PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE.format(question=question)

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

DRAFTER_GUIDANCE_TEMPLATE = """

A reviewer checked your PREVIOUS draft against the actual code and found \
problems. You MUST fix these in this attempt, using the retrieved chunks above:
{guidance}"""


SYNTHESIZER_SYSTEM_PROMPT = """You are the synthesizer agent in a \
codebase-onboarding assistant. You receive a draft answer plus the subset of \
its citations that survived the critic's verification, and you produce the \
final answer shown to the user.

Rules:
- Use ONLY the verified citations listed below. They are the sole evidence \
you may present as fact.
- Any claim in the draft whose citation was DROPPED (listed separately) must \
NOT appear as fact in your answer — remove it, or rephrase it as an explicit \
unknown (e.g. "I could not verify where X happens"). Never restate a dropped \
claim as if it were confirmed.
- Renumber the verified citations sequentially starting at 1, in the order \
they first appear in your final answer text, and update the inline [n] \
markers in the answer text to match. Every [n] marker in your answer must \
correspond to a returned citation, and every returned citation must be \
referenced by at least one [n] marker.
- Copy each verified citation's file_path, start_line, end_line, symbol, and \
claim EXACTLY as given — only the id may change (from renumbering).
- Do not invent new citations or new factual claims. You may lightly tighten \
wording for clarity."""

SYNTHESIZER_USER_TEMPLATE = """Question: {question}

Draft answer (its [n] markers use the DRAFT numbering — you will renumber):
{answer_markdown}

Verified citations — keep these, renumbered 1..n by first appearance \
(shown as draft_id: file_path:start-end (symbol) — claim):
{verified}

Dropped citations — their claims must NOT appear as fact in your answer:
{dropped}

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


def build_drafter_prompt(
    question: str, chunks: list[RetrievedChunk], guidance: str | None = None
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the drafter's structured call.

    `guidance` is the critic's regeneration_guidance on a `regenerate` loop —
    appended to the user prompt (per-attempt data, not a standing instruction)
    so the drafter fixes what the reviewer flagged.
    """
    context = format_chunks_for_drafter(chunks)
    user_prompt = DRAFTER_USER_TEMPLATE.format(question=question, context=context)
    if guidance:
        user_prompt += DRAFTER_GUIDANCE_TEMPLATE.format(guidance=guidance)
    return DRAFTER_SYSTEM_PROMPT, user_prompt


def _format_citation_line(citation: Citation) -> str:
    symbol = f" ({citation.symbol})" if citation.symbol else ""
    return (
        f"[{citation.id}]: {citation.file_path}:{citation.start_line}-"
        f"{citation.end_line}{symbol} — {citation.claim}"
    )


def build_synthesizer_prompt(
    question: str,
    draft: DraftAnswer,
    verified: list[Citation],
    dropped: list[tuple[Citation, CitationVerdict]],
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the synthesizer's structured call.

    `verified` are the citations the critic confirmed; `dropped` pairs each
    rejected citation with its verdict so the prompt can name what must not be
    stated as fact. The synthesizer's structured output is renumbered/validated
    in Python (app/nodes/synthesizer.py) — the prompt only shapes the prose.
    """
    verified_block = "\n".join(_format_citation_line(c) for c in verified) or "(none)"
    dropped_block = (
        "\n".join(
            f"{_format_citation_line(c)} [{verdict.status.value}]" for c, verdict in dropped
        )
        or "(none)"
    )
    user_prompt = SYNTHESIZER_USER_TEMPLATE.format(
        question=question,
        answer_markdown=draft.answer_markdown,
        verified=verified_block,
        dropped=dropped_block,
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
