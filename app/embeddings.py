"""app/embeddings.py — embedding backend behind a provider-agnostic port.

Same fallback pattern as app.llm.get_chat_model: node/indexing code depends
on the Embedder protocol, never on a specific SDK. OpenAIEmbedder is the
default; LocalEmbedder (sentence-transformers) lets demos run without API
credits.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.config import Settings, get_settings


@runtime_checkable
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """Wraps OpenAI text-embedding-3-small (default) via langchain-openai."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(model=model, api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


class LocalEmbedder:
    """Local sentence-transformers fallback — no API credits required."""

    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.encode(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()

    if settings.embedder_provider == "openai":
        return OpenAIEmbedder(
            model=settings.openai_embedding_model, api_key=settings.openai_api_key
        )

    if settings.embedder_provider == "local":
        return LocalEmbedder(model=settings.local_embedding_model)

    raise ValueError(f"Unknown embedder provider: {settings.embedder_provider}")
