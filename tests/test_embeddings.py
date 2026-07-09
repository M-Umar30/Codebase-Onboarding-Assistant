"""app/embeddings.py dispatch tests. Real backends aren't constructed here:
OpenAIEmbedder needs no network to build, but LocalEmbedder downloads model
weights on first use — so dispatch is verified against stub classes instead
of hitting sentence-transformers.
"""

from __future__ import annotations

import pytest

from app import embeddings
from app.config import Settings


class _StubEmbedder:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


def test_get_embedder_dispatches_to_openai(monkeypatch: pytest.MonkeyPatch, test_settings: Settings) -> None:
    monkeypatch.setattr(embeddings, "OpenAIEmbedder", _StubEmbedder)
    settings = test_settings.model_copy(update={"embedder_provider": "openai"})
    embedder = embeddings.get_embedder(settings)
    assert isinstance(embedder, _StubEmbedder)


def test_get_embedder_dispatches_to_local(monkeypatch: pytest.MonkeyPatch, test_settings: Settings) -> None:
    monkeypatch.setattr(embeddings, "LocalEmbedder", _StubEmbedder)
    settings = test_settings.model_copy(update={"embedder_provider": "local"})
    embedder = embeddings.get_embedder(settings)
    assert isinstance(embedder, _StubEmbedder)


def test_get_embedder_rejects_unknown_provider(test_settings: Settings) -> None:
    settings = test_settings.model_copy(update={"embedder_provider": "bogus"})
    with pytest.raises(ValueError):
        embeddings.get_embedder(settings)
