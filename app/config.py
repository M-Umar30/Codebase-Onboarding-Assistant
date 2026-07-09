"""app/config.py — settings and per-node model assignment.

Exact provider/model names are config values, env-overridable, and never
hardcoded in node code. Node code calls `get_settings().node_config(name)`
(indirectly, via app.llm.get_chat_model) instead of branching on provider.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openai", "groq"]
NodeName = Literal["planner", "drafter", "critic", "synthesizer"]
EmbedderProvider = Literal["openai", "local"]


class NodeModelConfig(BaseModel):
    """Resolved provider + model for a single graph node."""

    provider: Provider
    model: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM providers ---
    openai_api_key: str | None = None
    groq_api_key: str | None = None

    # --- database ---
    database_url: str = "postgresql://onboard:onboard@localhost:5432/onboard"

    # --- embeddings ---
    embedder_provider: EmbedderProvider = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Must match the `chunks.embedding` column dimension in db/migrations —
    # 1536 for OpenAI text-embedding-3-small, 384 for the local fallback.
    embedding_dim: int = 1536

    # --- per-node model assignment ---
    # Default split: Groq (fast Llama) for planner + drafter; OpenAI for
    # critic semantic verification + routing verdict + synthesizer.
    planner_provider: Provider = "groq"
    planner_model: str = "llama-3.3-70b-versatile"

    drafter_provider: Provider = "groq"
    drafter_model: str = "llama-3.3-70b-versatile"

    critic_provider: Provider = "openai"
    critic_model: str = "gpt-4o-mini"

    synthesizer_provider: Provider = "openai"
    synthesizer_model: str = "gpt-4o-mini"

    def node_config(self, node_name: NodeName) -> NodeModelConfig:
        provider, model = {
            "planner": (self.planner_provider, self.planner_model),
            "drafter": (self.drafter_provider, self.drafter_model),
            "critic": (self.critic_provider, self.critic_model),
            "synthesizer": (self.synthesizer_provider, self.synthesizer_model),
        }[node_name]
        return NodeModelConfig(provider=provider, model=model)

    def api_key_for(self, provider: Provider) -> str | None:
        return {"openai": self.openai_api_key, "groq": self.groq_api_key}[provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
