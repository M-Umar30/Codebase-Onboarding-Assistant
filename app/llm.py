"""app/llm.py — provider-agnostic chat model factory.

Node code calls get_chat_model(node_name); it never branches on provider
itself. Which provider/model backs a node is entirely a config decision
(see app/config.py), reused from project #1's get_chat_model() pattern.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.config import NodeName, Settings, get_settings


def get_chat_model(node_name: NodeName, settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    node_config = settings.node_config(node_name)
    api_key = settings.api_key_for(node_config.provider)

    if node_config.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=node_config.model, api_key=api_key)

    if node_config.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=node_config.model, api_key=api_key)

    raise ValueError(f"Unknown provider: {node_config.provider}")
