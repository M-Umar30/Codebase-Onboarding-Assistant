"""Smoke tests for app/llm.py — construct get_chat_model for every node
name with test-only credentials. No network call happens: constructing a
LangChain chat model client only sets up parameters, it doesn't call out.
"""

from __future__ import annotations

import pytest
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.config import NodeName, Settings
from app.llm import get_chat_model

NODE_NAMES: list[NodeName] = ["planner", "drafter", "critic", "synthesizer"]


@pytest.mark.parametrize("node_name", NODE_NAMES)
def test_get_chat_model_constructs_for_every_node(node_name: NodeName, test_settings: Settings) -> None:
    model = get_chat_model(node_name, settings=test_settings)
    assert model is not None


def test_planner_uses_groq(test_settings: Settings) -> None:
    model = get_chat_model("planner", settings=test_settings)
    assert isinstance(model, ChatGroq)


def test_drafter_uses_groq(test_settings: Settings) -> None:
    model = get_chat_model("drafter", settings=test_settings)
    assert isinstance(model, ChatGroq)


def test_critic_uses_openai(test_settings: Settings) -> None:
    model = get_chat_model("critic", settings=test_settings)
    assert isinstance(model, ChatOpenAI)


def test_synthesizer_uses_openai(test_settings: Settings) -> None:
    model = get_chat_model("synthesizer", settings=test_settings)
    assert isinstance(model, ChatOpenAI)


def test_unknown_provider_raises() -> None:
    settings = Settings(_env_file=None, openai_api_key="k", groq_api_key="k")
    settings.critic_provider = "anthropic"  # bypass Literal typing, exercise runtime guard
    with pytest.raises(ValueError):
        get_chat_model("critic", settings=settings)
