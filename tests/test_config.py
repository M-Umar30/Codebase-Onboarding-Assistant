"""Smoke tests for app/config.py — config-only, no network."""

from __future__ import annotations

from app.config import NodeName, Settings


def test_default_node_assignment_matches_locked_decision(test_settings: Settings) -> None:
    assert test_settings.node_config("planner").provider == "groq"
    assert test_settings.node_config("drafter").provider == "groq"
    assert test_settings.node_config("critic").provider == "openai"
    assert test_settings.node_config("synthesizer").provider == "openai"


def test_node_config_returns_configured_model(test_settings: Settings) -> None:
    node_config = test_settings.node_config("critic")
    assert node_config.model == test_settings.critic_model


def test_all_node_names_resolve(test_settings: Settings) -> None:
    node_names: list[NodeName] = ["planner", "drafter", "critic", "synthesizer"]
    for node_name in node_names:
        node_config = test_settings.node_config(node_name)
        assert node_config.provider in ("openai", "groq")
        assert node_config.model


def test_api_key_for_dispatches_by_provider(test_settings: Settings) -> None:
    assert test_settings.api_key_for("openai") == "test-openai-key"
    assert test_settings.api_key_for("groq") == "test-groq-key"


def test_env_override(monkeypatch: object) -> None:
    settings = Settings(_env_file=None, planner_model="custom-model", openai_api_key="k", groq_api_key="k")
    assert settings.node_config("planner").model == "custom-model"
