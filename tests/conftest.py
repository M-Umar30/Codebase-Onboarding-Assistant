"""Shared fixtures. Tests never read the real .env — settings are built
in-process with explicit test values so no real API key is required and no
network call can happen by accident.
"""

from __future__ import annotations

import pytest

from app.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-openai-key",
        groq_api_key="test-groq-key",
    )
