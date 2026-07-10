"""Shared pytest helpers for local RAG application tests."""

import pytest

from local_rag_app.config import reset_settings_cache


@pytest.fixture(autouse=True)
def reset_cached_settings(monkeypatch):
    """Keep tests isolated from local developer credentials and gateway settings."""
    monkeypatch.setenv("LOCAL_RAG_DISABLE_ENV_FILE", "true")
    reset_settings_cache()
    yield
    reset_settings_cache()
