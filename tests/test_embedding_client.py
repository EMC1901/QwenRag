"""测试 embedding_client，不访问真实网络。"""

import math

from rag_preprocess import embedding_client
from rag_preprocess.embedding_client import (
    embed_batch,
    normalize_embedding,
    validate_embedding,
)


class _FakeResponse:
    def __init__(self, data, status_ok=True):
        self._data = data
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("http error")

    def json(self):
        return self._data


def test_embed_batch_parses_openai_compatible_response(monkeypatch):
    def fake_post_embeddings(*, base_url, payload, timeout):
        assert base_url == "http://example.test/v1"
        assert payload["model"] == "test-model"
        assert payload["input"] == ["a", "b"]
        return _FakeResponse({
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        })

    monkeypatch.setattr(embedding_client, "_post_embeddings", fake_post_embeddings)

    results = embed_batch(
        ["a", "b"],
        model="test-model",
        batch_size=2,
        base_url="http://example.test/v1",
        max_retries=0,
    )

    assert [r.success for r in results] == [True, True]
    assert results[0].vector == [1.0, 0.0]
    assert results[1].vector == [0.0, 1.0]


def test_embed_batch_returns_failed_results_on_error(monkeypatch):
    def fake_post_embeddings(*, base_url, payload, timeout):
        raise RuntimeError("service down")

    monkeypatch.setattr(embedding_client, "_post_embeddings", fake_post_embeddings)

    results = embed_batch(
        ["a", "b"],
        model="test-model",
        batch_size=2,
        base_url="http://example.test/v1",
        max_retries=0,
    )

    assert len(results) == 2
    assert all(not r.success for r in results)
    assert "service down" in (results[0].error_message or "")


def test_validate_embedding_checks_dim_and_numbers():
    assert validate_embedding([0.1, 0.2], expected_dim=2)
    assert not validate_embedding([0.1], expected_dim=2)
    assert not validate_embedding([math.inf, 0.2], expected_dim=2)


def test_normalize_embedding():
    vector = normalize_embedding([3.0, 4.0])
    assert round(vector[0], 6) == 0.6
    assert round(vector[1], 6) == 0.8
    assert normalize_embedding([0.0, 0.0]) == [0.0, 0.0]
