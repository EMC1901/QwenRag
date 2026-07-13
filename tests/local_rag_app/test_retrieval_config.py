"""Configuration tests for the stage-1 local-retrieval contract."""

import pytest
from pydantic import ValidationError

from local_rag_app.config import PROJECT_ROOT, Settings


GATEWAY_SETTINGS = {
    "LOCAL_RAG_ANSWER_MODE": "gateway",
    "MODEL_GATEWAY_BASE_URL": "http://gateway.test:8010/v1",
    "MODEL_GATEWAY_API_KEY": "test-key",
    "UPSTREAM_LLM_MODEL": "qwen",
}


def retrieval_settings(**overrides) -> Settings:
    """Build an active retrieval configuration without reading developer files."""
    values = {
        **GATEWAY_SETTINGS,
        "ENABLE_RAG_ROUTER": True,
        "ENABLE_LOCAL_RETRIEVAL": True,
        "UPSTREAM_EMBEDDING_MODEL": "qwen3-embedding-0.6b",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_retrieval_defaults_are_safe_and_feature_is_disabled() -> None:
    """Stage 1 must not activate local retrieval in existing installations."""
    settings = Settings(_env_file=None)

    assert settings.enable_local_retrieval is False
    assert settings.rag_knowledge_base_dir == (PROJECT_ROOT / "rag_data").resolve()
    assert settings.upstream_embedding_model == "qwen3-embedding-0.6b"
    assert settings.rag_embedding_dim == 1024
    assert settings.rag_vector_top_k == 40
    assert settings.rag_fts_top_k == 40
    assert settings.rag_final_top_k == 8
    assert settings.rag_max_chunks_per_doc == 3
    assert settings.rag_enable_fts is True
    assert settings.rag_allow_fts_fallback is True
    assert settings.rag_rrf_k == 60
    assert settings.rag_vector_weight == pytest.approx(0.6)
    assert settings.rag_fts_weight == pytest.approx(0.4)
    assert settings.rag_allow_partial_index is False


def test_gateway_configuration_remains_valid_when_retrieval_is_disabled() -> None:
    """An unused embedding model must not become a new gateway-mode requirement."""
    settings = Settings(
        **GATEWAY_SETTINGS,
        ENABLE_RAG_ROUTER=True,
        ENABLE_LOCAL_RETRIEVAL=False,
        UPSTREAM_EMBEDDING_MODEL="",
        _env_file=None,
    )

    assert settings.enable_rag_router is True
    assert settings.enable_local_retrieval is False
    assert settings.upstream_embedding_model == ""


def test_active_retrieval_accepts_complete_configuration_without_io() -> None:
    """Settings validation resolves paths but does not require assets or a network."""
    settings = retrieval_settings(
        RAG_KNOWLEDGE_BASE_DIR="does-not-exist-yet",
    )

    assert settings.enable_local_retrieval is True
    assert settings.rag_knowledge_base_dir == (
        PROJECT_ROOT / "does-not-exist-yet"
    ).resolve()
    assert not settings.rag_knowledge_base_dir.exists()


def test_active_retrieval_requires_embedding_model() -> None:
    """The query model identity is required once retrieval can be called."""
    with pytest.raises(ValidationError, match="UPSTREAM_EMBEDDING_MODEL"):
        retrieval_settings(UPSTREAM_EMBEDDING_MODEL="  ")


@pytest.mark.parametrize("dimension", [0, -1, "not-an-integer", ""])
def test_rejects_invalid_embedding_dimension(dimension: int | str) -> None:
    """An invalid query-vector dimension must fail before application startup."""
    with pytest.raises(ValidationError):
        retrieval_settings(RAG_EMBEDDING_DIM=dimension)


@pytest.mark.parametrize(
    "field",
    ["RAG_VECTOR_TOP_K", "RAG_FTS_TOP_K", "RAG_FINAL_TOP_K"],
)
def test_rejects_non_positive_candidate_limits(field: str) -> None:
    """Every retrieval candidate limit must be a positive integer."""
    with pytest.raises(ValidationError):
        retrieval_settings(**{field: 0})


def test_rejects_final_top_k_larger_than_all_candidates() -> None:
    """The final result count cannot exceed the two candidate pools combined."""
    with pytest.raises(ValidationError, match="RAG_FINAL_TOP_K"):
        retrieval_settings(
            RAG_VECTOR_TOP_K=2,
            RAG_FTS_TOP_K=3,
            RAG_FINAL_TOP_K=6,
        )


def test_rejects_per_document_limit_larger_than_final_top_k() -> None:
    """Per-document selection cannot exceed the complete final result set."""
    with pytest.raises(ValidationError, match="RAG_MAX_CHUNKS_PER_DOC"):
        retrieval_settings(RAG_FINAL_TOP_K=2, RAG_MAX_CHUNKS_PER_DOC=3)


def test_rejects_zero_total_retrieval_weight() -> None:
    """At least one retrieval source must contribute to the final ranking."""
    with pytest.raises(ValidationError, match="positive sum"):
        retrieval_settings(RAG_VECTOR_WEIGHT=0, RAG_FTS_WEIGHT=0)


def test_vector_weight_must_be_positive_when_fts_is_disabled() -> None:
    """Pure-vector mode cannot be configured with a zero vector weight."""
    with pytest.raises(ValidationError, match="RAG_VECTOR_WEIGHT"):
        retrieval_settings(
            RAG_ENABLE_FTS=False,
            RAG_VECTOR_WEIGHT=0,
            RAG_FTS_WEIGHT=1,
        )


def test_resolves_absolute_knowledge_base_path_without_rewriting_it() -> None:
    """Absolute delivery paths remain absolute and normalized."""
    absolute_path = (PROJECT_ROOT / "test-absolute-kb-path").resolve()
    settings = retrieval_settings(RAG_KNOWLEDGE_BASE_DIR=absolute_path)

    assert settings.rag_knowledge_base_dir == absolute_path
    assert not absolute_path.exists()


def test_rejects_empty_knowledge_base_directory() -> None:
    """An empty root must not silently resolve to the process working directory."""
    with pytest.raises(ValidationError, match="RAG_KNOWLEDGE_BASE_DIR"):
        retrieval_settings(RAG_KNOWLEDGE_BASE_DIR=" ")
