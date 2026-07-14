"""Tests for strict internal RAG context data contracts."""

import pytest
from pydantic import ValidationError

from local_rag_app.context_models import ContextBuildResult, SelectedContextHit
from local_rag_app.retrieval_models import RetrievalHit


def _hit() -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        chunk_id="chunk-1",
        doc_id="doc-1",
        chunk_text="资料正文",
        title="测试资料",
        relative_path="fixtures/test.docx",
        final_score=0.5,
        matched_by="vector",
    )


def _selected() -> SelectedContextHit:
    return SelectedContextHit(
        evidence_no=1,
        hit=_hit(),
        text_for_prompt="资料正文",
        estimated_tokens=10,
    )


def test_context_models_accept_a_traceable_selected_hit() -> None:
    """The final prompt can retain the full source record for stage 8."""
    result = ContextBuildResult(
        system_prompt="系统提示词",
        user_prompt="用户提示词",
        selected_hits=[_selected()],
        dropped_hit_count=2,
        estimated_input_tokens=100,
        estimated_context_tokens=10,
        estimated_history_tokens=0,
        history_message_count=0,
    )

    assert result.selected_hits[0].hit.chunk_id == "chunk-1"
    assert result.selected_hits[0].truncated is False


@pytest.mark.parametrize(
    "field, value",
    [
        ("evidence_no", 0),
        ("text_for_prompt", ""),
        ("estimated_tokens", 0),
    ],
)
def test_selected_context_hit_rejects_invalid_required_values(
    field: str,
    value: int | str,
) -> None:
    """Evidence numbering and token metrics must be meaningful and non-empty."""
    values = _selected().model_dump()
    values[field] = value

    with pytest.raises(ValidationError):
        SelectedContextHit(**values)


def test_context_models_forbid_unknown_fields() -> None:
    """Internal contracts must not silently carry accidental private fields."""
    with pytest.raises(ValidationError):
        ContextBuildResult(
            system_prompt="系统提示词",
            user_prompt="用户提示词",
            selected_hits=[],
            dropped_hit_count=0,
            estimated_input_tokens=1,
            estimated_context_tokens=0,
            estimated_history_tokens=0,
            history_message_count=0,
            private_prompt_dump="forbidden",
        )
