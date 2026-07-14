"""Strict internal data contracts for RAG context construction."""

from pydantic import BaseModel, ConfigDict, Field

from local_rag_app.retrieval_models import RetrievalHit


class SelectedContextHit(BaseModel):
    """One retrieval hit actually included in the LLM prompt."""

    model_config = ConfigDict(extra="forbid", strict=True)

    evidence_no: int = Field(ge=1)
    hit: RetrievalHit
    text_for_prompt: str = Field(min_length=1)
    estimated_tokens: int = Field(gt=0)
    truncated: bool = False


class ContextBuildResult(BaseModel):
    """The private prompt payload and safe aggregate metrics for one request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    selected_hits: list[SelectedContextHit]
    dropped_hit_count: int = Field(ge=0)
    estimated_input_tokens: int = Field(gt=0)
    estimated_context_tokens: int = Field(ge=0)
    estimated_history_tokens: int = Field(ge=0)
    history_message_count: int = Field(ge=0)
