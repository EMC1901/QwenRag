"""Strict internal data contracts for user-visible RAG references."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


PositiveInt = Annotated[int, Field(ge=1)]


class ReferenceFile(BaseModel):
    """One deduplicated source file displayed beneath a RAG answer."""

    model_config = ConfigDict(extra="forbid", strict=True)

    reference_no: PositiveInt
    display_name: str = Field(min_length=1)
    locations: list[str]
    evidence_nos: list[PositiveInt] = Field(min_length=1)


class ReferenceBuildResult(BaseModel):
    """A complete deterministic reference section and safe aggregate counts."""

    model_config = ConfigDict(extra="forbid", strict=True)

    section_text: str = Field(min_length=1)
    files: list[ReferenceFile] = Field(min_length=1)
    selected_hit_count: PositiveInt
    location_count: int = Field(ge=0)
