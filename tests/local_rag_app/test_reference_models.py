"""Tests for strict user-visible RAG reference data contracts."""

import pytest
from pydantic import ValidationError

from local_rag_app.reference_models import ReferenceBuildResult, ReferenceFile


def _file() -> ReferenceFile:
    return ReferenceFile(
        reference_no=1,
        display_name="项目说明书.docx",
        locations=["第一章 / 段落 1-3"],
        evidence_nos=[1],
    )


def test_reference_models_accept_a_complete_reference_section() -> None:
    """The formatter output retains files, locations, evidence, and safe counts."""
    result = ReferenceBuildResult(
        section_text="参考文件：\n[1] 项目说明书.docx",
        files=[_file()],
        selected_hit_count=1,
        location_count=1,
    )

    assert result.files[0].evidence_nos == [1]
    assert result.location_count == 1


@pytest.mark.parametrize(
    "field, value",
    [
        ("reference_no", 0),
        ("display_name", ""),
        ("locations", [""]),
        ("evidence_nos", []),
        ("evidence_nos", [0]),
    ],
)
def test_reference_file_rejects_invalid_required_values(
    field: str,
    value: int | str | list[int] | list[str],
) -> None:
    """Displayed references require positive identifiers and non-empty text lists."""
    values = _file().model_dump()
    values[field] = value

    with pytest.raises(ValidationError):
        ReferenceFile(**values)


def test_reference_file_allows_a_locationless_source() -> None:
    """A PDF reference can be shown without inventing page or paragraph metadata."""
    values = _file().model_dump()
    values["locations"] = []

    assert ReferenceFile(**values).locations == []


def test_reference_models_forbid_unknown_fields() -> None:
    """Private source fields must not silently become public-reference fields."""
    with pytest.raises(ValidationError):
        ReferenceBuildResult(
            section_text="参考文件：\n[1] 项目说明书.docx",
            files=[_file()],
            selected_hit_count=1,
            location_count=1,
            private_relative_path="forbidden",
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("section_text", ""),
        ("files", []),
        ("selected_hit_count", 0),
        ("location_count", -1),
    ],
)
def test_reference_build_result_rejects_invalid_aggregate_values(
    field: str,
    value: int | str | list[ReferenceFile],
) -> None:
    """Reference result aggregates must remain meaningful for later logging."""
    values = {
        "section_text": "参考文件：\n[1] 项目说明书.docx",
        "files": [_file()],
        "selected_hit_count": 1,
        "location_count": 1,
    }
    values[field] = value

    with pytest.raises(ValidationError):
        ReferenceBuildResult(**values)
