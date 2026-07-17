"""Offline tests for deterministic and privacy-safe RAG reference formatting."""

from __future__ import annotations

from copy import deepcopy

import pytest

from local_rag_app.context_models import SelectedContextHit
from local_rag_app.reference_formatter import ReferenceFormatError, ReferenceFormatter
from local_rag_app.retrieval_models import RetrievalHit


def _selected(
    evidence_no: int,
    *,
    doc_id: str = "doc-1",
    relative_path: str = "contracts/project.docx",
    title: str | None = "项目说明",
    doc_title: str | None = "项目说明书",
    section_path: str | None = "第一章",
    article_no: str | None = "第一条",
    article_range: str | None = None,
    paragraph_start: int | None = 1,
    paragraph_end: int | None = 3,
    extension: str | None = None,
) -> SelectedContextHit:
    hit = RetrievalHit(
        rank=evidence_no,
        chunk_id=f"chunk-{evidence_no}",
        doc_id=doc_id,
        chunk_text=f"PRIVATE-CHUNK-{evidence_no}",
        title=title,
        doc_title=doc_title,
        section_path=section_path,
        article_no=article_no,
        article_range=article_range,
        relative_path=relative_path,
        paragraph_start=paragraph_start,
        paragraph_end=paragraph_end,
        final_score=0.5,
        matched_by="vector",
        extension=extension,
    )
    return SelectedContextHit(
        evidence_no=evidence_no,
        hit=hit,
        text_for_prompt=hit.chunk_text,
        estimated_tokens=10,
    )


def test_formatter_groups_one_file_and_renders_the_contract() -> None:
    """One selected hit becomes one explicit file, location, and evidence mapping."""
    result = ReferenceFormatter().build([_selected(1)])

    assert result.selected_hit_count == 1
    assert result.location_count == 1
    assert result.files[0].display_name == "项目说明书（project.docx）"
    assert result.files[0].locations == ["第一章 / 第一条 / 段落 1-3"]
    assert result.files[0].evidence_nos == [1]
    assert result.section_text == (
        "参考文件：\n"
        "[1] 项目说明书（project.docx）\n"
        "    位置：第一章 / 第一条 / 段落 1-3\n"
        "    对应资料：[资料1]"
    )


def test_formatter_groups_same_normalized_path_and_deduplicates_locations() -> None:
    """Separator and case variations still identify the same Windows source file."""
    first = _selected(3, relative_path="Contracts\\Project.docx")
    second = _selected(
        1,
        relative_path="contracts/project.docx",
        section_path="第二章",
        paragraph_start=8,
        paragraph_end=9,
    )
    duplicate_location = _selected(2, relative_path="contracts/project.docx")

    result = ReferenceFormatter().build([first, second, duplicate_location])

    assert len(result.files) == 1
    assert result.files[0].reference_no == 1
    assert result.files[0].evidence_nos == [1, 2, 3]
    assert result.files[0].locations == [
        "第二章 / 第一条 / 段落 8-9",
        "第一章 / 第一条 / 段落 1-3",
    ]
    assert result.location_count == 2


def test_formatter_does_not_merge_same_basename_from_different_directories() -> None:
    """Display names may match, but full relative paths remain the grouping identity."""
    result = ReferenceFormatter().build(
        [
            _selected(1, relative_path="contract-a/说明.docx"),
            _selected(2, relative_path="contract-b/说明.docx"),
        ]
    )

    assert [item.reference_no for item in result.files] == [1, 2]
    assert [item.display_name for item in result.files] == ["项目说明书（说明.docx）", "项目说明书（说明.docx）"]


def test_formatter_prefers_article_range_and_handles_paragraph_variants() -> None:
    """Locations use explicit ranges before article numbers and never invent a page."""
    result = ReferenceFormatter().build(
        [
            _selected(
                1,
                article_no="第一条",
                article_range="第十条-第十二条",
                paragraph_start=12,
                paragraph_end=12,
            ),
            _selected(
                2,
                relative_path="other.docx",
                paragraph_start=None,
                paragraph_end=18,
            ),
        ]
    )

    assert result.files[0].locations == ["第一章 / 第十条-第十二条 / 段落 12"]
    assert result.files[1].locations == ["第一章 / 第一条 / 段落 18"]
    assert "页" not in result.section_text


def test_formatter_uses_title_or_explicit_missing_location_fallback() -> None:
    """Missing structured metadata is displayed honestly rather than guessed."""
    with_title = _selected(
        1,
        section_path=None,
        article_no=None,
        paragraph_start=None,
        paragraph_end=None,
        title="概览",
    )
    without_title = _selected(
        2,
        relative_path="missing.docx",
        section_path=None,
        article_no=None,
        paragraph_start=None,
        paragraph_end=None,
        title=None,
    )

    result = ReferenceFormatter().build([with_title, without_title])

    assert result.files[0].locations == ["标题：概览"]
    assert result.files[1].locations == ["未提供具体位置"]


def test_formatter_hides_paths_and_private_retrieval_fields() -> None:
    """Rendered references must never disclose local paths, scores, IDs, or chunk text."""
    selected = _selected(
        1,
        doc_id="PRIVATE-DOC-ID",
        relative_path="C:\\PRIVATE-PATH-SECRET\\private.docx",
        section_path="PRIVATE-SECTION-SECRET",
    )

    result = ReferenceFormatter().build([selected])

    assert result.files[0].display_name == "项目说明书（private.docx）"
    assert "C:\\PRIVATE-PATH-SECRET" not in result.section_text
    assert "PRIVATE-DOC-ID" not in result.section_text
    assert "PRIVATE-CHUNK-1" not in result.section_text
    assert "0.5" not in result.section_text
    assert "PRIVATE-SECTION-SECRET" in result.section_text


def test_formatter_hides_absolute_paths_from_untrusted_location_fields() -> None:
    """A malformed section value cannot disclose a Windows or server-side path."""
    result = ReferenceFormatter().build(
        [
            _selected(
                1,
                section_path="C:\\PRIVATE-SECTION-PATH\\internal",
            ),
            _selected(
                2,
                relative_path="server.docx",
                section_path="/projects/PRIVATE-SERVER-PATH/internal",
            ),
        ]
    )

    assert "PRIVATE-SECTION-PATH" not in result.section_text
    assert "PRIVATE-SERVER-PATH" not in result.section_text
    assert result.section_text.count("\\[已隐藏路径\\]") == 2


def test_formatter_escapes_markdown_html_controls_and_long_values() -> None:
    """Customer metadata remains plain text even when it contains renderer syntax."""
    selected = _selected(
        1,
        relative_path="[点击](网址).docx",
        section_path="<script>\nPRIVATE\t</script>|`_*#!",
    )
    long_location = _selected(
        2,
        relative_path="long.docx",
        section_path="甲" * 350,
        article_no=None,
        paragraph_start=None,
        paragraph_end=None,
    )

    result = ReferenceFormatter().build([selected, long_location])

    assert result.files[0].display_name.startswith("项目说明书（\\[点击\\]")
    assert "网址" in result.files[0].display_name
    assert "&lt;script&gt; PRIVATE &lt;/script&gt;\\|\\`\\_\\*\\#\\!" in (
        result.files[0].locations[0]
    )
    assert "\n" not in result.files[0].locations[0]
    assert len(result.files[1].locations[0]) <= 301
    assert result.files[1].locations[0].endswith("…")


def test_formatter_rejects_empty_or_duplicate_evidence_and_does_not_mutate_input() -> None:
    """Evidence numbering is a request-local contract and input models stay unchanged."""
    formatter = ReferenceFormatter()
    selected = [_selected(2), _selected(1, relative_path="other.docx")]
    before = deepcopy(selected)

    first = formatter.build(selected)
    second = formatter.build(selected)

    assert first == second
    assert selected == before
    with pytest.raises(ReferenceFormatError, match="required"):
        formatter.build([])
    with pytest.raises(ReferenceFormatError, match="unique"):
        formatter.build([_selected(1), _selected(1, relative_path="other.docx")])


def test_formatter_shows_title_and_filename_but_omits_pdf_location() -> None:
    """PDF references do not invent page or paragraph positions from chunk data."""
    result = ReferenceFormatter().build(
        [
            _selected(
                1,
                relative_path="reports/annual-report.pdf",
                doc_title="Annual Report",
                section_path="Chapter 4",
                paragraph_start=9,
                paragraph_end=12,
                extension=".pdf",
            ),
            _selected(
                2,
                relative_path="notes.txt",
                doc_title="Notes",
                section_path="Introduction",
                paragraph_start=2,
                paragraph_end=2,
                extension=".txt",
            ),
        ]
    )

    assert result.files[0].display_name == "Annual Report（annual-report.pdf）"
    assert result.files[0].locations == []
    assert result.files[1].locations == ["Introduction / 第一条 / 段落 2"]
    assert "位置：Chapter 4" not in result.section_text
    assert result.section_text.count("位置：") == 1
