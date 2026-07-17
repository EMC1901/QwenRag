"""Pure construction of safe, deterministic reference sections for RAG answers."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
import re
import unicodedata
from typing import Iterable

from local_rag_app.context_models import SelectedContextHit
from local_rag_app.reference_models import ReferenceBuildResult, ReferenceFile
from local_rag_app.retrieval_models import RetrievalHit


_MAX_DISPLAY_FIELD_CHARS = 300
_MARKDOWN_SPECIALS = frozenset("\\[]()*_`#!|")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\)[^\s]*")
_UNIX_ABSOLUTE_PATH = re.compile(
    r"(?<!\S)/(?:projects|home|var|tmp|sevenH|opt|usr|etc)(?:/[^\s]*)?"
)


class ReferenceFormatError(ValueError):
    """Raised when selected-hit metadata cannot form a trustworthy reference list."""


@dataclass
class _ReferenceGroup:
    """Mutable local aggregation state that never leaves one formatter call."""

    display_name: str
    evidence_nos: list[int] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)


class ReferenceFormatter:
    """Convert actual prompt evidence into a safe user-visible source section."""

    def build(
        self,
        selected_hits: list[SelectedContextHit],
    ) -> ReferenceBuildResult:
        """Group selected prompt hits by file without exposing private metadata."""
        if not selected_hits:
            raise ReferenceFormatError("Selected context hits are required")

        groups: dict[str, _ReferenceGroup] = {}
        seen_evidence_nos: set[int] = set()
        ordered_hits = sorted(selected_hits, key=lambda item: item.evidence_no)

        for selected in ordered_hits:
            if selected.evidence_no in seen_evidence_nos:
                raise ReferenceFormatError("Selected context evidence numbers must be unique")
            seen_evidence_nos.add(selected.evidence_no)

            hit = selected.hit
            source_key = self._source_key(hit)
            group = groups.get(source_key)
            if group is None:
                group = _ReferenceGroup(display_name=self._display_name(hit))
                groups[source_key] = group

            self._append_unique(group.evidence_nos, selected.evidence_no)
            location = self._location(hit)
            if location:
                self._append_unique(group.locations, location)

        files = [
            ReferenceFile(
                reference_no=index,
                display_name=group.display_name,
                locations=group.locations,
                evidence_nos=group.evidence_nos,
            )
            for index, group in enumerate(groups.values(), start=1)
        ]
        if not files:
            raise ReferenceFormatError("No reference files could be prepared")

        return ReferenceBuildResult(
            section_text=self._render_section(files),
            files=files,
            selected_hit_count=len(ordered_hits),
            location_count=sum(len(item.locations) for item in files),
        )

    @staticmethod
    def _append_unique(values: list[int] | list[str], value: int | str) -> None:
        if value not in values:
            values.append(value)

    def _source_key(self, hit: RetrievalHit) -> str:
        normalized_path = self._normalize_relative_path(hit.relative_path)
        if normalized_path:
            return f"path:{normalized_path.casefold()}"
        normalized_doc_id = self._normalize_text(hit.doc_id)
        if normalized_doc_id:
            return f"doc:{normalized_doc_id.casefold()}"
        raise ReferenceFormatError("A selected source requires a stable identity")

    @staticmethod
    def _normalize_relative_path(value: str) -> str:
        parts = [
            part
            for part in value.replace("\\", "/").split("/")
            if part and part != "."
        ]
        return "/".join(parts)

    def _display_name(self, hit: RetrievalHit) -> str:
        basename = self._safe_basename(hit.relative_path)
        filename = self._display_text(basename)
        title = hit.doc_title or hit.title
        title_display = self._display_text(title)
        if filename and title_display:
            stem = basename.rsplit(".", 1)[0] if "." in basename else basename
            if self._normalize_text(title).casefold() != self._normalize_text(stem).casefold():
                return f"{title_display}（{filename}）"
            return filename
        if filename:
            return filename
        if title_display:
            return title_display
        for value in (basename, hit.doc_title, hit.title, "未命名资料"):
            display = self._display_text(value)
            if display:
                return display
        raise ReferenceFormatError("A selected source requires a display name")

    @staticmethod
    def _safe_basename(relative_path: str) -> str:
        parts = [
            part
            for part in relative_path.replace("\\", "/").split("/")
            if part and part not in {".", ".."}
        ]
        return parts[-1] if parts else ""

    def _location(self, hit: RetrievalHit) -> str:
        extension = (hit.extension or self._safe_basename(hit.relative_path).rsplit(".", 1)[-1]).lower()
        if extension in {".pdf", "pdf"}:
            return ""
        parts: list[str] = []
        for value in (hit.section_path, hit.article_range or hit.article_no):
            normalized = self._normalize_text(value)
            if normalized:
                parts.append(normalized)

        paragraph = self._format_paragraph_range(
            hit.paragraph_start,
            hit.paragraph_end,
        )
        if paragraph:
            parts.append(paragraph)

        if not parts:
            title = self._normalize_text(hit.title)
            if title:
                parts.append(f"标题：{title}")
        if not parts:
            parts.append("未提供具体位置")
        return self._display_text(" / ".join(parts))

    @staticmethod
    def _format_paragraph_range(start: int | None, end: int | None) -> str:
        if start is None and end is None:
            return ""
        if start is None:
            return f"段落 {end}"
        if end is None or start == end:
            return f"段落 {start}"
        return f"段落 {start}-{end}"

    def _display_text(self, value: str | None) -> str:
        normalized = self._normalize_text(value)
        if not normalized:
            return ""
        return self._escape_markdown(self._truncate(normalized))

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        if value is None:
            return ""
        normalized = unicodedata.normalize("NFC", str(value))
        without_controls = "".join(
            " " if unicodedata.category(char).startswith("C") else char
            for char in normalized
        )
        collapsed = " ".join(without_controls.split())
        without_windows_paths = _WINDOWS_ABSOLUTE_PATH.sub("[已隐藏路径]", collapsed)
        return _UNIX_ABSOLUTE_PATH.sub("[已隐藏路径]", without_windows_paths)

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= _MAX_DISPLAY_FIELD_CHARS:
            return value
        return value[: _MAX_DISPLAY_FIELD_CHARS - 1] + "…"

    @staticmethod
    def _escape_markdown(value: str) -> str:
        escaped_html = html.escape(value, quote=False)
        return "".join(
            f"\\{char}" if char in _MARKDOWN_SPECIALS else char
            for char in escaped_html
        )

    @staticmethod
    def _render_section(files: Iterable[ReferenceFile]) -> str:
        lines = ["参考文件："]
        for item in files:
            evidence = "、".join(f"[资料{number}]" for number in item.evidence_nos)
            lines.extend(
                [
                    f"[{item.reference_no}] {item.display_name}",
                    f"    位置：{'；'.join(item.locations)}",
                    f"    对应资料：{evidence}",
                    "",
                ]
            )
        return "\n".join(lines).rstrip()

    @staticmethod
    def _render_section(files: Iterable[ReferenceFile]) -> str:
        """Render source locations only when the source format supports them."""
        lines = ["参考文件："]
        for item in files:
            evidence = "、".join(f"[资料{number}]" for number in item.evidence_nos)
            lines.append(f"[{item.reference_no}] {item.display_name}")
            if item.locations:
                lines.append(f"    位置：{'；'.join(item.locations)}")
            lines.extend([f"    对应资料：{evidence}", ""])
        return "\n".join(lines).rstrip()
