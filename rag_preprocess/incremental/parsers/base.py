from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re, unicodedata

class ParseFailure(RuntimeError):
    def __init__(self, code: str): super().__init__(code); self.code=code

@dataclass(frozen=True)
class ParsedBlockV2:
    block_index: int; block_type: str; text: str; paragraph_index: int | None = None
    table_index: int | None = None; row_index: int | None = None; page_number: int | None = None
    style_name: str | None = None; source_locator: str | None = None; ocr_confidence: float | None = None
    quality_status: str = "ok"

@dataclass(frozen=True)
class ParseWarning:
    code: str; message: str; page_number: int | None = None; metrics: dict[str, int | float | str] = field(default_factory=dict)

@dataclass
class ParsedDocumentV2:
    doc_id: str; file_name: str; extension: str; title: str; parse_method: str; blocks: list[ParsedBlockV2]
    warnings: list[ParseWarning] = field(default_factory=list); page_count: int | None = None
    paragraph_count: int = 0; table_row_count: int = 0; source_encoding: str | None = None
    ocr_pages: list[int] = field(default_factory=list)

def tidy(text: str) -> str:
    return re.sub(r"[ \t]+", " ", unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")).strip()

def fallback_title(path: Path) -> str: return path.stem

def usable_title(text: str) -> bool:
    value=tidy(text); return 2 <= len(value) <= 100 and not value.endswith(("。", ";", "；", ".")) and value not in {"目录"}
