from __future__ import annotations
from pathlib import Path
import zipfile
from rag_preprocess.docx_parser import parse_docx
from .base import ParseFailure, ParsedBlockV2, ParsedDocumentV2, fallback_title, tidy, usable_title

def parse_docx_v2(path: Path, doc_id: str) -> ParsedDocumentV2:
    if path.suffix.lower() != ".docx": raise ParseFailure("DOCX_EXTENSION_INVALID")
    try:
        with zipfile.ZipFile(path) as archive:
            if "[Content_Types].xml" not in archive.namelist(): raise ParseFailure("DOCX_INVALID_CONTAINER")
    except ParseFailure: raise
    except (OSError, zipfile.BadZipFile): raise ParseFailure("DOCX_INVALID_CONTAINER")
    parsed=parse_docx(path, doc_id)
    if parsed.parse_error: raise ParseFailure("DOCX_PARSE_FAILED")
    blocks=[]; title=None
    for old in parsed.blocks:
        text=tidy(old.text)
        if not text: continue
        style=old.style_name
        if title is None and usable_title(text) and style and style.casefold() in {"title", "heading 1", "标题", "标题 1"}: title=text
        blocks.append(ParsedBlockV2(len(blocks), old.block_type, text, paragraph_index=(old.paragraph_index + 1 if old.paragraph_index is not None else None), table_index=old.table_index, row_index=old.row_index, style_name=style, source_locator=(f"paragraph:{old.paragraph_index + 1}" if old.paragraph_index is not None else f"table:{old.table_index + 1}:row:{old.row_index + 1}")))
    if title is None:
        for block in blocks[:5]:
            if block.block_type == "paragraph" and usable_title(block.text) and len(block.text) < 60: title=block.text; break
    return ParsedDocumentV2(doc_id, path.name, ".docx", title or fallback_title(path), "docx", blocks, paragraph_count=parsed.paragraph_count, table_row_count=sum(b.block_type=="table_row" for b in blocks))
