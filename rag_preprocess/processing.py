"""Stage 9: versioned, internal clean/structure/chunk hand-off."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from rag_preprocess.chunker import ChunkConfig, StructuredDocument, build_chunks
from rag_preprocess.law_structure import StructuredBlock, detect_and_annotate_blocks
from rag_preprocess.text_cleaner import clean_text
from .incremental.parsers.base import ParsedDocumentV2
from .incremental.persistence import read_checkpoint, write_checkpoint

INTERMEDIATE_SCHEMA_VERSION=1
@dataclass
class ProcessingResult:
    chunks: list; rejected_blocks: int=0

def process_document(document: ParsedDocumentV2, config: ChunkConfig | None=None) -> ProcessingResult:
    cleaned=[]; rejected=0
    for block in document.blocks:
        text=clean_text(block.text)
        if not text.strip(): rejected+=1; continue
        cleaned.append(StructuredBlock(block_id=f"{document.doc_id}:{block.block_index}",text=text,block_index=len(cleaned),block_type=block.block_type))
    structured=detect_and_annotate_blocks(cleaned)
    chunks=build_chunks(StructuredDocument(doc_id=document.doc_id,title=document.title,blocks=structured),config)
    for chunk in chunks:
        source=[b for b in document.blocks if b.block_index <= chunk.chunk_index]
        loc=[b.paragraph_index for b in source if b.paragraph_index]
        if loc: chunk.paragraph_start,chunk.paragraph_end=min(loc),max(loc)
    return ProcessingResult(chunks,rejected)

def write_intermediate(path: Path, document: ParsedDocumentV2) -> None:
    write_checkpoint(path,{"schema_version":INTERMEDIATE_SCHEMA_VERSION,"document":asdict(document)})

def intermediate_is_current(path: Path) -> bool:
    try: return read_checkpoint(path).get("schema_version")==INTERMEDIATE_SCHEMA_VERSION
    except (OSError,ValueError): return False
