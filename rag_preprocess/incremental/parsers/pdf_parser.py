from __future__ import annotations
from pathlib import Path
from .base import ParseFailure, ParsedBlockV2, ParsedDocumentV2, ParseWarning, fallback_title, tidy, usable_title

def _page_blocks(page):
    items=[]
    for x0,y0,x1,y1,text,*_ in page.get_text("blocks"):
        value=tidy(text)
        if value: items.append((x0,y0,x1,y1,value))
    return sorted(items,key=lambda b:(round(b[1]/12),b[0],b[1]))

def parse_pdf_electronic(path: Path, doc_id: str, *, checkpoint_path: Path | None=None) -> ParsedDocumentV2:
    if not path.read_bytes()[:5] == b"%PDF-": raise ParseFailure("PDF_HEADER_INVALID")
    try: import fitz; document=fitz.open(path)
    except Exception: raise ParseFailure("PDF_OPEN_FAILED")
    try:
        if document.needs_pass: raise ParseFailure("PDF_PASSWORD_REQUIRED")
        blocks=[]; warnings=[]; ocr=[]; title=None
        for number,page in enumerate(document,1):
            page_blocks=_page_blocks(page); valid=sum(len(x[4]) for x in page_blocks)
            images=page.get_images(full=False)
            if not page_blocks and not images: continue
            if valid<20 and images: ocr.append(number); warnings.append(ParseWarning("PDF_OCR_REQUIRED","页面需要 OCR",number)); continue
            for *_,text in page_blocks:
                if title is None and usable_title(text) and len(text)<80: title=text
                blocks.append(ParsedBlockV2(len(blocks),"paragraph",text,page_number=number,source_locator=f"pdf-page:{number}"))
            if checkpoint_path:
                from ..persistence import write_checkpoint
                write_checkpoint(checkpoint_path,{"last_completed_page":number})
        return ParsedDocumentV2(doc_id,path.name,".pdf",title or fallback_title(path),"pdf-electronic",blocks,warnings,page_count=document.page_count,paragraph_count=len(blocks),ocr_pages=ocr)
    except ParseFailure: raise
    except Exception: raise ParseFailure("PDF_PARSE_FAILED")
    finally: document.close()

def parse_pdf(path: Path, doc_id: str, *, ocr=None, work_dir: Path | None=None, checkpoint_path: Path | None=None) -> ParsedDocumentV2:
    """Parse electronic pages immediately and OCR only pages the detector marked.

    OCR bitmaps are private work artefacts; callers should remove their task work
    directory under the existing retention policy after publishing.
    """
    result=parse_pdf_electronic(path,doc_id,checkpoint_path=checkpoint_path)
    if not result.ocr_pages or ocr is None: return result
    try: import fitz; document=fitz.open(path)
    except Exception: raise ParseFailure("PDF_OPEN_FAILED")
    try:
        image_dir=(work_dir or path.parent)/"ocr-pages"; image_dir.mkdir(parents=True,exist_ok=True)
        from .ocr_quality import assess_page, document_failed
        qualities=[]
        for number in result.ocr_pages:
            page=document[number-1]; image_path=image_dir/f"page-{number}.png"
            page.get_pixmap(dpi=200, alpha=False).save(image_path)
            lines=ocr.recognize(image_path)
            lines=sorted(lines,key=lambda row:(round(row[0][1]/12),row[0][0]))
            quality=assess_page([(text,score) for _,text,score in lines]); qualities.append(quality)
            if quality.status != "ok": result.warnings.append(ParseWarning("OCR_PAGE_QUALITY_"+quality.status.upper(),"OCR 页面质量异常",number,{"average_confidence":quality.average_confidence,"low_line_ratio":quality.low_line_ratio}))
            for _,text,score in lines:
                value=tidy(text)
                if value: result.blocks.append(ParsedBlockV2(len(result.blocks),"paragraph",value,page_number=number,source_locator=f"pdf-page:{number}",ocr_confidence=score,quality_status=quality.status))
            if checkpoint_path:
                from ..persistence import write_checkpoint
                write_checkpoint(checkpoint_path,{"last_completed_page":number})
        if document_failed(qualities): raise ParseFailure("OCR_DOCUMENT_QUALITY_FAILED")
        result.parse_method="pdf-mixed-ocr"; result.paragraph_count=len(result.blocks)
        return result
    finally: document.close()
