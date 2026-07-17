"""Fixed OCR quality rules from the implementation plan."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class OcrPageQuality:
    average_confidence: float; low_line_ratio: float; valid_chars: int; garbled_ratio: float; status: str

def assess_page(lines: list[tuple[str,float]], *, blank: bool=False) -> OcrPageQuality:
    if blank: return OcrPageQuality(1.0,0.0,0,0.0,"ok")
    valid="".join(text for text,_ in lines); count=sum(not c.isspace() for c in valid)
    average=sum(score for _,score in lines)/max(1,len(lines)); low=sum(score<.80 for _,score in lines)/max(1,len(lines))
    garbled=sum(c=="\ufffd" for c in valid)/max(1,len(valid))
    severe=average<.60 or count<20 or garbled>=.10
    return OcrPageQuality(average,low,count,garbled,"severe" if severe else ("warning" if average<.85 or low>=.20 else "ok"))

def document_failed(qualities: list[OcrPageQuality]) -> bool:
    severe=[q.status=="severe" for q in qualities]; total=sum(severe)
    consecutive=0; maximum=0
    for item in severe: consecutive=consecutive+1 if item else 0; maximum=max(maximum,consecutive)
    return total/max(1,len(qualities))>=.20 or maximum>=5 or total>=10
