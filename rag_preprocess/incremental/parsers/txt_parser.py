from __future__ import annotations
from pathlib import Path
import re
from .base import ParseFailure, ParsedBlockV2, ParsedDocumentV2, fallback_title, tidy, usable_title

def _decode(data: bytes) -> tuple[str,str]:
    choices=[]
    if data.startswith(b"\xef\xbb\xbf"): choices=[("utf-8-sig", data)]
    elif data.startswith(b"\xff\xfe"): choices=[("utf-16", data)]
    elif data.startswith(b"\xfe\xff"): choices=[("utf-16", data)]
    else:
        choices=[("utf-8",data),("gb18030",data)]
        if data.count(b"\x00") > len(data)//8: choices += [("utf-16-le",data),("utf-16-be",data)]
    for encoding, raw in choices:
        try: return raw.decode(encoding, errors="strict"), encoding
        except UnicodeDecodeError: continue
    try:
        from charset_normalizer import from_bytes
        match=from_bytes(data).best()
        if match and match.encoding and match.encoding.lower().replace("_","-") in {"utf-8","utf-16","utf-16-le","utf-16-be","gb18030","gbk"}: return str(match), match.encoding
    except Exception: pass
    raise ParseFailure("TXT_DECODE_FAILED")

def parse_txt_v2(path: Path, doc_id: str) -> ParsedDocumentV2:
    try: raw=path.read_bytes()
    except OSError: raise ParseFailure("TXT_READ_FAILED")
    # NUL is only legitimate for a plausibly UTF-16 byte stream.
    nul_ratio=raw.count(b"\x00") / max(1,len(raw))
    if .10 < nul_ratio < .40 and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise ParseFailure("TXT_BINARY_OR_GARBLED")
    try: text, encoding=_decode(raw)
    except ParseFailure: raise
    bad=sum(ch == "\ufffd" or (ord(ch)<32 and ch not in "\n\r\t") for ch in text)
    if not text.strip() or bad / max(1,len(text)) >= .1: raise ParseFailure("TXT_BINARY_OR_GARBLED")
    lines=text.replace("\r\n","\n").replace("\r","\n").split("\n"); blocks=[]; buf=[]; start=1
    def flush():
        nonlocal buf,start
        value=tidy(" ".join(buf))
        if value: blocks.append(ParsedBlockV2(len(blocks),"paragraph",value,paragraph_index=start,source_locator=f"paragraph:{start}"))
        buf=[]
    for index,line in enumerate(lines,1):
        if not line.strip(): flush(); continue
        if "\t" in line:
            flush(); cols=[tidy(x) for x in line.split("\t") if tidy(x)]
            if cols: blocks.append(ParsedBlockV2(len(blocks),"table_row","表格行："+" | ".join(cols),paragraph_index=index,source_locator=f"paragraph:{index}"))
            continue
        if not buf: start=index
        buf.append(line)
    flush()
    first=blocks[0].text if blocks else ""; title=fallback_title(path)
    if usable_title(first) and (len(lines)>1 and not lines[1].strip() or re.match(r"^第.{1,8}[章编]|.*(办法|规定|通知|制度)$",first)) : title=first
    return ParsedDocumentV2(doc_id,path.name,".txt",title,"txt",blocks,paragraph_count=sum(b.block_type=="paragraph" for b in blocks),table_row_count=sum(b.block_type=="table_row" for b in blocks),source_encoding=encoding)
