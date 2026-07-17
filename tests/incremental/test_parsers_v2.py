from pathlib import Path
import pytest
from rag_preprocess.incremental.parsers.base import ParseFailure
from rag_preprocess.incremental.parsers.txt_parser import parse_txt_v2

def test_txt_utf8_table_title_and_one_based_locations(tmp_path: Path):
    file=tmp_path/'fallback.txt'; file.write_text('管理办法\n\n第一条 内容\n甲\t乙\n',encoding='utf-8')
    document=parse_txt_v2(file,'d')
    assert document.title=='管理办法' and document.blocks[0].paragraph_index==1
    assert document.blocks[-1].block_type=='table_row' and document.source_encoding=='utf-8'

def test_txt_strict_decode_rejects_binary(tmp_path: Path):
    file=tmp_path/'bad.txt'; file.write_bytes(b'\x81\x00\xff'*100)
    with pytest.raises(ParseFailure): parse_txt_v2(file,'d')

def test_docx_adapter_preserves_order_and_uses_one_based(tmp_path: Path):
    docx=pytest.importorskip('docx'); from rag_preprocess.incremental.parsers.docx_adapter import parse_docx_v2
    file=tmp_path/'a.docx'; d=docx.Document(); d.add_paragraph('标题').style='Title'; d.add_table(rows=1,cols=2).rows[0].cells[0].text='甲'; d.tables[0].rows[0].cells[1].text='乙'; d.add_paragraph('正文'); d.save(file)
    document=parse_docx_v2(file,'d')
    assert [b.block_type for b in document.blocks]==['paragraph','table_row','paragraph'] and document.blocks[0].paragraph_index==1
