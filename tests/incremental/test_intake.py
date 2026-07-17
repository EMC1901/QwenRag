from pathlib import Path
from rag_preprocess.incremental.intake import FrozenFile, classify, required_free_bytes, scan_and_freeze, verify_ready

def test_freeze_is_single_level_and_later_file_is_not_in_manifest(tmp_path: Path):
    incoming=tmp_path/'incoming'; incoming.mkdir(); (incoming/'a.txt').write_text('甲'); (incoming/'nested').mkdir(); (incoming/'bad.exe').write_bytes(b'x')
    rows=scan_and_freeze(incoming,tmp_path/'files.json'); (incoming/'later.pdf').write_bytes(b'%PDF-')
    assert {r.file_name for r in rows} == {'a.txt','nested','bad.exe'}
    assert next(r for r in rows if r.file_name=='bad.exe').state=='UNSUPPORTED'

def test_large_file_ready_and_classification(tmp_path: Path):
    source=tmp_path/'a.txt'; source.write_bytes(b'x'*(30*1024*1024+1)); row=FrozenFile('a.txt','a.txt',str(source),0,0,extension='.txt')
    verify_ready(row,probes=1); classify([row], {'a.txt':('d1',row.sha256)})
    assert row.state=='FROZEN' and row.action=='DUPLICATE_UNCHANGED'

def test_hash_match_under_different_name_warns(tmp_path: Path):
    path=tmp_path/'renamed.txt'; path.write_text('same'); row=FrozenFile('renamed.txt','renamed.txt',str(path),4,0,extension='.txt'); verify_ready(row,probes=1)
    classify([row], {'old.txt':('old','%s'%row.sha256)})
    assert row.action=='NEW' and row.warning_codes==['DUPLICATE_CONTENT_DIFFERENT_NAME']

def test_disk_formula_has_safety_margin(): assert required_free_bytes([100,200],100,50)==780
