"""测试 scanner 模块。"""

import tempfile
from pathlib import Path

from rag_preprocess.scanner import (
    is_word_file,
    compute_sha256_for_path,
    detect_volume,
    scan_rawdata,
    ScanSummary,
    format_summary,
)


# ── is_word_file ──────────────────────────────────────────────

def test_is_word_file_docx():
    assert is_word_file(Path("test.docx")) is True


def test_is_word_file_doc():
    assert is_word_file(Path("test.doc")) is True


def test_is_word_file_docm():
    assert is_word_file(Path("test.docm")) is True


def test_is_word_file_pdf():
    assert is_word_file(Path("test.pdf")) is False


def test_is_word_file_txt():
    assert is_word_file(Path("test.txt")) is False


def test_is_word_file_no_extension():
    assert is_word_file(Path("test")) is False


# ── compute_sha256_for_path ───────────────────────────────────

def test_compute_sha256_for_path_deterministic():
    h1 = compute_sha256_for_path("law-flk-vol1-main/docx/test.docx")
    h2 = compute_sha256_for_path("law-flk-vol1-main/docx/test.docx")
    assert h1 == h2
    assert len(h1) == 64


def test_compute_sha256_for_path_different():
    h1 = compute_sha256_for_path("path/a.docx")
    h2 = compute_sha256_for_path("path/b.docx")
    assert h1 != h2


# ── detect_volume ─────────────────────────────────────────────

def test_detect_volume_vol1():
    assert detect_volume("law-flk-vol1-main/docx/法律/test.docx") == "vol1"


def test_detect_volume_vol2():
    assert detect_volume("law-flk-vol2-main/docx/地方法规/test.docx") == "vol2"


def test_detect_volume_unknown():
    assert detect_volume("other-dir/some/file.docx") == "unknown"


# ── scan_rawdata ──────────────────────────────────────────────

def test_scan_rawdata_small(tmp_path: Path):
    """用临时目录验证扫描逻辑。"""
    rawdata = tmp_path / "Rawdata"
    vol1_dir = rawdata / "law-flk-vol1-main" / "docx" / "法律"
    vol2_dir = rawdata / "law-flk-vol2-main" / "docx" / "地方法规"
    vol1_dir.mkdir(parents=True)
    vol2_dir.mkdir(parents=True)

    # 创建一些 .docx 文件和一个非 Word 文件
    (vol1_dir / "法规A.docx").write_text("content A")
    (vol1_dir / "法规B.docx").write_text("content B")
    (vol2_dir / "法规C.doc").write_text("content C")
    (vol1_dir / "readme.txt").write_text("some text")

    files, summary = scan_rawdata(rawdata, compute_hash=True)

    assert summary.total_files == 4
    assert summary.word_files == 3
    assert summary.docx_count == 2
    assert summary.doc_count == 1
    assert summary.docm_count == 0
    assert summary.non_word_count == 1
    assert summary.vol1_count == 2
    assert summary.vol2_count == 1

    assert len(files) == 3
    for f in files:
        assert f.source_file_id is not None
        assert len(f.source_file_id) == 64
        assert f.file_hash_sha256 != ""
        assert f.is_word_file is True


def test_scan_rawdata_no_hash(tmp_path: Path):
    """验证 compute_hash=False 跳过 hash 计算。"""
    rawdata = tmp_path / "Rawdata"
    vol1_dir = rawdata / "law-flk-vol1-main" / "docx"
    vol1_dir.mkdir(parents=True)
    (vol1_dir / "test.docx").write_text("test")

    files, _ = scan_rawdata(rawdata, compute_hash=False)
    assert len(files) == 1
    assert files[0].file_hash_sha256 == ""


# ── format_summary ────────────────────────────────────────────

def test_format_summary():
    s = ScanSummary(
        total_files=10, word_files=8, docx_count=5, doc_count=2,
        docm_count=1, total_bytes=1024, vol1_count=3, vol2_count=5,
        non_word_count=2,
    )
    text = format_summary(s)
    assert "总文件数" in text
    assert "8" in text
    assert "5" in text
