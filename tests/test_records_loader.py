"""测试 records_loader 模块。"""

import json
import tempfile
from pathlib import Path

from rag_preprocess.records_loader import (
    build_expected_relative_path,
    load_records,
    match_records_to_files,
    LawRecord,
)


# ── build_expected_relative_path ──────────────────────────────

def test_build_expected_relative_path_vol1():
    result = build_expected_relative_path("法律/宪法", "test.docx", "vol1")
    assert result == "law-flk-vol1-main/docx/法律/宪法/test.docx"


def test_build_expected_relative_path_vol2():
    result = build_expected_relative_path("地方法规/广东", "test.docx", "vol2")
    assert result == "law-flk-vol2-main/docx/地方法规/广东/test.docx"


def test_build_expected_relative_path_no_my_file():
    result = build_expected_relative_path("some/dir", None, "vol1")
    assert result is None


def test_build_expected_relative_path_no_my_dir():
    result = build_expected_relative_path(None, "test.docx", "vol1")
    assert result == "law-flk-vol1-main/docx/test.docx"


def test_build_expected_relative_path_unknown_volume():
    result = build_expected_relative_path("some", "test.docx", "vol3")
    assert result == "docx/some/test.docx"


def test_build_expected_relative_path_backslash_in_my_dir():
    result = build_expected_relative_path("法律\\宪法", "test.docx", "vol1")
    assert result == "law-flk-vol1-main/docx/法律/宪法/test.docx"


# ── load_records ──────────────────────────────────────────────

def test_load_records_from_temp_file(tmp_path: Path):
    """用临时 JSON 文件验证 load_records。"""
    data = {
        "data": [
            {
                "bbbs": "abc123",
                "title": "测试法规",
                "gbrq": "2020-01-01",
                "sxrq": "2020-06-01",
                "sxx": 3,
                "zdjgName": "测试机关",
                "flxz": "行政法规",
                "zdjgCodeId": 100,
                "flfgCodeId": 200,
                "my_dir": "法律/宪法",
                "my_file": "test.docx",
                "my_status": 404,
                "my_time": "20250101T12+0800",
            },
            {
                "bbbs": "def456",
                "title": "另一法规",
                "gbrq": None,
                "sxrq": None,
                "sxx": None,
                "my_dir": "地方法规",
                "my_file": None,
            },
        ],
        "stats": {"total": 2},
    }
    json_path = tmp_path / "records.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    records = load_records(json_path, "vol1")

    assert len(records) == 2

    r1 = records[0]
    assert r1.record_id == "abc123"
    assert r1.title == "测试法规"
    assert r1.sxx == "3"
    assert r1.my_status == "404"
    assert r1.expected_relative_path == "law-flk-vol1-main/docx/法律/宪法/test.docx"

    r2 = records[1]
    assert r2.record_id == "def456"
    assert r2.title == "另一法规"
    assert r2.gbrq is None
    assert r2.expected_relative_path is None


# ── match_records_to_files ────────────────────────────────────

def test_match_records_to_files():
    records = [
        LawRecord(
            record_id="r1", volume="vol1", bbbs="r1",
            title="法规A", gbrq="2020-01-01", sxrq="2020-06-01",
            sxx="3", my_file="a.docx",
            expected_relative_path="vol1/docx/a.docx",
        ),
        LawRecord(
            record_id="r2", volume="vol1", title=None, gbrq=None,
            sxrq=None, sxx=None, my_file=None, bbbs=None,
            expected_relative_path=None,
        ),
        LawRecord(
            record_id="r3", volume="vol2", bbbs="r3",
            title="法规C", gbrq="2021-01-01", sxrq="2021-06-01",
            sxx="3", my_file="c.docx",
            expected_relative_path="vol2/docx/c.docx",
        ),
    ]
    source_paths = {"vol1/docx/a.docx", "vol2/docx/d.docx"}

    result = match_records_to_files(records, source_paths)

    assert result.records_total == 3
    assert result.matched == 1  # 只有 r1 匹配
    assert result.records_without_file == 2
    assert result.files_without_record == 1  # vol2/docx/d.docx 无 record

    # 缺字段统计 — 只有 r2 缺字段
    assert result.missing_title == 1
    assert result.missing_gbrq == 1
    assert result.missing_sxrq == 1
    assert result.missing_sxx == 1
    assert result.missing_my_file == 1
    assert result.missing_bbbs == 1


def test_match_all_matched():
    records = [
        LawRecord(
            record_id="r1", volume="vol1", title="X",
            expected_relative_path="path/a.docx",
        ),
    ]
    source_paths = {"path/a.docx"}

    result = match_records_to_files(records, source_paths)
    assert result.matched == 1
    assert result.records_without_file == 0
    assert result.files_without_record == 0
