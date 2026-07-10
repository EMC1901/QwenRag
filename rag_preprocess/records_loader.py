"""读取两个 records.json，建立文件到元数据的映射。"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LawRecord:
    """单条法规记录。"""

    record_id: str
    volume: str
    bbbs: str | None = None
    title: str | None = None
    gbrq: str | None = None
    sxrq: str | None = None
    sxx: str | None = None
    zdjg_name: str | None = None
    flxz: str | None = None
    zdjg_code_id: str | None = None
    flfg_code_id: str | None = None
    my_dir: str | None = None
    my_file: str | None = None
    my_status: str | None = None
    my_time: str | None = None
    expected_relative_path: str | None = None
    matched_source_file_id: str | None = None


@dataclass
class MatchResult:
    """records 与文件匹配结果。"""

    records_total: int = 0
    matched: int = 0
    records_without_file: int = 0
    files_without_record: int = 0
    # 缺字段统计
    missing_title: int = 0
    missing_gbrq: int = 0
    missing_sxrq: int = 0
    missing_sxx: int = 0
    missing_my_file: int = 0
    missing_bbbs: int = 0


# volume 标签 → 一级目录映射
VOLUME_DIR_MAP = {
    "vol1": "law-flk-vol1-main",
    "vol2": "law-flk-vol2-main",
}


def load_records(records_path: Path, volume: str) -> list[LawRecord]:
    """读取一个 records.json。

    records.json 结构: {"data": [...], "stats": {...}, "update": "...", ...}
    """
    with open(records_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw.get("data", []) if isinstance(raw, dict) else raw

    records: list[LawRecord] = []
    for item in items:
        bbbs = item.get("bbbs")

        # sxx 在 JSON 中可能是整数，统一转为字符串
        sxx = item.get("sxx")
        sxx_str = str(sxx) if sxx is not None else None

        # my_status 在 JSON 中可能是整数
        my_status = item.get("my_status")
        my_status_str = str(my_status) if my_status is not None else None

        # 生成 expected_relative_path
        my_dir = item.get("my_dir")
        my_file = item.get("my_file")
        expected_path = build_expected_relative_path(my_dir, my_file, volume)

        record = LawRecord(
            record_id=bbbs if bbbs else _fallback_record_id(item),
            volume=volume,
            bbbs=bbbs,
            title=item.get("title"),
            gbrq=item.get("gbrq"),
            sxrq=item.get("sxrq"),
            sxx=sxx_str,
            zdjg_name=item.get("zdjgName"),
            flxz=item.get("flxz"),
            zdjg_code_id=item.get("zdjgCodeId"),
            flfg_code_id=item.get("flfgCodeId"),
            my_dir=my_dir,
            my_file=my_file,
            my_status=my_status_str,
            my_time=item.get("my_time"),
            expected_relative_path=expected_path,
            matched_source_file_id=None,
        )
        records.append(record)

    return records


def build_expected_relative_path(
    my_dir: str | None,
    my_file: str | None,
    volume: str,
) -> str | None:
    """根据 my_dir + my_file 生成预期文件路径。

    返回例如: law-flk-vol1-main/docx/宪法/file.docx
    """
    if not my_file:
        return None

    vol_dir = VOLUME_DIR_MAP.get(volume, "")

    parts = []
    if vol_dir:
        parts.append(vol_dir)
    parts.append("docx")
    if my_dir:
        # my_dir 使用 / 分隔，兼容 Windows 的 \
        for seg in my_dir.replace("\\", "/").split("/"):
            seg = seg.strip()
            if seg:
                parts.append(seg)
    parts.append(my_file)

    return Path(*parts).as_posix()


def match_records_to_files(
    records: list[LawRecord],
    source_file_rel_paths: set[str],
) -> MatchResult:
    """把 records 和文件清单做双向匹配。

    返回 MatchResult 包含匹配统计和缺字段统计。
    """
    result = MatchResult()
    result.records_total = len(records)

    matched_paths: set[str] = set()

    for r in records:
        # 统计缺字段
        if not r.title:
            result.missing_title += 1
        if not r.gbrq:
            result.missing_gbrq += 1
        if not r.sxrq:
            result.missing_sxrq += 1
        if not r.sxx:
            result.missing_sxx += 1
        if not r.my_file:
            result.missing_my_file += 1
        if not r.bbbs:
            result.missing_bbbs += 1

        # 匹配
        if r.expected_relative_path and r.expected_relative_path in source_file_rel_paths:
            result.matched += 1
            matched_paths.add(r.expected_relative_path)
        else:
            result.records_without_file += 1

    # 找出无 record 的 Word 文件
    for fp in source_file_rel_paths:
        ext = Path(fp).suffix.lower()
        if ext in {".docx", ".doc", ".docm"} and fp not in matched_paths:
            result.files_without_record += 1

    return result


def _fallback_record_id(item: dict) -> str:
    """当 bbbs 缺失时，用 title+my_file 生成备用 ID。"""
    import hashlib
    raw = f"{item.get('title', '')}:{item.get('my_file', '')}"
    return "fallback_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
