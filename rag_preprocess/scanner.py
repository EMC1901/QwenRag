"""扫描 Rawdata，生成文件清单。"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


@dataclass
class SourceFile:
    """单个源文件信息。"""

    source_file_id: str
    volume: str
    relative_path: str
    file_name: str
    extension: str
    file_size: int
    file_hash_sha256: str
    mtime: str | None
    path_length: int
    is_word_file: bool


@dataclass
class ScanSummary:
    """扫描汇总统计。"""

    total_files: int = 0
    word_files: int = 0
    docx_count: int = 0
    doc_count: int = 0
    docm_count: int = 0
    total_bytes: int = 0
    vol1_count: int = 0
    vol2_count: int = 0
    non_word_count: int = 0


WORD_EXTENSIONS = {".docx", ".doc", ".docm"}

# 第一级目录 → volume 标签映射
VOLUME_DIRS = {
    "law-flk-vol1-main": "vol1",
    "law-flk-vol2-main": "vol2",
}


def is_word_file(path: Path) -> bool:
    """判断是否为 .docx/.doc/.docm。"""
    return path.suffix.lower() in WORD_EXTENSIONS


def compute_sha256(path: Path) -> str:
    """计算文件内容 sha256。"""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_sha256_for_path(rel_path: str) -> str:
    """用相对路径的 SHA256 生成 source_file_id。"""
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()


def detect_volume(relative_path: str) -> str:
    """根据相对路径的第一级目录判断属于 vol1 还是 vol2。"""
    first_part = relative_path.split("/")[0]
    return VOLUME_DIRS.get(first_part, "unknown")


def scan_rawdata(
    rawdata_dir: Path,
    compute_hash: bool = True,
) -> tuple[list[SourceFile], ScanSummary]:
    """扫描 Rawdata，返回 Word 文件列表及汇总统计。

    注意：只返回 Word 类文件（.docx/.doc/.docm），其他文件仅计入统计。
    """
    rawdata_dir = rawdata_dir.resolve()
    all_count = 0
    word_paths: list[Path] = []
    summary = ScanSummary()

    # 第一遍：分类统计
    for file_path in rawdata_dir.rglob("*"):
        if not file_path.is_file():
            continue
        all_count += 1

        if is_word_file(file_path):
            word_paths.append(file_path)
        else:
            summary.non_word_count += 1

    summary.total_files = all_count
    summary.word_files = len(word_paths)

    # 第二遍：生成 SourceFile 列表 + 计算 SHA256
    word_iter = word_paths
    if compute_hash and word_paths:
        word_iter = tqdm(word_paths, desc="计算文件 SHA256", unit="file")

    results: list[SourceFile] = []
    for fp in word_iter:
        rel_path = fp.relative_to(rawdata_dir).as_posix()
        ext = fp.suffix.lower()
        size = fp.stat().st_size

        # 按类型和 volume 统计
        if ext == ".docx":
            summary.docx_count += 1
        elif ext == ".doc":
            summary.doc_count += 1
        elif ext == ".docm":
            summary.docm_count += 1

        vol = detect_volume(rel_path)
        if vol == "vol1":
            summary.vol1_count += 1
        elif vol == "vol2":
            summary.vol2_count += 1

        summary.total_bytes += size

        sf = SourceFile(
            source_file_id=compute_sha256_for_path(rel_path),
            volume=vol,
            relative_path=rel_path,
            file_name=fp.name,
            extension=ext,
            file_size=size,
            file_hash_sha256=compute_sha256(fp) if compute_hash else "",
            mtime=datetime.fromtimestamp(fp.stat().st_mtime).isoformat(),
            path_length=len(rel_path),
            is_word_file=True,
        )
        results.append(sf)

    return results, summary


def format_summary(summary: ScanSummary) -> str:
    """格式化扫描汇总为可打印字符串。"""
    from rag_preprocess.utils import format_bytes

    lines = [
        "=" * 50,
        "Rawdata 扫描结果",
        "=" * 50,
        f"  总文件数:        {summary.total_files:>8}",
        f"  Word 类文件:     {summary.word_files:>8}",
        f"    - .docx:       {summary.docx_count:>8}",
        f"    - .doc:        {summary.doc_count:>8}",
        f"    - .docm:       {summary.docm_count:>8}",
        f"  非 Word 文件:    {summary.non_word_count:>8}",
        f"  总大小:          {format_bytes(summary.total_bytes):>8}",
        f"  Vol1 文件:       {summary.vol1_count:>8}",
        f"  Vol2 文件:       {summary.vol2_count:>8}",
        "=" * 50,
    ]
    return "\n".join(lines)
