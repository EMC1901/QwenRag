"""汇总构建质量报告。"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


@dataclass
class BuildReport:
    """构建质量报告。"""

    input_summary: dict = field(default_factory=dict)
    records_summary: dict = field(default_factory=dict)
    conversion_summary: dict = field(default_factory=dict)
    parse_summary: dict = field(default_factory=dict)
    dedup_summary: dict = field(default_factory=dict)
    chunk_summary: dict = field(default_factory=dict)
    embedding_summary: dict = field(default_factory=dict)
    index_summary: dict = field(default_factory=dict)
    qa_summary: dict = field(default_factory=dict)
    errors_summary: dict = field(default_factory=dict)


def generate_manifest(
    chunk_count: int,
    embedding_model: str,
    embedding_dim: int,
    build_time: str | None = None,
) -> dict:
    """生成 manifest.json 内容。"""
    return {
        "dataset_name": "law_flk_rag_knowledge_base",
        "build_time": build_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_version": "v1.0",
        "source_file_count": 0,    # 阶段 1 填入
        "docx_count": 0,            # 阶段 1 填入
        "doc_count": 0,             # 阶段 1 填入
        "docm_count": 0,            # 阶段 1 填入
        "chunk_count": chunk_count,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "index_type": "faiss",
        "metadata_db": "metadata.db",
        "vector_index": "vector_index/index.faiss",
    }


def save_build_report(report: BuildReport, output_dir: Path) -> None:
    """保存构建报告到 build_report.json。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "build_report.json"

    # 将 dataclass 转为 dict
    data = {
        "input_summary": report.input_summary,
        "records_summary": report.records_summary,
        "conversion_summary": report.conversion_summary,
        "parse_summary": report.parse_summary,
        "dedup_summary": report.dedup_summary,
        "chunk_summary": report.chunk_summary,
        "embedding_summary": report.embedding_summary,
        "index_summary": report.index_summary,
        "qa_summary": report.qa_summary,
        "errors_summary": report.errors_summary,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
