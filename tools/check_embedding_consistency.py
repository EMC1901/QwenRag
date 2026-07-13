#!/usr/bin/env python3
"""校验阶段 8 的 SQLite 状态与 JSONL 向量文件是否一致。

默认只做计数校验；``--mode full`` 会以流式方式逐行验证全部向量，
不会将大型 JSONL 文件一次性载入内存，也不会修改任何输入文件。
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EMBEDDING_DIM = 1024
REQUIRED_FIELDS = {"vector_id", "chunk_id", "model", "dim", "normalized", "vector"}


@dataclass
class ConsistencyReport:
    """一次只读校验的结果。"""

    total_chunks: int = 0
    success_count: int = 0
    failed_count: int = 0
    mapped_count: int = 0
    missing_vector_id_count: int = 0
    unique_vector_id_count: int = 0
    invalid_vector_id_count: int = 0
    jsonl_line_count: int = 0
    issues: list[str] = field(default_factory=list)
    read_error: str | None = None

    @property
    def is_consistent(self) -> bool:
        return not self.issues and self.read_error is None

    @property
    def is_complete(self) -> bool:
        return (
            self.is_consistent
            and self.total_chunks > 0
            and self.success_count == self.total_chunks
            and self.failed_count == 0
            and self.mapped_count == self.total_chunks
        )

    def exit_code(self, require_complete: bool) -> int:
        if self.read_error is not None:
            return 3
        if self.issues:
            return 1
        if require_complete and not self.is_complete:
            return 2
        return 0


def _add_issue(report: ConsistencyReport, message: str) -> None:
    """保留足够的错误上下文，避免损坏大文件时无限增长输出。"""
    if len(report.issues) < 20:
        report.issues.append(message)


def _open_read_only_db(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _validate_chunks_schema(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
    required = {"chunk_id", "vector_id", "embedding_status"}
    missing = required - columns
    if missing:
        raise sqlite3.DatabaseError(f"chunks 表缺少字段: {', '.join(sorted(missing))}")


def _count_nonempty_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _validate_json_record(
    record: Any,
    line_number: int,
    expected_mappings: dict[int, str],
    seen_vector_ids: set[int],
    seen_chunk_ids: set[str],
    report: ConsistencyReport,
) -> None:
    if not isinstance(record, dict):
        _add_issue(report, f"JSONL 第 {line_number} 行不是对象")
        return
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        _add_issue(report, f"JSONL 第 {line_number} 行缺少字段: {', '.join(sorted(missing))}")
        return

    vector_id = record["vector_id"]
    chunk_id = record["chunk_id"]
    dim = record["dim"]
    vector = record["vector"]
    if isinstance(vector_id, bool) or not isinstance(vector_id, int) or vector_id < 0:
        _add_issue(report, f"JSONL 第 {line_number} 行 vector_id 非负整数")
        return
    if not isinstance(chunk_id, str) or not chunk_id:
        _add_issue(report, f"JSONL 第 {line_number} 行 chunk_id 为空或不是字符串")
        return
    if not isinstance(record["model"], str) or not record["model"]:
        _add_issue(report, f"JSONL 第 {line_number} 行 model 为空或不是字符串")
    if not isinstance(record["normalized"], bool):
        _add_issue(report, f"JSONL 第 {line_number} 行 normalized 不是布尔值")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim != EMBEDDING_DIM:
        _add_issue(report, f"JSONL 第 {line_number} 行 dim 应为 {EMBEDDING_DIM}")
    if not isinstance(vector, list) or len(vector) != EMBEDDING_DIM:
        _add_issue(report, f"JSONL 第 {line_number} 行 vector 长度应为 {EMBEDDING_DIM}")
    elif not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in vector
    ):
        _add_issue(report, f"JSONL 第 {line_number} 行 vector 含非有限数值")

    if vector_id in seen_vector_ids:
        _add_issue(report, f"JSONL 第 {line_number} 行 vector_id 重复: {vector_id}")
    seen_vector_ids.add(vector_id)
    if chunk_id in seen_chunk_ids:
        _add_issue(report, f"JSONL 第 {line_number} 行 chunk_id 重复: {chunk_id}")
    seen_chunk_ids.add(chunk_id)

    expected_chunk_id = expected_mappings.get(vector_id)
    if expected_chunk_id is None:
        _add_issue(report, f"JSONL 第 {line_number} 行 vector_id 未映射到 SQLite: {vector_id}")
    elif expected_chunk_id != chunk_id:
        _add_issue(
            report,
            f"JSONL 第 {line_number} 行 chunk_id 与 SQLite 不一致: {vector_id}",
        )


def check_embedding_consistency(
    db_path: Path,
    vector_path: Path,
    *,
    mode: str = "quick",
) -> ConsistencyReport:
    """对阶段 8 结果进行只读检查，返回可供 CLI 与续跑逻辑使用的报告。"""
    report = ConsistencyReport()
    if mode not in {"quick", "full"}:
        report.read_error = f"未知检查模式: {mode}"
        return report
    if not db_path.is_file():
        report.read_error = f"数据库不存在或不是文件: {db_path}"
        return report
    if not vector_path.is_file():
        report.read_error = f"向量文件不存在或不是文件: {vector_path}"
        return report

    try:
        with _open_read_only_db(db_path) as conn:
            _validate_chunks_schema(conn)
            report.total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            report.success_count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding_status = 'success'"
            ).fetchone()[0]
            report.failed_count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding_status = 'failed'"
            ).fetchone()[0]
            report.mapped_count = conn.execute(
                """SELECT COUNT(*) FROM chunks
                   WHERE embedding_status = 'success' AND vector_id IS NOT NULL"""
            ).fetchone()[0]
            report.missing_vector_id_count = conn.execute(
                """SELECT COUNT(*) FROM chunks
                   WHERE embedding_status = 'success' AND vector_id IS NULL"""
            ).fetchone()[0]
            report.unique_vector_id_count = conn.execute(
                """SELECT COUNT(DISTINCT vector_id) FROM chunks
                   WHERE embedding_status = 'success' AND vector_id IS NOT NULL"""
            ).fetchone()[0]
            report.invalid_vector_id_count = conn.execute(
                """SELECT COUNT(*) FROM chunks
                   WHERE embedding_status = 'success' AND vector_id < 0"""
            ).fetchone()[0]

            if mode == "full":
                expected_mappings = {
                    int(row["vector_id"]): row["chunk_id"]
                    for row in conn.execute(
                        """SELECT vector_id, chunk_id FROM chunks
                           WHERE embedding_status = 'success' AND vector_id IS NOT NULL"""
                    )
                }
            else:
                expected_mappings = {}
    except (OSError, sqlite3.Error) as exc:
        report.read_error = f"无法读取数据库 {db_path}: {exc}"
        return report

    try:
        if mode == "quick":
            report.jsonl_line_count = _count_nonempty_jsonl_lines(vector_path)
        else:
            seen_vector_ids: set[int] = set()
            seen_chunk_ids: set[str] = set()
            with vector_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    report.jsonl_line_count += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        _add_issue(report, f"JSONL 第 {line_number} 行不是合法 JSON: {exc.msg}")
                        continue
                    _validate_json_record(
                        record,
                        line_number,
                        expected_mappings,
                        seen_vector_ids,
                        seen_chunk_ids,
                        report,
                    )
            missing_db_vector_ids = set(expected_mappings) - seen_vector_ids
            if missing_db_vector_ids:
                _add_issue(report, f"JSONL 缺少 SQLite 中的 {len(missing_db_vector_ids)} 条 vector_id 映射")
    except (OSError, UnicodeDecodeError) as exc:
        report.read_error = f"无法读取向量文件 {vector_path}: {exc}"
        return report

    if report.success_count != report.mapped_count:
        _add_issue(report, "success 数量与 success/vector_id 映射数量不一致")
    if report.missing_vector_id_count:
        _add_issue(report, "存在 success 但 vector_id 为空的 chunk")
    if report.unique_vector_id_count != report.mapped_count:
        _add_issue(report, "SQLite 中 success/vector_id 存在重复 vector_id")
    if report.invalid_vector_id_count:
        _add_issue(report, "SQLite 中存在负数 vector_id")
    if report.jsonl_line_count != report.mapped_count:
        _add_issue(report, "JSONL 非空行数与 SQLite success/vector_id 数量不一致")
    return report


def _print_report(report: ConsistencyReport, mode: str, require_complete: bool) -> None:
    print(f"检查模式: {mode}")
    print(f"chunks 总数: {report.total_chunks}")
    print(f"embedding success: {report.success_count}")
    print(f"embedding failed: {report.failed_count}")
    print(f"success 且有 vector_id: {report.mapped_count}")
    print(f"success 缺 vector_id: {report.missing_vector_id_count}")
    print(f"SQLite 唯一 vector_id: {report.unique_vector_id_count}")
    print(f"JSONL 非空行数: {report.jsonl_line_count}")
    if report.read_error:
        print(f"状态: READ_ERROR: {report.read_error}")
    elif report.issues:
        print("状态: ISSUE")
        for issue in report.issues:
            print(f"  - {issue}")
    elif require_complete and not report.is_complete:
        print("状态: INCOMPLETE")
    else:
        print("状态: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="检查阶段 8 SQLite/JSONL 一致性")
    parser.add_argument("--db", default="rag_data/metadata.db", help="SQLite 数据库路径")
    parser.add_argument(
        "--vector-file",
        default="rag_data/vector_index/embeddings.jsonl",
        help="embedding JSONL 文件路径",
    )
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--require-complete", action="store_true", help="未全部完成时返回 2")
    args = parser.parse_args()

    report = check_embedding_consistency(Path(args.db), Path(args.vector_file), mode=args.mode)
    _print_report(report, args.mode, args.require_complete)
    raise SystemExit(report.exit_code(args.require_complete))


if __name__ == "__main__":
    main()
