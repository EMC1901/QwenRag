"""Customer-safe final result reports (metadata only, never document body)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Mapping

from .persistence import write_result


def write_final_result(
    path: Path,
    task_id: str,
    rows: list[object],
    *,
    task: Mapping[str, object] | None = None,
) -> None:
    """Create a UTF-8 BOM report understandable without technical logs."""

    task = task or {}
    state_counts = Counter(str(getattr(row, "state", "UNKNOWN")) for row in rows)
    action_counts = Counter(str(getattr(row, "action", "UNKNOWN")) for row in rows)
    lines = [
        "任务概要",
        f"- 任务编号：{task_id}",
        f"- 开始时间：{task.get('started_at', '未记录')}",
        f"- 结束时间：{task.get('finished_at', '未记录')}",
        f"- 最终状态：{task.get('state', 'UNKNOWN')}",
        f"- Delta 编号：{task.get('delta_id', '无')}",
        f"- 发布后的 manifest 修订号：{task.get('manifest_revision', '无')}",
        (
            "- 文件统计："
            f"新增 {action_counts['NEW']}，更新 {action_counts['UPDATE']}，"
            f"重复未变化 {action_counts['DUPLICATE_UNCHANGED']}，"
            f"未就绪 {state_counts['NOT_READY']}，不支持 {state_counts['UNSUPPORTED']}，"
            f"已归档 {state_counts['ARCHIVED']}，"
            f"待补归档 {state_counts['PUBLISHED_ARCHIVE_FAILED']}，"
            f"失败 {state_counts['FAILED']}"
        ),
        "",
        "逐文件结果",
    ]
    for row in rows:
        file_name = _safe_value(getattr(row, "file_name", None), "未命名文件")
        lines.extend(
            [
                f"- 文件名：{file_name}",
                f"  动作：{_safe_value(getattr(row, 'action', None), '未处理')}",
                f"  状态：{_safe_value(getattr(row, 'state', None), '未知')}",
                f"  标题：{_safe_title(getattr(row, 'title', None))}",
                f"  文件哈希：{_safe_value(getattr(row, 'sha256', None), '')[:12] or '未取得'}",
                f"  警告数：{len(getattr(row, 'warning_codes', None) or [])}",
            ]
        )
        archive_path = getattr(row, "archive_relative_path", None)
        if isinstance(archive_path, str) and archive_path:
            lines.append(f"  归档相对路径：{archive_path}")
        error = getattr(row, "error_code", None)
        if error:
            lines.append(f"  结果说明：{_user_message(str(error))}")

    lines.extend(
        [
            "",
            "一致性检查",
            f"- 发布校验：{task.get('validation', '未发布或未执行')}",
            f"- Embedding 模型：{task.get('embedding_model', '未记录')}",
            f"- 向量维度：{task.get('embedding_dim', '未记录')}",
            "",
            "后续操作",
            "- 若状态为 SUCCEEDED 或 PARTIAL_SUCCESS，请由管理员启动问答服务并执行一次验收查询。",
            "- 若存在“待补归档”或“失败”，请保留投递目录中的原文件并联系技术支持人员。",
        ]
    )
    write_result(path, "\n".join(lines) + "\n")


def _safe_value(value: object, fallback: str) -> str:
    return value if isinstance(value, str) and value else fallback


def _user_message(error_code: str) -> str:
    if error_code == "ARCHIVE_FAILED":
        return "资料已发布到知识库，但归档未完成；请保留原文件并联系技术支持人员补归档。"
    return error_code


def _safe_title(value: object) -> str:
    """Keep reports useful without accepting arbitrary content or absolute paths."""
    if not isinstance(value, str) or not value:
        return "未识别"
    normalized = " ".join(value.split())[:300]
    normalized = re.sub(r"(?i)(?:[a-z]:[\\/]|\\\\)[^\s]*", "[已隐藏路径]", normalized)
    normalized = re.sub(r"(?<!\S)/(?:projects|home|var|tmp|sevenH|opt|usr|etc)(?:/[^\s]*)?", "[已隐藏路径]", normalized)
    return normalized or "未识别"
