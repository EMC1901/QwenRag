"""通用工具函数。"""

import hashlib
import logging
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    """配置日志。"""
    logger = logging.getLogger("rag_preprocess")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)

    return logger


def compute_content_hash(text: str) -> str:
    """计算文本内容的 SHA256。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_bytes(size_bytes: int) -> str:
    """将字节数转为人类可读格式。"""
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TiB"


def safe_open_jsonl(path: Path):
    """安全打开 JSONL 文件进行追加写入。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8")
