"""统一保存配置，例如 Rawdata 路径、输出路径、chunk 参数、embedding 模型。"""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    """全局构建配置。"""

    # 路径
    rawdata_dir: Path = Path("Rawdata")
    output_dir: Path = Path("rag_data")
    converted_dir: Path = Path("rag_data/converted_docx")

    # 数据库
    db_path: Path = Path("rag_data/metadata.db")

    # chunk 参数
    target_chunk_tokens: int = 700
    max_chunk_tokens: int = 1200
    overlap_tokens: int = 80

    # embedding
    embedding_model: str = "qwen3-embedding-0.6b"
    # Persist an immutable upstream artifact/version label with every new index.
    # The delivery launcher supplies this from deployment.json in stage 4.
    embedding_revision: str = "unconfigured"
    embedding_dim: int = 1024
    embedding_batch_size: int = 128
    vector_normalized: bool = True
    vector_metric: str = "inner_product"

    # FAISS
    faiss_index_path: Path = Path("rag_data/vector_index/index.faiss")

    # 构建控制
    limit: int | None = None
    resume: bool = False
    force: bool = False
    log_level: str = "INFO"


# 默认全局配置实例
default_config = Config()
