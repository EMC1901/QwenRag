"""处理路径、相对路径、输出目录创建。"""

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """确保目录存在，不存在则创建。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_to_base(file_path: Path, base_dir: Path) -> str:
    """获取文件相对于 base_dir 的相对路径字符串。"""
    try:
        return file_path.relative_to(base_dir).as_posix()
    except ValueError:
        return file_path.as_posix()


def ensure_output_dirs(output_dir: Path) -> None:
    """创建所有输出子目录。"""
    dirs = [
        output_dir / "vector_index",
        output_dir / "logs",
        output_dir / "exports",
        output_dir / "converted_docx",
    ]
    for d in dirs:
        ensure_dir(d)
