"""处理 .doc 和 .docm 转换。"""

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConversionResult:
    """转换结果。"""

    success: bool
    input_path: Path
    output_path: Path | None = None
    error_message: str | None = None
    stderr: str | None = None


def convert_doc_to_docx(input_path: Path, output_dir: Path) -> ConversionResult:
    """把 doc 转成 docx（使用 LibreOffice headless）。"""
    return _convert_with_libreoffice(input_path, output_dir)


def convert_docm_to_docx(input_path: Path, output_dir: Path) -> ConversionResult:
    """把 docm 转成 docx，不执行宏。"""
    return _convert_with_libreoffice(input_path, output_dir)


def _convert_with_libreoffice(input_path: Path, output_dir: Path) -> ConversionResult:
    """使用 LibreOffice headless 转换文档。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "soffice",
        "--headless",
        "--convert-to", "docx",
        "--outdir", str(output_dir),
        str(input_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return ConversionResult(
                success=False,
                input_path=input_path,
                error_message=f"LibreOffice exited with code {result.returncode}",
                stderr=result.stderr,
            )

        # LibreOffice 输出的文件名基于输入文件名，后缀变为 .docx
        output_name = input_path.stem + ".docx"
        output_path = output_dir / output_name

        if output_path.exists():
            return ConversionResult(success=True, input_path=input_path, output_path=output_path)
        else:
            return ConversionResult(
                success=False,
                input_path=input_path,
                error_message="Output file not found after conversion",
                stderr=result.stderr,
            )
    except FileNotFoundError:
        return ConversionResult(
            success=False,
            input_path=input_path,
            error_message="LibreOffice (soffice) not found in PATH",
        )
    except subprocess.TimeoutExpired:
        return ConversionResult(
            success=False,
            input_path=input_path,
            error_message="Conversion timed out (120s)",
        )


def get_parseable_docx_path(source_file, rawdata_dir: Path, converted_dir: Path) -> Path | None:
    """返回可解析的 docx 路径。

    - .docx 直接返回原路径
    - .doc/.docm 返回转换后的路径（需要先转换）
    """
    ext = source_file.extension.lower()
    raw_path = rawdata_dir / source_file.relative_path

    if ext == ".docx":
        return raw_path if raw_path.exists() else None

    # .doc / .docm 需要先转换
    converted_name = Path(source_file.file_name).stem + ".docx"
    converted_path = converted_dir / converted_name
    if converted_path.exists():
        return converted_path

    return None
