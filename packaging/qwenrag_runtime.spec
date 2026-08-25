# -*- mode: python ; coding: utf-8 -*-
"""One-folder PyInstaller definition for the offline QwenRAG runtime."""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


# Hook discovery imports part of PaddleOCR.  It must never contact model hosts
# while producing the offline customer runtime.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

PROJECT_ROOT = Path(SPECPATH).parent

PACKAGES = (
    "qwenrag_runtime",
    "local_rag_app",
    "model_gateway",
    "rag_preprocess",
    "uvicorn",
    "fastapi",
    "starlette",
    "faiss",
    "paddle",
    "paddleocr",
    "fitz",
    "docx",
    "charset_normalizer",
)

hiddenimports = []
for package in PACKAGES:
    hiddenimports.extend(collect_submodules(package))

binaries = []
datas = []
for package in ("faiss", "paddle", "paddleocr", "fitz"):
    binaries.extend(collect_dynamic_libs(package))
    datas.extend(collect_data_files(package))
datas.extend(collect_data_files("charset_normalizer"))

# Customer OCR model files are intentionally not bundled in _internal.  The
# installer copies them to {app}\resources\ocr, where RuntimePaths resolves
# them as a read-only external resource directory.
a = Analysis(
    [str(PROJECT_ROOT / "packaging" / "runtime_entry.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    excludes=["pytest", "tests", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="QwenRagRuntime",
    console=True,
    debug=False,
    strip=False,
    upx=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="QwenRagRuntime",
    strip=False,
    upx=False,
)
