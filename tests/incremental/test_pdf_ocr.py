"""OCR adapter configuration must stay local and portable on Windows CPU."""
from __future__ import annotations

import sys
import types
from pathlib import Path

from rag_preprocess.incremental.parsers.pdf_ocr import OfflinePaddleOcr


def test_offline_ocr_supplies_matching_local_models_and_disables_optional_downloads(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "PP-OCRv5_mobile_det").mkdir()
    (tmp_path / "PP-OCRv5_mobile_rec").mkdir()
    captured: dict[str, object] = {}

    class FakePaddleOcr:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=FakePaddleOcr))
    adapter = OfflinePaddleOcr(tmp_path, "PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec")
    adapter._load()

    assert captured["text_detection_model_name"] == "PP-OCRv5_mobile_det"
    assert captured["text_recognition_model_name"] == "PP-OCRv5_mobile_rec"
    assert captured["use_doc_orientation_classify"] is False
    assert captured["use_doc_unwarping"] is False
    assert captured["use_textline_orientation"] is False
    assert captured["enable_mkldnn"] is False
