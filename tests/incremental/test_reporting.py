from pathlib import Path
from types import SimpleNamespace

from rag_preprocess.incremental.reporting import write_final_result


def test_result_report_has_metadata_not_document_content(tmp_path: Path) -> None:
    path = tmp_path / "result.txt"
    write_final_result(
        path,
        "task",
        [
            SimpleNamespace(state="ARCHIVED", action="NEW", file_name="good.txt", sha256="a" * 64),
            SimpleNamespace(state="FAILED", action="NEW", file_name="bad.pdf", error_code="PDF_OPEN_FAILED"),
        ],
        task={"state": "PARTIAL_SUCCESS", "embedding_model": "model", "embedding_dim": 3},
    )
    text = path.read_text(encoding="utf-8-sig")
    assert "任务概要" in text and "bad.pdf" in text
    assert "PDF_OPEN_FAILED" in text
    assert "PRIVATE-CHUNK" not in text
