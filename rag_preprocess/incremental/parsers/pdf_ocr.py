"""Offline PaddleOCR adapter. Model paths are explicit; it never downloads models."""
from __future__ import annotations
from pathlib import Path

class OfflinePaddleOcr:
    def __init__(self, model_root: Path, det_name: str, rec_name: str, cpu_threads: int=4):
        self.model_root=model_root; self.det=model_root/det_name; self.rec=model_root/rec_name; self.cpu_threads=cpu_threads; self._engine=None
    def _load(self):
        if self._engine is not None: return self._engine
        if not self.det.is_dir() or not self.rec.is_dir(): raise RuntimeError("OCR_LOCAL_MODEL_MISSING")
        import os; os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR
        # PaddleOCR 3.x otherwise pairs a supplied model directory with its
        # server-model default name, and enables three separately downloaded
        # document-preprocessing models.  Keep every model selection local.
        self._engine=PaddleOCR(
            device="cpu",
            text_detection_model_name=self.det.name,
            text_detection_model_dir=str(self.det),
            text_recognition_model_name=self.rec.name,
            text_recognition_model_dir=str(self.rec),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # Paddle 3.3.1 + oneDNN cannot run these mobile model graphs on
            # the target Windows CPU. Use the portable Paddle CPU executor.
            enable_mkldnn=False,
            enable_hpi=False,
            cpu_threads=self.cpu_threads,
        )
        return self._engine
    def recognize(self, image_path: Path) -> list[tuple[tuple[float,float,float,float],str,float]]:
        result=self._load().predict(str(image_path)); lines=[]
        for page in result:
            data=page.json if hasattr(page,"json") else page
            payload=data.get("res",data) if isinstance(data,dict) else {}
            for box,text,score in zip(payload.get("rec_boxes",[]),payload.get("rec_texts",[]),payload.get("rec_scores",[])):
                lines.append(((float(box[0]),float(box[1]),float(box[2]),float(box[3])),str(text),float(score)))
        return lines
