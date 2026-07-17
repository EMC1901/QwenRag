from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from rag_preprocess.embedding_client import EmbeddingResult
from rag_preprocess.incremental.embedding import EmbeddingFileError, embed_file_chunks, preflight_embedding
from rag_preprocess.incremental.settings import load_incremental_settings

def _settings(tmp_path: Path):
    return load_incremental_settings(project_root=tmp_path, environ={"INCREMENTAL_KB_ROOT":"data", "OCR_MODEL_DIR":"models/ocr", "EMBEDDING_DIM":"3", "EMBEDDING_BATCH_SIZE":"2"})

def test_file_vectors_are_normalized_and_resume_without_second_request(tmp_path: Path):
    calls=[]
    def embedder(texts, **_kwargs): calls.append(list(texts)); return [EmbeddingResult(True,[3.0,4.0,0.0]) for _ in texts]
    settings=_settings(tmp_path); chunks=[SimpleNamespace(chunk_id="a",chunk_text_for_embedding="one"),SimpleNamespace(chunk_id="b",chunk_text_for_embedding="two")]
    preflight_embedding(settings,embedder=embedder)
    first=embed_file_chunks(settings,"v1",chunks,tmp_path/'vectors',embedder=embedder)
    second=embed_file_chunks(settings,"v1",chunks,tmp_path/'vectors',embedder=embedder)
    assert first == second and len(calls)==2
    assert '"normalized":true' in first.read_text(encoding='utf-8')

def test_retries_transient_failure_but_rejects_bad_vector(tmp_path: Path):
    settings=_settings(tmp_path); attempts=[]; delays=[]
    def transient(texts, **_kwargs):
        attempts.append(1)
        return [EmbeddingResult(False,error_message="connection refused") for _ in texts] if len(attempts)==1 else [EmbeddingResult(True,[1.0,0.0,0.0]) for _ in texts]
    chunk=[SimpleNamespace(chunk_id="a",chunk_text_for_embedding="one")]
    embed_file_chunks(settings,"v2",chunk,tmp_path/'vectors',embedder=transient,sleeper=delays.append,jitter=lambda:0)
    assert delays == [1.0]
    with pytest.raises(EmbeddingFileError):
        embed_file_chunks(settings,"v3",chunk,tmp_path/'other',embedder=lambda texts, **_kwargs:[EmbeddingResult(True,[1.0])],sleeper=lambda _delay:None,jitter=lambda:0)
