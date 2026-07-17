from pathlib import Path
import pytest
from local_rag_app.knowledge_base import KnowledgeBaseLoadError, _resolve_generation_root

def test_generation_pointer_selects_only_relative_child(tmp_path: Path):
    target=tmp_path/'kb_generations'/'g1'; target.mkdir(parents=True); (tmp_path/'current_generation.txt').write_text('kb_generations/g1\n',encoding='utf-8')
    assert _resolve_generation_root(tmp_path)==target

def test_generation_pointer_rejects_absolute_and_escape_paths(tmp_path: Path):
    (tmp_path/'current_generation.txt').write_text('../outside\n',encoding='utf-8')
    with pytest.raises(KnowledgeBaseLoadError): _resolve_generation_root(tmp_path)
