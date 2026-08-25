from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from qwenrag_runtime.deployment import DeploymentConfig


ROOT = Path(__file__).resolve().parents[2]
DELIVERY_DOCS = ROOT / "packaging" / "release_docs"


def test_customer_delivery_documents_are_complete() -> None:
    required = {
        "安装说明.md",
        "模型部署与适配说明.md",
        "初始知识库说明.md",
        "用户使用说明.md",
        "故障排查手册.md",
        "客户机实施与验收清单.md",
        "deployment.customer.example.json",
    }

    present = {path.name for path in DELIVERY_DOCS.iterdir() if path.is_file()}

    assert required <= present
    for document in required - {"deployment.customer.example.json"}:
        assert (DELIVERY_DOCS / document).read_text(encoding="utf-8-sig").strip()


def test_customer_model_template_matches_runtime_configuration_contract() -> None:
    template_path = DELIVERY_DOCS / "deployment.customer.example.json"
    payload = json.loads(template_path.read_text(encoding="utf-8-sig"))
    deployment = DeploymentConfig.model_validate(payload)

    assert deployment.llm.expected_model == "qwen3.6-27b"
    assert deployment.embedding.expected_model == "qwen3-embedding-0.6b"
    assert deployment.embedding.expected_dimension == 1024
    assert deployment.rag.embedding_dimension == 1024
    assert deployment.llm.base_url == "http://127.0.0.1:8001/v1"
    assert deployment.embedding.base_url == "http://127.0.0.1:8002/v1"
    assert deployment.ports.gateway == 8010
    assert deployment.ports.rag == 18080
    for url in (
        deployment.llm.base_url,
        deployment.llm.ready_url,
        deployment.embedding.base_url,
        deployment.embedding.ready_url,
    ):
        assert urlsplit(url).hostname == "127.0.0.1"


def test_customer_documents_do_not_contain_developer_machine_paths() -> None:
    forbidden = ("C:\\projects\\QwenRag", "\\.venv-delivery", "10.", "192.168.")
    for path in DELIVERY_DOCS.iterdir():
        if path.suffix.lower() not in {".md", ".json"}:
            continue
        content = path.read_text(encoding="utf-8-sig")
        assert not any(value in content for value in forbidden), path.name
