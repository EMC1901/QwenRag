"""Safe local checks used by the offline installer, never by Chatbox clients."""

from __future__ import annotations

from pathlib import Path

from .deployment import deployment_files, load_deployment, load_secrets
from .paths import RuntimePaths
from .runtime_lock import is_runtime_lock_held


def diagnose_install(paths: RuntimePaths) -> dict[str, object]:
    """Report install prerequisites without contacting models or exposing secrets."""
    files = deployment_files(paths)
    configuration = "missing"
    try:
        load_deployment(files.deployment_path)
        load_secrets(files.secrets_path)
        configuration = "ok"
    except Exception:
        pass
    knowledge_base = "ok" if _has_knowledge_base(paths.knowledge_base_root) else "missing_or_invalid"
    runtime_lock = paths.runtime_root / "locks" / "rag.lock"
    running = is_runtime_lock_held(runtime_lock)
    return {
        "status": "ready" if configuration == "ok" and knowledge_base == "ok" and not running else "not_ready",
        "configuration": configuration,
        "knowledge_base": knowledge_base,
        "runtime_active": running,
        "data_root": str(paths.data_root),
    }


def runtime_is_active(paths: RuntimePaths) -> bool:
    """Return only whether the shared RAG/ingestion session lock is held."""
    return is_runtime_lock_held(paths.runtime_root / "locks" / "rag.lock")


def _has_knowledge_base(root: Path) -> bool:
    required = (
        root / "metadata.db",
        root / "vector_index" / "index.faiss",
        root / "vector_index" / "index.meta.json",
        root / "knowledge_manifest.json",
    )
    return all(item.is_file() for item in required)
