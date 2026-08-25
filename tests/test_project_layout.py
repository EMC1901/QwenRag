"""Keep project dependency manifests and delivery entry points organized."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_ROOT = PROJECT_ROOT / "requirements"


def test_dependency_manifests_are_grouped_by_project_component() -> None:
    expected = {
        "base.txt",
        "gateway.txt",
        "local-rag.txt",
        "incremental-rag.txt",
        "incremental-rag.lock.txt",
        "delivery.in",
        "delivery.lock.txt",
    }

    present = {path.name for path in REQUIREMENTS_ROOT.iterdir() if path.is_file()}

    assert expected <= present
    assert not list(PROJECT_ROOT.glob("requirements-*.txt"))


def test_delivery_dependency_includes_resolve_inside_requirements_directory() -> None:
    manifest = REQUIREMENTS_ROOT / "delivery.in"
    include_paths = [
        line.removeprefix("-r ").strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.startswith("-r ")
    ]

    assert include_paths == [
        "local-rag.txt",
        "gateway.txt",
        "incremental-rag.lock.txt",
    ]
    assert all((REQUIREMENTS_ROOT / relative_path).is_file() for relative_path in include_paths)


def test_runtime_build_uses_the_grouped_delivery_lock() -> None:
    build_script = PROJECT_ROOT / "packaging" / "scripts" / "build_runtime.ps1"
    source = build_script.read_text(encoding="utf-8")

    assert "requirements\\delivery.lock.txt" in source
