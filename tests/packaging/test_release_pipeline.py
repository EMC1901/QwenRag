from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
VERIFY_RELEASE = ROOT / "packaging" / "scripts" / "verify_release.ps1"
BUILD_INSTALLER = ROOT / "packaging" / "scripts" / "build_installer.ps1"
BUILD_RUNTIME = ROOT / "packaging" / "scripts" / "build_runtime.ps1"
STAGE_KB = ROOT / "packaging" / "scripts" / "stage_initial_kb.ps1"
BUILD_RELEASE = ROOT / "packaging" / "scripts" / "build_release.ps1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _create_release(root: Path) -> None:
    (root / "QwenRAG-1.0.0-Setup.exe").write_bytes(b"installer")
    (root / "QwenRAG-1.0.0-Setup-1.bin").write_bytes(b"volume")
    (root / "安装说明.md").write_text("离线安装说明", encoding="utf-8")
    (root / "模型部署与适配说明.md").write_text("模型由实施人员部署", encoding="utf-8")
    (root / "初始知识库说明.md").write_text("初始知识库快照说明", encoding="utf-8")
    (root / "用户使用说明.md").write_text("用户使用说明", encoding="utf-8")
    (root / "故障排查手册.md").write_text("故障排查手册", encoding="utf-8")
    (root / "客户机实施与验收清单.md").write_text("客户机验收清单", encoding="utf-8")
    (root / "deployment.customer.example.json").write_text("{}", encoding="utf-8")
    manifest = {
        "product": "QwenRAG",
        "version": "1.0.0",
        "installation_media": [
            "QwenRAG-1.0.0-Setup.exe",
            "QwenRAG-1.0.0-Setup-1.bin",
        ],
    }
    (root / "release-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    entries = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    (root / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")


def _verify(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(VERIFY_RELEASE),
            "-ReleaseDirectory",
            str(root),
            "-RequireDocumentation",
        ],
        cwd=ROOT,
        text=True,
        encoding="gbk",
        errors="replace",
        capture_output=True,
        check=False,
    )


def test_release_verifier_accepts_complete_hashed_release(tmp_path: Path) -> None:
    _create_release(tmp_path)

    result = _verify(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"status":"ok"' in result.stdout


def test_release_verifier_rejects_tampered_media(tmp_path: Path) -> None:
    _create_release(tmp_path)
    (tmp_path / "QwenRAG-1.0.0-Setup-1.bin").write_bytes(b"tampered")

    result = _verify(tmp_path)

    assert result.returncode != 0
    assert "Hash mismatch" in result.stderr


def test_release_scripts_enforce_offline_and_safe_release_contracts() -> None:
    runtime_source = BUILD_RUNTIME.read_text(encoding="utf-8")
    installer_source = BUILD_INSTALLER.read_text(encoding="utf-8")
    verifier_source = VERIFY_RELEASE.read_text(encoding="utf-8")
    snapshot_source = STAGE_KB.read_text(encoding="utf-8")
    release_source = BUILD_RELEASE.read_text(encoding="utf-8")

    assert "--no-index" in runtime_source
    assert "--require-hashes" in runtime_source
    assert "Git worktree is dirty" in runtime_source
    assert "Release directory already exists" in installer_source
    assert "ExistingInnoOutput" in installer_source
    assert "release-manifest.json" in installer_source
    assert "SHA256SUMS.txt" in installer_source
    assert "Assert-HashManifest" in verifier_source
    assert "$hostUtilityModule" in verifier_source
    assert "Assert-OcrAssets" in verifier_source
    assert "stage_kb_snapshot.py" in snapshot_source
    assert "EmbeddingRevision" in release_source
    assert "Full automated acceptance test suite failed" in release_source
