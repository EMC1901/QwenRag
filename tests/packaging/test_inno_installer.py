from __future__ import annotations

from pathlib import Path


INSTALLER = (
    Path(__file__).resolve().parents[2] / "packaging" / "installer" / "QwenRAG.iss"
)


def _installer() -> str:
    return INSTALLER.read_text(encoding="utf-8")


def test_installer_is_per_user_offline_and_preserves_mutable_data() -> None:
    source = _installer()

    assert "PrivilegesRequired=lowest" in source
    assert "DefaultDirName={localappdata}\\Programs\\QwenRAG" in source
    assert "DiskSpanning=yes" in source
    assert "DestDir: \"{app}\\resources\\ocr\"" in source
    uninstall_section = source.split("[UninstallDelete]", 1)[1].split("[Code]", 1)[0]
    assert "{localappdata}\\QwenRAG\\data" not in uninstall_section


def test_installer_blocks_active_runtime_and_invalid_existing_knowledge_base() -> None:
    source = _installer()

    assert "check-runtime-active" in source
    assert "ExistingKbState = KbInvalid" in source
    assert "ExistingKbState = KbMissing" in source
    assert "GetSpaceOnDisk64" in source
    assert "MinimumFreeSpaceWithKbMB" in source
    assert "InitialKbDir" in source and "nocompression" in source


def test_installer_migrates_config_before_initialization_and_runs_diagnostics() -> None:
    source = _installer()

    migration = source.index("RunRuntime('config migrate')")
    initialization = source.index("RunRuntime('config init')")
    diagnostic = source.index("RunRuntime('diagnose-install')")
    assert migration < initialization < diagnostic
    assert "kb-init-empty" in source
    assert "VerifyInitialKnowledgeBase" in source
