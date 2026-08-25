from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_e2e_scripts_protect_existing_data_and_cover_mock_and_real_model_paths() -> None:
    clean_vm = (ROOT / "tests" / "e2e" / "run-clean-vm-tests.ps1").read_text(encoding="utf-8")
    mock = (ROOT / "tests" / "e2e" / "run-mock-stack-tests.ps1").read_text(encoding="utf-8")
    real = (ROOT / "tests" / "e2e" / "run-real-model-contract.ps1").read_text(encoding="utf-8")

    assert "AcceptDestructiveTest" in clean_vm
    assert "Clean-VM test requires no existing" in clean_vm
    assert "diagnose-install" in clean_vm
    assert "Uninstall incorrectly removed customer data" in clean_vm
    assert "test_mock_model_stack.py" in mock
    assert "config test-models" in real
