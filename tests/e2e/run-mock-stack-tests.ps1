param(
    [string] $Python = (Join-Path $PSScriptRoot '..\..\.venv-delivery\Scripts\python.exe')
)

$ErrorActionPreference = 'Stop'
& $Python -m pytest tests\e2e\test_mock_model_stack.py -q
if ($LASTEXITCODE -ne 0) { throw 'Mock-model stack acceptance test failed.' }
