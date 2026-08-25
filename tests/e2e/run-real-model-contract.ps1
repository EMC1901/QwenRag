param(
    [string] $Runtime = (Join-Path $env:LOCALAPPDATA 'Programs\QwenRAG\QwenRagRuntime.exe')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Runtime -PathType Leaf)) { throw "Installed runtime is missing: $Runtime" }
& $Runtime config validate
if ($LASTEXITCODE -ne 0) { throw 'Deployment configuration validation failed.' }
& $Runtime config test-models
if ($LASTEXITCODE -ne 0) { throw 'Local LLM or Embedding contract test failed.' }
Write-Output 'Local real-model contract acceptance passed.'
