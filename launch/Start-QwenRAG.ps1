[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$ErrorActionPreference = 'Stop'

$runtime = Join-Path -Path (Split-Path -Parent $PSScriptRoot) -ChildPath 'QwenRagRuntime.exe'
if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) {
    Write-Host 'QwenRagRuntime.exe was not found. Please reinstall QwenRAG or contact support.' -ForegroundColor Red
    Read-Host 'Press Enter to close this window'
    exit 20
}

& $runtime run
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "QwenRAG failed to start (exit code $exitCode). Check %LOCALAPPDATA%\QwenRAG\logs\supervisor." -ForegroundColor Red
    Read-Host 'Press Enter to close this window'
}
exit $exitCode
