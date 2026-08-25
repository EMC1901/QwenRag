param(
    [Parameter(Mandatory = $true)][string] $SetupPath,
    [switch] $AcceptDestructiveTest
)

$ErrorActionPreference = 'Stop'
if (-not $AcceptDestructiveTest) { throw 'This script installs and uninstalls QwenRAG. Re-run with -AcceptDestructiveTest on a clean VM only.' }
$setup = (Resolve-Path $SetupPath).Path
$mediaDirectory = Split-Path -Parent $setup
if (@(Get-ChildItem -LiteralPath $mediaDirectory -Filter 'QwenRAG-*-Setup-*.bin').Count -eq 0) { throw 'Installation media is incomplete: Setup-*.bin is missing.' }
$programRoot = Join-Path $env:LOCALAPPDATA 'Programs\QwenRAG'
$dataRoot = Join-Path $env:LOCALAPPDATA 'QwenRAG'
if ((Test-Path -LiteralPath $programRoot) -or (Test-Path -LiteralPath $dataRoot)) { throw 'Clean-VM test requires no existing QwenRAG program or data directory.' }
$logRoot = Join-Path $env:TEMP ('qwenrag-e2e-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
try {
    $install = Start-Process -FilePath $setup -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /LOG=`"$(Join-Path $logRoot 'install.log')`"" -Wait -PassThru
    if ($install.ExitCode -ne 0) { throw "Installer failed with exit code $($install.ExitCode)." }
    $runtime = Join-Path $programRoot 'QwenRagRuntime.exe'
    & $runtime diagnose-install
    if ($LASTEXITCODE -ne 0) { throw 'Installed runtime diagnostics failed.' }
    $uninstaller = Get-ChildItem -LiteralPath $programRoot -Filter 'unins*.exe' | Select-Object -First 1 -ExpandProperty FullName
    if ([string]::IsNullOrWhiteSpace($uninstaller)) { throw 'Uninstaller is missing.' }
    $uninstall = Start-Process -FilePath $uninstaller -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /LOG=`"$(Join-Path $logRoot 'uninstall.log')`"" -Wait -PassThru
    if ($uninstall.ExitCode -ne 0) { throw "Uninstaller failed with exit code $($uninstall.ExitCode)." }
    if (Test-Path -LiteralPath $programRoot) { throw 'Uninstall did not remove the program directory.' }
    if (-not (Test-Path -LiteralPath $dataRoot)) { throw 'Uninstall incorrectly removed customer data.' }
    Write-Output 'Clean-VM installer acceptance passed.'
} finally {
    if (Test-Path -LiteralPath $logRoot) { Remove-Item -LiteralPath $logRoot -Recurse -Force }
}
