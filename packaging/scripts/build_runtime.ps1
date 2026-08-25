[CmdletBinding()]
param(
    [string] $Python = (Join-Path $PSScriptRoot '..\..\.venv-delivery\Scripts\python.exe'),
    [string] $OutputRoot = (Join-Path $PSScriptRoot '..\build'),
    [string] $Wheelhouse = (Join-Path $PSScriptRoot '..\..\wheelhouse'),
    [string] $RequirementsLock = (Join-Path $PSScriptRoot '..\..\requirements\delivery.lock.txt'),
    [switch] $SkipDependencyInstall,
    [switch] $AllowDirty
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonPath = (Resolve-Path $Python).Path
$wheelhousePath = (Resolve-Path $Wheelhouse).Path
$requirementsLockPath = (Resolve-Path $RequirementsLock).Path
$specPath = Join-Path $projectRoot 'packaging\qwenrag_runtime.spec'
$distPath = Join-Path $OutputRoot 'dist'
$workPath = Join-Path $OutputRoot 'work'

if ($env:OS -ne 'Windows_NT') { throw 'QwenRAG Windows runtime must be built on Windows.' }
$commit = (& git -C $projectRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Unable to read the current Git commit.' }
$dirtyEntries = @(& git -C $projectRoot status --porcelain)
if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect the Git worktree.' }
if ($dirtyEntries.Count -gt 0 -and -not $AllowDirty) {
    throw 'Git worktree is dirty. Commit or explicitly use -AllowDirty for a non-release developer build.'
}
if (-not (Test-Path -LiteralPath $wheelhousePath -PathType Container)) { throw "Offline wheelhouse is missing: $wheelhousePath" }
if (-not (Test-Path -LiteralPath $requirementsLockPath -PathType Leaf)) { throw "Delivery lock file is missing: $requirementsLockPath" }

if (-not $SkipDependencyInstall) {
    & $pythonPath -m pip install --no-index --find-links $wheelhousePath --require-hashes -r $requirementsLockPath
    if ($LASTEXITCODE -ne 0) { throw 'Offline dependency installation from the wheelhouse failed.' }
}
& $pythonPath -m PyInstaller --version
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller is not available in the isolated delivery build environment.' }

& $pythonPath -m pytest tests\qwenrag_runtime tests\incremental -q
if ($LASTEXITCODE -ne 0) { throw 'Runtime unit tests failed; frozen build was not started.' }

# one-folder and UPX policy are declared by the checked-in spec.  PyInstaller
# rejects makespec-only flags when a .spec path is passed here.
& $pythonPath -m PyInstaller --noconfirm --clean `
    --distpath $distPath --workpath $workPath $specPath
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }

$runtime = Join-Path $distPath 'QwenRagRuntime\QwenRagRuntime.exe'
if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) { throw 'Frozen runtime executable was not produced.' }
& (Join-Path $PSScriptRoot 'test_frozen_runtime.ps1') -Runtime $runtime
if ($LASTEXITCODE -ne 0) { throw 'Frozen runtime smoke test failed.' }

[ordered]@{
    commit = $commit
    source_dirty = ($dirtyEntries.Count -gt 0)
    python = (& $pythonPath --version).Trim()
    pyinstaller = (& $pythonPath -m PyInstaller --version).Trim()
    runtime = 'dist/QwenRagRuntime/QwenRagRuntime.exe'
    built_at_utc = [DateTime]::UtcNow.ToString('o')
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputRoot 'build-info.json') -Encoding utf8
