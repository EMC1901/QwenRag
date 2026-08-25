[CmdletBinding()]
param(
    [string] $Python = (Join-Path $PSScriptRoot '..\..\.venv-delivery\Scripts\python.exe'),
    [string] $InnoCompiler = 'C:\Program Files\Inno Setup 7\ISCC.exe',
    [string] $InitialKnowledgeBase,
    [string] $EmbeddingRevision,
    [switch] $SkipRuntimeBuild,
    [switch] $AllowDirty
)

$ErrorActionPreference = 'Stop'
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Full automated acceptance test suite failed.' }
if (-not $SkipRuntimeBuild) {
    & (Join-Path $PSScriptRoot 'build_runtime.ps1') -Python $Python -AllowDirty:$AllowDirty
    if ($LASTEXITCODE -ne 0) { throw 'Runtime build failed.' }
}
if (-not [string]::IsNullOrWhiteSpace($InitialKnowledgeBase)) {
    if ([string]::IsNullOrWhiteSpace($EmbeddingRevision)) { throw 'EmbeddingRevision is required when packaging an initial knowledge base.' }
    $version = (& (Join-Path $PSScriptRoot '..\build\dist\QwenRagRuntime\QwenRagRuntime.exe') version).Trim()
    & (Join-Path $PSScriptRoot 'stage_initial_kb.ps1') -SourceKnowledgeBase $InitialKnowledgeBase -Version $version -EmbeddingRevision $EmbeddingRevision -Python $Python
    if ($LASTEXITCODE -ne 0) { throw 'Initial knowledge-base staging failed.' }
}
& (Join-Path $PSScriptRoot 'build_installer.ps1') -InnoCompiler $InnoCompiler -AllowDirty:$AllowDirty
if ($LASTEXITCODE -ne 0) { throw 'Installer release build failed.' }
