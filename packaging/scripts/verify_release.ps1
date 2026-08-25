[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $ReleaseDirectory,
    [string] $RuntimeDir,
    [string] $OcrDir,
    [string] $InitialKbDir,
    [switch] $RequireDocumentation
)

$ErrorActionPreference = 'Stop'

# A PowerShell 7 parent can pass its module search path to Windows PowerShell
# 5.1, where the newer Utility module loads without exposing Get-FileHash.
# Import the module shipped with the current host explicitly before hashing.
$hostUtilityModule = Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1'
Import-Module -Name $hostUtilityModule -ErrorAction Stop

function Get-RelativeName([string] $Root, [string] $Path) {
    $normalizedRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    $normalizedPath = [IO.Path]::GetFullPath($Path)
    if (-not $normalizedPath.StartsWith($normalizedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes verification root: $Path"
    }
    return $normalizedPath.Substring($normalizedRoot.Length).Replace('\', '/')
}

function Assert-HashManifest([string] $Root) {
    $manifest = Join-Path $Root 'SHA256SUMS.txt'
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) { throw 'SHA256SUMS.txt is missing.' }
    $entries = @((Get-Content -LiteralPath $manifest -Encoding utf8) | Where-Object { $_.Trim() })
    if ($entries.Count -eq 0) { throw 'SHA256SUMS.txt is empty.' }
    foreach ($line in $entries) {
        $parts = $line -split '\s{2,}', 2
        if ($parts.Count -ne 2 -or $parts[0] -notmatch '^[A-Fa-f0-9]{64}$') { throw "Invalid SHA256SUMS entry: $line" }
        $target = Join-Path $Root ($parts[1].Replace('/', '\'))
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw "Hashed file is missing: $($parts[1])" }
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash
        if ($actual -ne $parts[0].ToUpperInvariant()) { throw "Hash mismatch: $($parts[1])" }
    }
    return $entries.Count
}

function Assert-NoForbiddenFiles([string] $Root) {
    $forbidden = '(?i)(^|/)(\.env($|\.)|rawdata|\.pytest_cache|__pycache__|logs?)(/|$)|(^|/)worker[^/]*\.log$'
    foreach ($item in Get-ChildItem -LiteralPath $Root -Recurse -Force) {
        $relative = Get-RelativeName $Root $item.FullName
        if ($relative -match $forbidden) { throw "Forbidden release asset: $relative" }
    }
}

function Assert-SafeTextFiles([string] $Root) {
    $extensions = @('.json', '.md', '.txt', '.ps1', '.ini')
    $keyPattern = '(?im)(api[_ -]?key|password|secret|token)\s*["'']?\s*[:=]\s*["'']?(?!<|YOUR_|REPLACE_|null|none|false)[A-Za-z0-9_\-]{20,}'
    foreach ($file in Get-ChildItem -LiteralPath $Root -Recurse -File | Where-Object { $extensions -contains $_.Extension.ToLowerInvariant() }) {
        $content = Get-Content -LiteralPath $file.FullName -Raw -Encoding utf8
        if ($content -match '(?i)C:\\projects\\QwenRag|C:\\Users\\emc20') { throw "Developer-machine path found in: $(Get-RelativeName $Root $file.FullName)" }
        if ($content -match $keyPattern) { throw "Possible secret found in: $(Get-RelativeName $Root $file.FullName)" }
    }
}

function Assert-OcrAssets([string] $Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return }
    foreach ($relative in @('PP-OCRv5_mobile_det\inference.pdiparams', 'PP-OCRv5_mobile_rec\inference.pdiparams')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Path $relative) -PathType Leaf)) { throw "OCR asset is missing: $relative" }
    }
}

function Assert-InitialKnowledgeBase([string] $Path) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Container)) { return }
    foreach ($relative in @('SHA256SUMS.txt', 'snapshot.json', 'metadata.db', 'vector_index\index.faiss', 'vector_index\index.meta.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Path $relative) -PathType Leaf)) { throw "Initial knowledge-base payload is incomplete: $relative" }
    }
    [void](Assert-HashManifest $Path)
    try { $snapshot = Get-Content -LiteralPath (Join-Path $Path 'snapshot.json') -Raw -Encoding utf8 | ConvertFrom-Json } catch { throw 'Initial knowledge-base snapshot.json is invalid.' }
    if ([string]::IsNullOrWhiteSpace($snapshot.embedding_contract.embedding_model) -or [string]::IsNullOrWhiteSpace($snapshot.embedding_contract.embedding_revision) -or $snapshot.embedding_contract.embedding_revision -eq 'legacy-unknown' -or [int]$snapshot.embedding_contract.embedding_dimension -le 0 -or $snapshot.embedding_contract.vector_normalized -ne $true) {
        throw 'Initial knowledge-base embedding contract is incomplete.'
    }
}

$release = (Resolve-Path $ReleaseDirectory).Path
if (-not (Test-Path -LiteralPath (Join-Path $release 'release-manifest.json') -PathType Leaf)) { throw 'release-manifest.json is missing.' }
$manifest = Get-Content -LiteralPath (Join-Path $release 'release-manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($manifest.version)) { throw 'release-manifest.json has no version.' }
if (@($manifest.installation_media).Count -eq 0) { throw 'release-manifest.json has no installation media.' }
foreach ($media in $manifest.installation_media) {
    if (-not (Test-Path -LiteralPath (Join-Path $release $media) -PathType Leaf)) { throw "Installation media is missing: $media" }
}
if ($RequireDocumentation) {
    foreach ($name in @('安装说明.md', '模型部署与适配说明.md', '初始知识库说明.md', '用户使用说明.md', '故障排查手册.md', '客户机实施与验收清单.md', 'deployment.customer.example.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $release $name) -PathType Leaf)) { throw "Release documentation is missing: $name" }
    }
}
$hashCount = Assert-HashManifest $release
Assert-NoForbiddenFiles $release
Assert-SafeTextFiles $release
if (-not [string]::IsNullOrWhiteSpace($RuntimeDir)) { Assert-NoForbiddenFiles ((Resolve-Path $RuntimeDir).Path) }
Assert-OcrAssets $OcrDir
Assert-InitialKnowledgeBase $InitialKbDir
$totalBytes = (Get-ChildItem -LiteralPath $release -Recurse -File | Measure-Object -Property Length -Sum).Sum
[ordered]@{ status = 'ok'; version = $manifest.version; hash_entries = $hashCount; total_bytes = $totalBytes; installation_media = @($manifest.installation_media) } | ConvertTo-Json -Compress
$global:LASTEXITCODE = 0
