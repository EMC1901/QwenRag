[CmdletBinding()]
param(
    [string] $InnoCompiler = 'C:\Program Files\Inno Setup 7\ISCC.exe',
    [string] $RuntimeDir = (Join-Path $PSScriptRoot '..\build\dist\QwenRagRuntime'),
    [string] $OcrDir = (Join-Path $PSScriptRoot '..\..\models\ocr'),
    [string] $InitialKbDir = (Join-Path $PSScriptRoot '..\payload\initial_kb'),
    [string] $DocumentationDir = (Join-Path $PSScriptRoot '..\release_docs'),
    [string] $ReleaseRoot = (Join-Path $PSScriptRoot '..\..\release'),
    [string] $ExistingInnoOutput,
    [switch] $AllowDirty
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$installer = Join-Path $PSScriptRoot '..\installer\QwenRAG.iss'
if (-not (Test-Path -LiteralPath $InnoCompiler -PathType Leaf)) { throw "Inno Setup compiler is missing: $InnoCompiler" }
foreach ($path in @($RuntimeDir, $OcrDir, $DocumentationDir)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) { throw "Required release asset directory is missing: $path" }
}
$runtimeRoot = (Resolve-Path $RuntimeDir).Path
$ocrRoot = (Resolve-Path $OcrDir).Path
$docsRoot = (Resolve-Path $DocumentationDir).Path
$initialKbRoot = if (Test-Path -LiteralPath $InitialKbDir -PathType Container) { (Resolve-Path $InitialKbDir).Path } else { $InitialKbDir }
$minimumFreeSpaceWithKbMB = $null
$releaseDocuments = @('安装说明.md', '模型部署与适配说明.md', '初始知识库说明.md', '用户使用说明.md', '故障排查手册.md', '客户机实施与验收清单.md', 'deployment.customer.example.json')
foreach ($document in $releaseDocuments) {
    if (-not (Test-Path -LiteralPath (Join-Path $docsRoot $document) -PathType Leaf)) { throw "Required release document is missing: $document" }
}
$runtime = Join-Path $runtimeRoot 'QwenRagRuntime.exe'
if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) { throw 'Frozen runtime executable is missing.' }
$runtimeVersion = (& $runtime version).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($runtimeVersion)) { throw 'Unable to read frozen runtime version.' }
$installerVersion = ([regex]::Match((Get-Content -LiteralPath $installer -Raw), '(?m)^AppVersion=(.+)$')).Groups[1].Value.Trim()
if ($runtimeVersion -ne $installerVersion) { throw "Runtime version $runtimeVersion does not match installer version $installerVersion." }
$commit = (& git -C $projectRoot rev-parse HEAD).Trim()
$dirtyEntries = @(& git -C $projectRoot status --porcelain)
if ($dirtyEntries.Count -gt 0 -and -not $AllowDirty) { throw 'Git worktree is dirty. Use -AllowDirty only for a non-release developer build.' }
if ((Test-Path -LiteralPath $initialKbRoot -PathType Container) -and -not (Test-Path -LiteralPath (Join-Path $initialKbRoot 'SHA256SUMS.txt') -PathType Leaf)) { throw 'Initial knowledge-base payload is missing SHA256SUMS.txt.' }
if (Test-Path -LiteralPath $initialKbRoot -PathType Container) {
    $initialKbBytes = [int64]((Get-ChildItem -LiteralPath $initialKbRoot -Recurse -File | Measure-Object -Property Length -Sum).Sum)
    $runtimeAndOcrBytes = [int64]((@(Get-ChildItem -LiteralPath $runtimeRoot -Recurse -File) + @(Get-ChildItem -LiteralPath $ocrRoot -Recurse -File) | Measure-Object -Property Length -Sum).Sum)
    # One installed copy, temporary installer extraction, and 2 GB operating
    # headroom. The customer data volume is never silently placed on a nearly
    # full system drive.
    $minimumFreeSpaceWithKbMB = [int][Math]::Ceiling((($initialKbBytes * 2) + $runtimeAndOcrBytes + (2GB)) / 1MB)
}

$releaseName = "QwenRAG-$runtimeVersion-offline"
$releaseRootPath = [IO.Path]::GetFullPath($ReleaseRoot)
$releaseTarget = Join-Path $releaseRootPath $releaseName
if (Test-Path -LiteralPath $releaseTarget) { throw "Release directory already exists and will not be overwritten: $releaseTarget" }
$removeInnoOutput = [string]::IsNullOrWhiteSpace($ExistingInnoOutput)
$innoOutput = if ($removeInnoOutput) { Join-Path $projectRoot ("packaging\build\inno-" + [guid]::NewGuid().ToString('N')) } else { (Resolve-Path $ExistingInnoOutput).Path }
$staging = Join-Path $releaseRootPath (".stage-" + [guid]::NewGuid().ToString('N'))
if ($removeInnoOutput) { New-Item -ItemType Directory -Path $innoOutput -Force | Out-Null }
New-Item -ItemType Directory -Path $staging -Force | Out-Null
try {
    if ($removeInnoOutput) {
        $compilerArguments = @(("/O" + $innoOutput), ("/DRuntimeDir=" + $runtimeRoot), ("/DOcrDir=" + $ocrRoot), ("/DInitialKbDir=" + $initialKbRoot))
        if ($null -ne $minimumFreeSpaceWithKbMB) { $compilerArguments += ("/DMinimumFreeSpaceWithKbMB=" + $minimumFreeSpaceWithKbMB) }
        $compilerArguments += $installer
        & $InnoCompiler @compilerArguments
        if ($LASTEXITCODE -ne 0) { throw 'Inno Setup compilation failed.' }
    }
    $media = @(Get-ChildItem -LiteralPath $innoOutput -File | Where-Object { $_.Name -like 'QwenRAG-*-Setup.exe' -or $_.Name -like 'QwenRAG-*-Setup-*.bin' } | Sort-Object Name)
    if ($media.Count -eq 0) { throw 'Inno Setup did not produce installation media.' }
    foreach ($file in $media) { Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $staging $file.Name) }
    foreach ($document in $releaseDocuments) { Copy-Item -LiteralPath (Join-Path $docsRoot $document) -Destination (Join-Path $staging $document) }
    $manifestFiles = @(Get-ChildItem -LiteralPath $staging -File | Sort-Object Name | ForEach-Object { [ordered]@{ name = $_.Name; bytes = $_.Length; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash } })
    [ordered]@{
        product = 'QwenRAG'
        version = $runtimeVersion
        created_at_utc = [DateTime]::UtcNow.ToString('o')
        source_commit = $commit
        source_dirty = ($dirtyEntries.Count -gt 0)
        initial_knowledge_base_included = (Test-Path -LiteralPath $initialKbRoot -PathType Container)
        installation_media = @($media | ForEach-Object { $_.Name })
        files = $manifestFiles
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $staging 'release-manifest.json') -Encoding utf8
    $hashLines = @(Get-ChildItem -LiteralPath $staging -File | Where-Object { $_.Name -ne 'SHA256SUMS.txt' } | Sort-Object Name | ForEach-Object { "$(Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName | Select-Object -ExpandProperty Hash)  $($_.Name)" })
    $hashLines | Set-Content -LiteralPath (Join-Path $staging 'SHA256SUMS.txt') -Encoding utf8
    & (Join-Path $PSScriptRoot 'verify_release.ps1') -ReleaseDirectory $staging -RuntimeDir $runtimeRoot -OcrDir $ocrRoot -InitialKbDir $initialKbRoot -RequireDocumentation
    if (-not $?) { throw 'Release verification failed.' }
    Move-Item -LiteralPath $staging -Destination $releaseTarget
    Write-Output "Release created: $releaseTarget"
} finally {
    if ($removeInnoOutput -and (Test-Path -LiteralPath $innoOutput)) { Remove-Item -LiteralPath $innoOutput -Recurse -Force }
    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
}
