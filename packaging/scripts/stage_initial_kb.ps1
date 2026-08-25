[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $SourceKnowledgeBase,
    [Parameter(Mandatory = $true)][string] $Version,
    [Parameter(Mandatory = $true)][string] $EmbeddingRevision,
    [string] $Python = (Join-Path $PSScriptRoot '..\..\.venv-delivery\Scripts\python.exe'),
    [string] $Destination = (Join-Path $PSScriptRoot '..\payload\initial_kb')
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$source = (Resolve-Path $SourceKnowledgeBase).Path
$pythonPath = (Resolve-Path $Python).Path
if (Test-Path -LiteralPath $Destination) { throw "Snapshot destination already exists and will not be overwritten: $Destination" }

& $pythonPath (Join-Path $projectRoot 'packaging\stage_kb_snapshot.py') --source $source --destination $Destination --version $Version --embedding-revision $EmbeddingRevision
if ($LASTEXITCODE -ne 0) { throw 'Initial knowledge-base snapshot creation failed.' }

foreach ($required in @('SHA256SUMS.txt', 'snapshot.json', 'metadata.db', 'vector_index\index.faiss', 'vector_index\index.meta.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $Destination $required) -PathType Leaf)) {
        throw "Initial knowledge-base snapshot is incomplete: $required"
    }
}
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Destination 'SHA256SUMS.txt') | Format-List
